//! Preliminary surface density sampling scoped to one material context.
use super::{MaterialRuleError, MaterialSurfaceQueries};
use crate::{registry::BlockStateId, worldgen::DecodeError};
use horizon_core::{
    density::{DensitySampler, SampleError, SamplerContext},
    noise::{DensityVolume, Noise},
    position::BlockPos,
};
use std::{collections::BTreeSet, sync::Arc};

/// A lazy unit-spaced XZ plane and scalar fallback for preliminary heights.
/// Construct a fresh provider for each material context, with that world's
/// compiled preliminary-surface density root and bound sampler context.
pub struct MaterialDensitySources<S: MaterialSurfaceQueries> {
    surface: S,
    expected: DensityVolume,
    densities: Vec<BoundDensity>,
    /// Eagerly bound volumes, shared by bindings of the same density root.
    buffers: Vec<EagerBuffer>,
    /// Contiguous y-slice a slice of an eager volume is sampled into before
    /// it is scattered into the volume.
    slice_scratch: Vec<f32>,
    retained_samples: usize,
    plane: DensityVolume,
    sampler: Arc<dyn DensitySampler>,
    context: SamplerContext,
    buffer: Option<Vec<f32>>,
}
struct BoundDensity {
    sampler: Arc<dyn DensitySampler>,
    buffer: Option<usize>,
}
/// A bound density's volume over `expected`, filled one
/// [`DEFAULT_SLICE_Y`](horizon_core::density::DEFAULT_SLICE_Y)-high y slice at
/// a time, when a block in the slice is first read: the same sub-volumes
/// `sample_volume_sliced` evaluates, so the values are identical, but the
/// slices no block asks for (the air above a column's top, and most of a vein
/// richness that only matters where a vein is) are never sampled.
struct EagerBuffer {
    sampler: Arc<dyn DensitySampler>,
    data: Vec<f32>,
    /// Per slice: whether `data` holds its samples.
    sampled: Vec<bool>,
}
const SLICE_Y: usize = horizon_core::density::DEFAULT_SLICE_Y;
fn slice_count(volume: DensityVolume) -> usize {
    volume.sizes()[1].div_ceil(SLICE_Y).max(1)
}
impl<S: MaterialSurfaceQueries> MaterialDensitySources<S> {
    pub fn sampler_context_mut(&mut self) -> &mut SamplerContext {
        &mut self.context
    }
    /// Re-target the per-cell bindings to one block. A carver reuses one worker
    /// across every restored dirt block, so the 1×1×1 volumes the bindings and
    /// the preliminary plane sample must follow the cell; the compiled rule is
    /// kept.
    pub fn retarget(&mut self, position: [i32; 3]) -> Result<(), DecodeError> {
        let volume = DensityVolume::new([1; 3], position, [1; 3])
            .map_err(|_| DecodeError::Field("material density volume"))?;
        let plane = DensityVolume::new([1; 3], [position[0], 0, position[2]], [1; 3])
            .map_err(|_| DecodeError::Field("material preliminary plane"))?;
        self.expected = volume;
        self.plane = plane;
        // The plane is sampled lazily at the new x/z.
        self.buffer = None;
        // So are the bindings' single cells, on their first read.
        let slices = slice_count(volume);
        for eager in &mut self.buffers {
            let out = &mut eager.data;
            if out.capacity() < volume.size() && out.try_reserve(volume.size() - out.len()).is_err()
            {
                horizon_core::alloc::note();
                return Err(DecodeError::Sample(SampleError::Alloc));
            }
            out.resize(volume.size(), 0.0);
            eager.sampled.clear();
            eager.sampled.resize(slices, false);
        }
        Ok(())
    }
    /// Sample the y-slice of an eager volume holding row `y` (relative to
    /// `expected`) if it has not been sampled yet.
    #[inline]
    fn ensure_slice(&mut self, buffer: usize, y: usize) -> Result<(), SampleError> {
        let slice = y / SLICE_Y;
        if self.buffers[buffer].sampled[slice] {
            return Ok(());
        }
        self.sample_slice(buffer, slice)
    }
    #[cold]
    fn sample_slice(&mut self, buffer: usize, slice: usize) -> Result<(), SampleError> {
        let [sx, sy, sz] = self.expected.sizes();
        let eager = &mut self.buffers[buffer];
        if sy <= SLICE_Y {
            eager
                .sampler
                .sample_volume(&mut self.context, &mut eager.data, self.expected)?;
        } else {
            let oy = slice * SLICE_Y;
            let h = SLICE_Y.min(sy - oy);
            let [bx, by, bz] = self.expected.minimum();
            let sub = DensityVolume::new(
                [sx as i32, h as i32, sz as i32],
                [bx, by.wrapping_add(oy as i32), bz],
                self.expected.steps(),
            )?;
            let buf = &mut self.slice_scratch;
            if buf.capacity() < sub.size() && buf.try_reserve(sub.size() - buf.len()).is_err() {
                horizon_core::alloc::note();
                return Err(SampleError::Alloc);
            }
            buf.resize(sub.size(), 0.0);
            eager.sampler.sample_volume(&mut self.context, buf, sub)?;
            for z in 0..sz {
                for x in 0..sx {
                    let src = (x + z * sx) * h;
                    let dst = (x + z * sx) * sy + oy;
                    eager.data[dst..dst + h].copy_from_slice(&buf[src..src + h]);
                }
            }
        }
        eager.sampled[slice] = true;
        Ok(())
    }
    pub fn into_parts(self) -> (S, SamplerContext) {
        (self.surface, self.context)
    }
    pub fn new(
        surface: S,
        expected: DensityVolume,
        sampler: Arc<dyn DensitySampler>,
        context: SamplerContext,
    ) -> Result<Self, DecodeError> {
        let [sx, _, sz] = expected.sizes();
        // Bound retained allocation. Normal chunk planes contain 256 samples.
        if sx * sz > 1 << 20 {
            return Err(DecodeError::Field("material preliminary plane size"));
        }
        let [x, _, z] = expected.minimum();
        let plane = DensityVolume::new([sx as i32, 1, sz as i32], [x, 0, z], [1; 3])
            .map_err(|_| DecodeError::Field("material preliminary plane"))?;
        Ok(Self {
            surface,
            expected,
            densities: Vec::new(),
            buffers: Vec::new(),
            slice_scratch: Vec::new(),
            retained_samples: 0,
            plane,
            sampler,
            context,
            buffer: None,
        })
    }
}
impl<S: MaterialSurfaceQueries> MaterialSurfaceQueries for MaterialDensitySources<S> {
    fn bind_density(
        &mut self,
        sampler: Arc<dyn DensitySampler>,
        eager: bool,
    ) -> Result<usize, DecodeError> {
        if self.densities.len() >= 32768
            || (eager
                && (self.expected.size() > 1 << 20
                    || self.retained_samples + self.expected.size() > 1 << 24))
        {
            return Err(DecodeError::Field("material density storage bound"));
        }
        let shared = self
            .densities
            .iter()
            .find(|bound| bound.buffer.is_some() && Arc::ptr_eq(&bound.sampler, &sampler))
            .and_then(|bound| bound.buffer);
        let buffer = match (eager, shared) {
            (false, _) => None,
            // Sampling a root over the same volume and context is deterministic,
            // so a second binding (iron and copper veins' richness) reuses it.
            (true, Some(buffer)) => Some(buffer),
            (true, None) => {
                let mut buffer = Vec::new();
                if buffer.try_reserve_exact(self.expected.size()).is_err() {
                    horizon_core::alloc::note();
                    return Err(DecodeError::Sample(SampleError::Alloc));
                }
                buffer.resize(self.expected.size(), 0.0);
                self.retained_samples += buffer.len();
                self.buffers.push(EagerBuffer {
                    sampler: sampler.clone(),
                    data: buffer,
                    sampled: vec![false; slice_count(self.expected)],
                });
                Some(self.buffers.len() - 1)
            }
        };
        let index = self.densities.len();
        self.densities.push(BoundDensity { sampler, buffer });
        Ok(index)
    }
    fn density(&mut self, index: usize, position: BlockPos) -> Result<f32, MaterialRuleError> {
        let source = self
            .densities
            .get(index)
            .ok_or(MaterialRuleError::MissingDensityProvider)?;
        let position = [position.x, position.y, position.z];
        if let Some(buffer) = source.buffer {
            if let Some(index) = block_index(self.expected, position) {
                let y = position[1].wrapping_sub(self.expected.minimum()[1]) as u32 as usize;
                self.ensure_slice(buffer, y)?;
                return Ok(self.buffers[buffer].data[index]);
            }
        }
        Ok(source.sampler.sample_value(&mut self.context, position)?)
    }
    fn prepare_density(&mut self, index: usize) -> Result<(), MaterialRuleError> {
        if let Some(buffer) = self.densities.get(index).and_then(|bound| bound.buffer) {
            self.ensure_slice(buffer, 0)?;
        }
        Ok(())
    }
    fn density_nonpositive(
        &mut self,
        index: usize,
        x: i32,
        z: i32,
        y_lo: i32,
        y_hi: i32,
    ) -> Result<bool, MaterialRuleError> {
        let Some(buffer) = self.densities.get(index).and_then(|bound| bound.buffer) else {
            return Ok(false);
        };
        // A column run inside the volume: one y-fastest span of its buffer.
        if self.expected.steps() != [1; 3] {
            return Ok(false);
        }
        let (Some(first), Some(last)) = (
            block_index(self.expected, [x, y_lo, z]),
            block_index(self.expected, [x, y_hi, z]),
        ) else {
            return Ok(false);
        };
        if first > last {
            return Ok(false);
        }
        let my = self.expected.minimum()[1];
        let slices = (y_lo.wrapping_sub(my) as u32 as usize) / SLICE_Y
            ..=(y_hi.wrapping_sub(my) as u32 as usize) / SLICE_Y;
        for slice in slices {
            self.ensure_slice(buffer, slice * SLICE_Y)?;
        }
        // NaN fails `<=`: a vein that can see one is not ruled out.
        Ok(self.buffers[buffer].data[first..=last]
            .iter()
            .all(|&density| density <= 0.0))
    }
    fn preliminary_surface(&mut self, position: BlockPos) -> Result<f32, MaterialRuleError> {
        let position = [position.x, 0, position.z];
        if let Some(index) = self.plane.index_of_block(position) {
            if self.buffer.is_none() {
                let mut buffer = Vec::new();
                if buffer.try_reserve_exact(self.plane.size()).is_err() {
                    horizon_core::alloc::note();
                    return Err(MaterialRuleError::Density(SampleError::Alloc));
                }
                buffer.resize(self.plane.size(), 0.0);
                // Retain the allocation and partial writes if sampling fails.
                let buffer = self.buffer.insert(buffer);
                self.sampler
                    .sample_volume(&mut self.context, buffer, self.plane)?;
            }
            Ok(self.buffer.as_ref().unwrap()[index])
        } else {
            Ok(self.sampler.sample_value(&mut self.context, position)?)
        }
    }
    fn possible_biomes(&self) -> Option<BTreeSet<String>> {
        self.surface.possible_biomes()
    }
    fn biome(&mut self, position: BlockPos) -> Result<Option<Arc<str>>, MaterialRuleError> {
        self.surface.biome(position)
    }
    fn biome_id(&mut self, position: BlockPos) -> Result<Option<u32>, MaterialRuleError> {
        self.surface.biome_id(position)
    }
    fn biome_index(&self, name: &str) -> Option<Option<u32>> {
        self.surface.biome_index(name)
    }
    fn melts_frozen_ocean(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        self.surface.melts_frozen_ocean(biome, position, sea_level)
    }
    fn cold_enough_to_snow(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        self.surface.cold_enough_to_snow(biome, position, sea_level)
    }
    fn bind_noise(&mut self, name: &str) -> Result<Arc<dyn Noise>, DecodeError> {
        self.surface.bind_noise(name)
    }
    fn surface_depth(&mut self, x: i32, z: i32) -> i32 {
        self.surface.surface_depth(x, z)
    }
    fn surface_secondary(&mut self, x: i32, z: i32) -> f64 {
        self.surface.surface_secondary(x, z)
    }
    fn band(&mut self, position: BlockPos) -> Result<BlockStateId, MaterialRuleError> {
        self.surface.band(position)
    }
}

/// `DensityVolume::index_of_block` without its per-axis divisions for the
/// unit-step volumes the material stage samples (one or two lookups per block).
#[inline]
fn block_index(volume: DensityVolume, position: [i32; 3]) -> Option<usize> {
    // Not `steps() != [1; 3]`: 32-bit ARM compares arrays through `memcmp`.
    let [step_x, step_y, step_z] = volume.steps();
    if step_x != 1 || step_y != 1 || step_z != 1 {
        return volume.index_of_block(position);
    }
    let [mx, my, mz] = volume.minimum();
    let [sx, sy, sz] = volume.sizes();
    let x = position[0].wrapping_sub(mx) as u32 as usize;
    let y = position[1].wrapping_sub(my) as u32 as usize;
    let z = position[2].wrapping_sub(mz) as u32 as usize;
    (x < sx && y < sy && z < sz).then(|| y + (x + z * sx) * sy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::density::SampleError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Surface;
    impl MaterialSurfaceQueries for Surface {
        fn surface_depth(&mut self, _: i32, _: i32) -> i32 {
            0
        }
        fn surface_secondary(&mut self, _: i32, _: i32) -> f64 {
            0.0
        }
    }
    /// Position-dependent values; counts bulk samples.
    #[derive(Debug, Default)]
    struct Counted(AtomicUsize);
    impl DensitySampler for Counted {
        fn sample_value(&self, _: &mut SamplerContext, p: [i32; 3]) -> Result<f32, SampleError> {
            Ok((p[0] * 10_000 + p[1] * 100 + p[2]) as f32)
        }
        fn sample_volume(
            &self,
            context: &mut SamplerContext,
            buffer: &mut [f32],
            volume: DensityVolume,
        ) -> Result<(), SampleError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            for z in 0..volume.sizes()[2] {
                for x in 0..volume.sizes()[0] {
                    for y in 0..volume.sizes()[1] {
                        let [mx, my, mz] = volume.minimum();
                        let p = [mx + x as i32, my + y as i32, mz + z as i32];
                        buffer[volume.index([x, y, z]).unwrap()] = self.sample_value(context, p)?;
                    }
                }
            }
            Ok(())
        }
    }
    fn sources() -> MaterialDensitySources<Surface> {
        let volume = DensityVolume::new([16, 24, 16], [32, -64, -16], [1; 3]).unwrap();
        MaterialDensitySources::new(
            Surface,
            volume,
            Arc::new(Counted::default()),
            SamplerContext::default(),
        )
        .unwrap()
    }

    #[test]
    fn eager_bindings_of_one_root_share_a_single_sample() {
        let mut sources = sources();
        let shared = Arc::new(Counted::default());
        let other = Arc::new(Counted::default());
        let a = sources.bind_density(shared.clone(), true).unwrap();
        let lazy = sources.bind_density(shared.clone(), false).unwrap();
        let b = sources.bind_density(shared.clone(), true).unwrap();
        let c = sources.bind_density(other.clone(), true).unwrap();
        // Eager bindings sample a y-slice when a block in it is first read.
        assert_eq!(shared.0.load(Ordering::Relaxed), 0);
        assert_eq!(other.0.load(Ordering::Relaxed), 0);
        assert_eq!([a, lazy, b, c], [0, 1, 2, 3]);
        for p in [[32, -64, -16], [47, -41, -1], [40, -50, -9], [48, 0, 0]] {
            let position = BlockPos::new(p[0], p[1], p[2]);
            let expected = (p[0] * 10_000 + p[1] * 100 + p[2]) as f32;
            for index in [a, lazy, b, c] {
                assert_eq!(sources.density(index, position).unwrap(), expected);
            }
        }
        assert_eq!(shared.0.load(Ordering::Relaxed), 1);
        assert_eq!(other.0.load(Ordering::Relaxed), 1);
    }

    /// Slices are sampled on demand over exactly the sub-volumes of
    /// `sample_volume_sliced`, whatever order they are read in.
    #[test]
    fn eager_slices_are_sampled_on_first_read() {
        let height = horizon_core::density::DEFAULT_SLICE_Y * 2 + 8;
        let volume = DensityVolume::new([16, height as i32, 16], [32, -64, -16], [1; 3]).unwrap();
        let mut sources = MaterialDensitySources::new(
            Surface,
            volume,
            Arc::new(Counted::default()),
            SamplerContext::default(),
        )
        .unwrap();
        let counted = Arc::new(Counted::default());
        let eager = sources.bind_density(counted.clone(), true).unwrap();
        let at = |y: usize| BlockPos::new(35, -64 + y as i32, -10);
        let want = |y: usize| (35 * 10_000 + (-64 + y as i32) * 100 - 10) as f32;
        let top = horizon_core::density::DEFAULT_SLICE_Y * 2 + 3;
        for y in [
            top,
            top + 1,
            5,
            0,
            horizon_core::density::DEFAULT_SLICE_Y + 7,
            top,
        ] {
            assert_eq!(sources.density(eager, at(y)).unwrap(), want(y));
        }
        assert_eq!(counted.0.load(Ordering::Relaxed), 3);
        // The slice a read lands in is whole: other columns of it came with it.
        let other = BlockPos::new(47, -64 + 9, 0);
        assert_eq!(
            sources.density(eager, other).unwrap(),
            (47 * 10_000 + (-64 + 9) * 100) as f32
        );
        assert_eq!(counted.0.load(Ordering::Relaxed), 3);
        // Retargeting re-samples the single cell on its first read.
        sources.retarget([40, 12, 3]).unwrap();
        assert_eq!(counted.0.load(Ordering::Relaxed), 3);
        assert_eq!(
            sources.density(eager, BlockPos::new(40, 12, 3)).unwrap(),
            (40 * 10_000 + 12 * 100 + 3) as f32
        );
        assert_eq!(counted.0.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn block_index_matches_index_of_block() {
        let unit = DensityVolume::new([16, 24, 16], [i32::MAX - 7, -64, -16], [1; 3]).unwrap();
        let strided = DensityVolume::new([5, 3, 5], [-8, -64, 0], [4, 8, 4]).unwrap();
        for volume in [unit, strided] {
            let [mx, my, mz] = volume.minimum();
            for dx in [-1, 0, 1, 4, 7, 8, 15, 16, 17, 20] {
                for dy in [-9, -1, 0, 8, 16, 23, 24] {
                    for dz in [-1, 0, 3, 15, 16, 20] {
                        let p = [
                            mx.wrapping_add(dx),
                            my.wrapping_add(dy),
                            mz.wrapping_add(dz),
                        ];
                        assert_eq!(block_index(volume, p), volume.index_of_block(p), "{p:?}");
                    }
                }
            }
            for p in [
                [i32::MIN, 0, 0],
                [i32::MAX, i32::MAX, i32::MAX],
                [0, i32::MIN, 0],
            ] {
                assert_eq!(block_index(volume, p), volume.index_of_block(p), "{p:?}");
            }
        }
    }
}
