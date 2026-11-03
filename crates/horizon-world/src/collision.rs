//! Version-pinned cached block collision data, without entity-context behavior.
//!
//! Java's no-context BlockState.getCollisionShape uses its cache when present.
//! The overload taking CollisionContext always calls block behavior instead.
//! This module covers the cached branch only; uncached states remain explicit.
use crate::{
    DATA_VERSION,
    chunk::BlockSection,
    compression::invalid,
    nbt::Tag,
    registry::{BlockRegistry, BlockStateId, BlockTable},
};
use horizon_core::geometry::{Axis, VoxelShape};
use std::{collections::HashMap, io};

/// `PipeBlock`'s shape for `size` pixels across (`makeShapes`): the core
/// cube, with an arm from the centre out to each joined side (`sides` by
/// `Direction`: down, up, north, south, west, east).
pub fn pipe_shape(size: f64, sides: [bool; 6]) -> io::Result<VoxelShape> {
    use horizon_core::geometry::{Aabb, ShapeOp, Vec3};
    let (lo, hi) = ((8.0 - size / 2.0) / 16.0, (8.0 + size / 2.0) / 16.0);
    let shape = |min: [f64; 3], max: [f64; 3]| {
        VoxelShape::from_box(Aabb::new(
            Vec3::new(min[0], min[1], min[2]),
            Vec3::new(max[0], max[1], max[2]),
        ))
        .map_err(|e| invalid(e.to_string()))
    };
    let mut result = shape([lo; 3], [hi; 3])?;
    let arms = [
        ([lo, 0.0, lo], [hi, 0.5, hi]),
        ([lo, 0.5, lo], [hi, 1.0, hi]),
        ([lo, lo, 0.0], [hi, hi, 0.5]),
        ([lo, lo, 0.5], [hi, hi, 1.0]),
        ([0.0, lo, lo], [0.5, hi, hi]),
        ([0.5, lo, lo], [1.0, hi, hi]),
    ];
    for (joined, (min, max)) in sides.into_iter().zip(arms) {
        if joined {
            result = shape(min, max)?
                .join(&result, ShapeOp::OR)
                .map_err(|e| invalid(e.to_string()))?;
        }
    }
    Ok(result)
}

#[derive(PartialEq, Eq, Hash)]
struct GridKey {
    coords: [Vec<u64>; 3],
    occupied: Vec<u8>,
    cube: bool,
    axes: u8,
    min: [i32; 3],
    max: [i32; 3],
}

/// A registry entry's semantic content — what `from_table` reads out of a
/// `Tag` — serializable so a prepare-time bake can hand the table to a
/// small-heap target without the NBT `Tag` tree.
pub struct BakedCollisionEntry {
    /// `canonical` NBT field; `None` marks "absent" (sets
    /// `canonical_identities` false, same as `from_table`).
    pub canonical: Option<u8>,
    /// `x`/`y`/`z` LongArrays decoded as f64 (bit order preserved).
    pub coords: [Vec<f64>; 3],
    /// `occupied` ByteArray.
    pub occupied: Vec<u8>,
    /// `cube` Byte flag.
    pub cube: bool,
    /// `cube_axes` Byte (0..=7).
    pub cube_axes: u8,
    /// `discrete_min`/`discrete_max` IntArrays.
    pub min: [i32; 3],
    pub max: [i32; 3],
}

/// `from_table`/`from_baked`'s shared state while the `shapes` list builds:
/// the dedup'd shapes themselves, the canonical-identity bookkeeping and
/// the grid map.
struct ShapeTables {
    shapes: Vec<VoxelShape>,
    canonical_identities: bool,
    seen_canonical: [bool; 3],
    grids: HashMap<GridKey, VoxelShape>,
}
impl ShapeTables {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            shapes: Vec::with_capacity(capacity),
            canonical_identities: true,
            seen_canonical: [false; 3],
            grids: HashMap::new(),
        }
    }
    /// One `shapes` entry's construction — identical whether the entry was
    /// decoded out of NBT `Tag`s (`from_table`) or off a prepare-time bake
    /// (`from_baked`): the canonical-identity bookkeeping, the `GridKey`
    /// dedupe, the shape build and the push.
    fn push(&mut self, entry: BakedCollisionEntry) -> io::Result<()> {
        let canonical = match entry.canonical {
            None => {
                self.canonical_identities = false;
                0
            }
            Some(value @ 0..=2) => usize::from(value),
            Some(_) => return Err(invalid("invalid canonical collision identity")),
        };
        if canonical != 0 {
            if self.seen_canonical[canonical] {
                return Err(invalid("duplicate canonical collision identity"));
            }
            self.seen_canonical[canonical] = true;
        }
        if entry.occupied.iter().any(|b| *b != 0 && *b != 1) {
            return Err(invalid("invalid collision occupancy"));
        }
        let occupied = entry.occupied.iter().map(|b| *b != 0).collect::<Vec<_>>();
        if entry.cube_axes > 7 {
            return Err(invalid("invalid cube-coordinate metadata"));
        }
        let coordinate_cubes = [
            entry.cube_axes & 1 != 0,
            entry.cube_axes & 2 != 0,
            entry.cube_axes & 4 != 0,
        ];
        let key = GridKey {
            coords: std::array::from_fn(|i| {
                entry.coords[i].iter().map(|v| v.to_bits()).collect()
            }),
            occupied: entry.occupied,
            cube: entry.cube,
            axes: entry.cube_axes,
            min: entry.min,
            max: entry.max,
        };
        let shape = if let Some(existing) = self.grids.get(&key) {
            existing.fresh_identity()
        } else {
            match entry.cube {
                false => VoxelShape::from_mixed_grid(entry.coords, &occupied, coordinate_cubes),
                true => {
                    if entry.cube_axes != 7 {
                        return Err(invalid("cube shape has array-coordinate metadata"));
                    }
                    let sizes = std::array::from_fn(|i| entry.coords[i].len().saturating_sub(1));
                    let shape = VoxelShape::from_cube(sizes, &occupied)
                        .map_err(|e| invalid(e.to_string()))?;
                    for axis in Axis::ALL {
                        if shape
                            .coordinates(axis)
                            .iter()
                            .map(|v| v.to_bits())
                            .ne(entry.coords[axis as usize].iter().map(|v| v.to_bits()))
                        {
                            return Err(invalid("cube collision coordinates differ"));
                        }
                    }
                    Ok(shape)
                }
            }
            .map_err(|e| invalid(e.to_string()))?
            .with_discrete_bounds(entry.min, entry.max)
            .map_err(|e| invalid(e.to_string()))?
        };
        let shape = if canonical == 0 {
            shape
        } else {
            let reference = if canonical == 1 {
                VoxelShape::empty()
            } else {
                VoxelShape::block()
            };
            let expected_count = usize::from(canonical == 2);
            if entry.cube != reference.is_cube()
                || coordinate_cubes != reference.cube_coordinate_axes()
                || occupied != vec![true; expected_count]
                || entry.min != [0; 3]
                || entry.max != [expected_count as i32; 3]
                || Axis::ALL.into_iter().any(|axis| {
                    key.coords[axis as usize]
                        .iter()
                        .copied()
                        .ne(reference.coordinates(axis).iter().map(|v| v.to_bits()))
                })
            {
                return Err(invalid(
                    "canonical collision identity has different geometry",
                ));
            }
            reference
        };
        self.grids.entry(key).or_insert_with(|| shape.clone());
        self.shapes.push(shape);
        Ok(())
    }
    /// `cached_shapes` -> the finished registry, bounds-checked against it
    /// the way `from_table` checked the IntArray.
    fn finish<'r>(
        self,
        registry: &'r BlockRegistry,
        state_shapes: Vec<i32>,
    ) -> io::Result<CachedCollisionRegistry<'r>> {
        if state_shapes.len() != registry.len()
            || state_shapes
                .iter()
                .any(|id| *id < -1 || *id >= self.shapes.len() as i32)
        {
            return Err(invalid("invalid cached shape mapping"));
        }
        Ok(CachedCollisionRegistry {
            registry,
            shapes: self.shapes,
            state_shapes,
            grid_count: self.grids.len(),
            canonical_identities: self.canonical_identities,
        })
    }
}

pub struct CachedCollisionRegistry<'r> {
    registry: &'r BlockRegistry,
    shapes: Vec<VoxelShape>,
    state_shapes: Vec<i32>,
    grid_count: usize,
    canonical_identities: bool,
}
impl<'r> CachedCollisionRegistry<'r> {
    pub fn registry(&self) -> &'r BlockRegistry {
        self.registry
    }
    pub fn from_nbt(tag: &Tag, registry: &'r BlockRegistry) -> io::Result<Self> {
        Self::from_table(BlockTable::check(tag, registry, "collision")?)
    }
    pub fn from_table(table: BlockTable<'_, 'r>) -> io::Result<Self> {
        let (tag, registry) = (table.tag(), table.registry());
        if tag.get("DataVersion") != Some(&Tag::Int(DATA_VERSION)) {
            return Err(invalid("collision DataVersion mismatch"));
        }
        let Some(Tag::List(entries)) = tag.get("shapes") else {
            return Err(invalid("missing collision shapes"));
        };
        if entries.len() > registry.len() {
            return Err(invalid("too many cached collision shapes"));
        }
        let mut tables = ShapeTables::with_capacity(entries.len());
        for entry in entries {
            let canonical = match entry.get("canonical") {
                None => None,
                Some(Tag::Byte(value @ 0..=2)) => Some(*value as u8),
                _ => return Err(invalid("invalid canonical collision identity")),
            };
            let mut coords: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for (i, axis) in ["x", "y", "z"].into_iter().enumerate() {
                let Some(Tag::LongArray(bits)) = entry.get(axis) else {
                    return Err(invalid("missing collision coordinates"));
                };
                coords[i] = bits.iter().map(|b| f64::from_bits(*b as u64)).collect();
            }
            let Some(Tag::ByteArray(occupied)) = entry.get("occupied") else {
                return Err(invalid("missing collision occupancy"));
            };
            if occupied.iter().any(|b| *b != 0 && *b != 1) {
                return Err(invalid("invalid collision occupancy"));
            }
            let Some(Tag::Byte(cube_axes)) = entry.get("cube_axes") else {
                return Err(invalid("missing cube-coordinate metadata"));
            };
            if *cube_axes < 0 || *cube_axes > 7 {
                return Err(invalid("invalid cube-coordinate metadata"));
            }
            let discrete_bounds = |name| -> io::Result<[i32; 3]> {
                let Some(Tag::IntArray(values)) = entry.get(name) else {
                    return Err(invalid("missing collision discrete extrema"));
                };
                values
                    .as_slice()
                    .try_into()
                    .map_err(|_| invalid("invalid collision discrete extrema length"))
            };
            let min = discrete_bounds("discrete_min")?;
            let max = discrete_bounds("discrete_max")?;
            let cube = match entry.get("cube") {
                Some(Tag::Byte(0)) => false,
                Some(Tag::Byte(1)) => true,
                _ => return Err(invalid("invalid collision cube flag")),
            };
            tables.push(BakedCollisionEntry {
                canonical,
                coords,
                occupied: occupied.clone(),
                cube,
                cube_axes: *cube_axes as u8,
                min,
                max,
            })?;
        }
        let Some(Tag::IntArray(state_shapes)) = tag.get("cached_shapes") else {
            return Err(invalid("missing cached shape mapping"));
        };
        tables.finish(registry, state_shapes.clone())
    }
    /// `from_table` for a prepare-time postcard bake of the same table:
    /// `entries` are `shapes`, `state_shapes` is `cached_shapes`. Replays the
    /// same construction (grid dedupe, canonical identity checks, discrete
    /// bounds) so the result is bit-identical to the NBT path's registry.
    pub fn from_baked(
        registry: &'r BlockRegistry,
        entries: Vec<BakedCollisionEntry>,
        state_shapes: Vec<i32>,
    ) -> io::Result<Self> {
        if entries.len() > registry.len() {
            return Err(invalid("too many cached collision shapes"));
        }
        let mut tables = ShapeTables::with_capacity(entries.len());
        for entry in entries {
            tables.push(entry)?;
        }
        tables.finish(registry, state_shapes)
    }
    /// Older exports omit native canonical identities. Such imports retain their
    /// local alias graph but cannot support identity-dependent world iteration.
    pub fn has_canonical_identities(&self) -> bool {
        self.canonical_identities
    }
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }
    pub fn unique_grid_count(&self) -> usize {
        self.grid_count
    }
    pub fn cached_state_count(&self) -> usize {
        self.state_shapes.iter().filter(|id| **id >= 0).count()
    }
    /// None means vanilla must call block behavior for this state. It does not
    /// mean air, an empty shape, or permission to ignore a collision.
    pub fn cached_shape(&self, state: BlockStateId) -> io::Result<Option<&VoxelShape>> {
        let id = *self
            .state_shapes
            .get(state.0 as usize)
            .ok_or_else(|| invalid("collision state outside registry"))?;
        Ok(if id < 0 {
            None
        } else {
            Some(&self.shapes[id as usize])
        })
    }
    pub fn section_shape(
        &self,
        section: &BlockSection<'_>,
        index: usize,
    ) -> io::Result<Option<&VoxelShape>> {
        if !std::ptr::eq(section.registry(), self.registry) {
            return Err(invalid("section uses a different collision registry"));
        }
        self.cached_shape(section.get(index)?)
    }
}

#[cfg(test)]
mod pipe_tests {
    use super::*;
    use crate::registry::BlockTableFile;
    use horizon_core::geometry::ShapeOp;
    use std::path::PathBuf;

    /// `pipe_shape` at the chorus plant's 10 pixels is vanilla's own shape for
    /// each of its 64 states.
    #[test]
    fn pipe_shapes_match_the_chorus_plant() {
        let root = std::env::var_os("HORIZON_RESOURCES")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../.local/horizon-client")
                    .join(horizon_core::VANILLA_VERSION)
            });
        if !root.join("cached-collision.nbt").exists() {
            eprintln!("skipping: no resources at {}", root.display());
            return;
        }
        let registry = crate::level::data::load_registry(&root).unwrap();
        let file = BlockTableFile::open(&root.join("cached-collision.nbt"), &registry).unwrap();
        let cached = CachedCollisionRegistry::from_table(file.table("collision").unwrap()).unwrap();
        let mut checked = 0;
        for id in 0..registry.len() as u32 {
            let state = BlockStateId(id);
            let s = registry.state(state).unwrap();
            if s.name != "minecraft:chorus_plant" {
                continue;
            }
            let sides = ["down", "up", "north", "south", "west", "east"]
                .map(|k| s.properties.get(k).map(String::as_str) == Some("true"));
            let ours = pipe_shape(10.0, sides).unwrap();
            let theirs = cached.cached_shape(state).unwrap().unwrap();
            assert!(
                !ours.join_is_not_empty(theirs, ShapeOp::XOR).unwrap(),
                "{sides:?}"
            );
            assert_eq!(ours.boxes(), theirs.boxes(), "{sides:?}");
            checked += 1;
        }
        assert_eq!(checked, 64);
    }
}
