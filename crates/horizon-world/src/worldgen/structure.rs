//! Structure templates, processors and placement shared by template features
//! and structure pieces.
pub mod end_city;
pub mod enhanced_terrain;
pub mod features;
pub mod formations;
pub mod fortress;
pub mod jigsaw;
pub mod legacy;
pub mod mansion;
pub mod mineshaft;
pub mod monument;
pub mod nether_fossil;
pub mod ocean_ruin;
pub mod piece_limited;
pub mod processor;
pub mod ruined_portal;
pub mod simple;
pub mod start;
pub mod start_checks;
pub mod stronghold;
pub mod template;
pub mod template_meta;
pub mod temple;

use crate::{
    nbt::{self, Limits},
    registry::{BlockRegistry, BlockStateId},
    worldgen::generator::PackedWorldgen,
};
use horizon_core::geometry::Direction;
use processor::ProcessorList;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{AtomicU32, AtomicUsize, Ordering},
    },
};
pub use template::{BlockInfo, EntityInfo, JigsawInfo, StructureTemplate, save_nbt};
pub use template_meta::TemplateMeta;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation {
    None,
    Clockwise90,
    Clockwise180,
    Counterclockwise90,
}
impl Rotation {
    pub const ALL: [Self; 4] = [
        Self::None,
        Self::Clockwise90,
        Self::Clockwise180,
        Self::Counterclockwise90,
    ];
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Self::None,
            "clockwise_90" => Self::Clockwise90,
            "180" => Self::Clockwise180,
            "counterclockwise_90" => Self::Counterclockwise90,
            _ => return None,
        })
    }
    /// Rotation.getRandom: one bounded draw over the four values.
    pub fn random(random: &mut impl horizon_core::random::RandomSource) -> Self {
        Self::ALL[random.next_int_bound(4).unwrap_or(0) as usize]
    }
    pub fn rotate_direction(self, direction: Direction) -> Direction {
        if matches!(direction, Direction::Up | Direction::Down) {
            return direction;
        }
        let order = [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ];
        let index = order.iter().position(|d| *d == direction).unwrap();
        order[(index + self as usize) % 4]
    }
    /// `Rotation.getSerializedName`: the value of `Rotation.CODEC`.
    pub fn serialized_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Clockwise90 => "clockwise_90",
            Self::Clockwise180 => "180",
            Self::Counterclockwise90 => "counterclockwise_90",
        }
    }
    pub fn get_rotated(self, other: Self) -> Self {
        Self::ALL[(self as usize + other as usize) % 4]
    }
    /// `Rotation.name()`, as structure pieces save it.
    pub fn saved_name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Clockwise90 => "CLOCKWISE_90",
            Self::Clockwise180 => "CLOCKWISE_180",
            Self::Counterclockwise90 => "COUNTERCLOCKWISE_90",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mirror {
    None,
    LeftRight,
    FrontBack,
}
impl Mirror {
    /// `Mirror.name()`, as structure pieces save it.
    pub fn saved_name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::LeftRight => "LEFT_RIGHT",
            Self::FrontBack => "FRONT_BACK",
        }
    }
}

/// Inclusive integer box (vanilla BoundingBox).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoundingBox {
    pub min: [i32; 3],
    pub max: [i32; 3],
}
impl BoundingBox {
    pub fn new(min: [i32; 3], max: [i32; 3]) -> Self {
        Self { min, max }
    }
    pub fn from_corners(a: [i32; 3], b: [i32; 3]) -> Self {
        Self {
            min: std::array::from_fn(|i| a[i].min(b[i])),
            max: std::array::from_fn(|i| a[i].max(b[i])),
        }
    }
    pub fn moved(self, d: [i32; 3]) -> Self {
        Self {
            min: std::array::from_fn(|i| self.min[i] + d[i]),
            max: std::array::from_fn(|i| self.max[i] + d[i]),
        }
    }
    pub fn is_inside(&self, p: [i32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }
    pub fn intersects(&self, other: &Self) -> bool {
        (0..3).all(|i| self.max[i] >= other.min[i] && self.min[i] <= other.max[i])
    }
    pub fn encapsulate(&mut self, other: &Self) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }
    pub fn inflated(self, xz: i32, y: i32) -> Self {
        Self {
            min: [self.min[0] - xz, self.min[1] - y, self.min[2] - xz],
            max: [self.max[0] + xz, self.max[1] + y, self.max[2] + xz],
        }
    }
    pub fn span(&self) -> [i32; 3] {
        std::array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }
    pub fn center(&self) -> [i32; 3] {
        std::array::from_fn(|i| self.min[i] + (self.max[i] - self.min[i] + 1) / 2)
    }
    pub fn corners(&self) -> [[i32; 3]; 8] {
        let (a, b) = (self.min, self.max);
        [
            [a[0], a[1], a[2]],
            [a[0], a[1], b[2]],
            [a[0], b[1], a[2]],
            [a[0], b[1], b[2]],
            [b[0], a[1], a[2]],
            [b[0], a[1], b[2]],
            [b[0], b[1], a[2]],
            [b[0], b[1], b[2]],
        ]
    }
}

/// Per-state BlockState.rotate/mirror results and template ordering flags,
/// exported from vanilla (`blocks/structure-blocks.json`).
#[derive(Debug, Default)]
pub struct StateTransforms {
    rotate: Vec<[u32; 3]>,
    mirror: Vec<[u32; 2]>,
    full: Vec<bool>,
    loot: BTreeSet<BlockStateId>,
    /// Block entity type of every state that has one (`BlockState.hasBlockEntity`).
    entity_types: HashMap<BlockStateId, Arc<str>>,
}
impl StateTransforms {
    pub fn from_json(json: &str, states: usize) -> Result<Self, super::DecodeError> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let arr = |key: &'static str| {
            value[key]
                .as_array()
                .cloned()
                .ok_or(super::DecodeError::Field(key))
        };
        let ids = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as u32)
                        .collect()
                })
                .unwrap_or_default()
        };
        let rotate = arr("rotate")?
            .iter()
            .map(|v| {
                let v = ids(v);
                [v[0], v[1], v[2]]
            })
            .collect::<Vec<_>>();
        let mirror = arr("mirror")?
            .iter()
            .map(|v| {
                let v = ids(v);
                [v[0], v[1]]
            })
            .collect::<Vec<_>>();
        let full = arr("full_block")?
            .iter()
            .map(|v| v.as_bool().unwrap_or(false))
            .collect::<Vec<_>>();
        if rotate.len() != states || mirror.len() != states || full.len() != states {
            return Err(super::DecodeError::Field("structure block table size"));
        }
        let loot = ids(&value["loot_containers"])
            .into_iter()
            .map(BlockStateId)
            .collect();
        let mut entity_types = HashMap::new();
        if let Some(types) = value["block_entities"].as_object() {
            for (name, states) in types {
                let name: Arc<str> = Arc::from(name.as_str());
                for id in ids(states) {
                    entity_types.insert(BlockStateId(id), name.clone());
                }
            }
        }
        Ok(Self {
            rotate,
            mirror,
            full,
            loot,
            entity_types,
        })
    }
    /// The block entity type a state carries, when it has one.
    pub fn entity_type(&self, state: BlockStateId) -> Option<&Arc<str>> {
        self.entity_types.get(&state)
    }
    pub fn rotate(&self, state: BlockStateId, rotation: Rotation) -> BlockStateId {
        match rotation {
            Rotation::None => state,
            r => self
                .rotate
                .get(state.0 as usize)
                .map_or(state, |t| BlockStateId(t[r as usize - 1])),
        }
    }
    pub fn mirror(&self, state: BlockStateId, mirror: Mirror) -> BlockStateId {
        match mirror {
            Mirror::None => state,
            m => self
                .mirror
                .get(state.0 as usize)
                .map_or(state, |t| BlockStateId(t[m as usize - 1])),
        }
    }
    pub fn full_block(&self, state: BlockStateId) -> bool {
        self.full.get(state.0 as usize).copied().unwrap_or(false)
    }
    pub fn loot_container(&self, state: BlockStateId) -> bool {
        self.loot.contains(&state)
    }
}

/// Bound on the decoded templates `template()` keeps for re-use: the
/// biggest closure one structure's pools can touch in a single assemble
/// (trial chambers, ~180 entries and ~11 MB; a village ~120) plus a
/// second structure assembling alongside on another worker. A miss reloads
/// through the same cold path the first touch used, so evicting an entry
/// changes residency only, never what `template()` returns.
/// Bound on the metas `template_meta` keeps: more than every template of
/// the vanilla data (~1,500), so it never evicts in practice; a map past it
/// starts over rather than growing.
const META_LIMIT: usize = 2048;
const TEMPLATE_CACHE_LIMIT: usize = 512;
const TEMPLATE_CACHE_BYTES: usize = 24 * 1024 * 1024;
/// The limits in force (entries, bytes) — [`set_template_cache_limits`]
/// lowers them for a host with a small heap.
static TEMPLATE_LIMITS: [AtomicUsize; 2] = [
    AtomicUsize::new(TEMPLATE_CACHE_LIMIT),
    AtomicUsize::new(TEMPLATE_CACHE_BYTES),
];
/// Bound the decoded-template cache at `entries` templates and `bytes` of
/// decoded size (both process-wide, taking effect on the next insert). A
/// pool element only keeps a template's id (`jigsaw::TemplateRef`), so this
/// cache is the only residency a structure's templates have: a small bound
/// reloads templates the assembly touches again, never changes a result.
pub fn set_template_cache_limits(entries: usize, bytes: usize) {
    TEMPLATE_LIMITS[0].store(entries.max(1), Ordering::Relaxed);
    TEMPLATE_LIMITS[1].store(bytes, Ordering::Relaxed);
}
/// The chunk radius a start-cache shard keeps when it fills (0 = never
/// prune, the default: a server's players can be far apart).
static START_PRUNE_RADIUS: AtomicU32 = AtomicU32::new(0);
/// Let a full start-cache shard drop the starts farther than `radius`
/// chunks from the chunk being inserted, instead of growing to its 64k
/// entries. A single-window host (one player; the window and a chunk's
/// 8-chunk reference reach fit well inside) passes a radius past that; a
/// start needed again is rebuilt, so only a start whose chunks are still
/// being generated across the prune could lose placement state — keep the
/// radius generous.
pub fn set_start_cache_prune_radius(radius: u32) {
    START_PRUNE_RADIUS.store(radius, Ordering::Relaxed);
}
pub(super) fn start_cache_prune_radius() -> i32 {
    START_PRUNE_RADIUS.load(Ordering::Relaxed).min(i32::MAX as u32) as i32
}
/// Whether [`StructureResources::release_templates`] actually drops the
/// cache (0) or is a no-op (1): a small-heap host whose cache is already
/// count+byte bounded keeps it across the per-job `release_caches` calls —
/// a structure assembly would otherwise re-read every template the last
/// job decoded.
static KEEP_TEMPLATE_CACHE: AtomicU32 = AtomicU32::new(0);
/// Keep the bounded template cache resident across `release_caches` job
/// boundaries (see [`KEEP_TEMPLATE_CACHE`]). Off stays the upstream
/// behaviour.
pub fn set_template_cache_persistent(persistent: bool) {
    KEEP_TEMPLATE_CACHE.store(persistent as u32, Ordering::Relaxed);
}

/// A cached template: its decoded size (the byte bound) and the stamp of
/// its last `template()` touch (the eviction pick, oldest first).
struct TemplateSlot {
    template: Option<Arc<StructureTemplate>>,
    bytes: usize,
    /// A `tick` stamp; u32 wraps, eviction compares ages (`wrapping_sub`),
    /// not raw stamps.
    tick: AtomicU32,
}
/// The `templates` map: a least-recently-touched eviction over the lazily
/// loaded templates, bounded by entries and by their decoded size.
struct TemplateCache {
    map: HashMap<String, TemplateSlot>,
    bytes: usize,
}
impl TemplateCache {
    /// Puts `template` under `id` stamped `now`, then evicts
    /// least-recently-touched entries until both bounds hold (the newest
    /// stamp is picked last, so an entry bigger than the byte bound alone
    /// stays).
    fn insert(
        &mut self,
        id: String,
        template: Option<Arc<StructureTemplate>>,
        bytes: usize,
        now: u32,
    ) {
        match self.map.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                let slot = slot.get_mut();
                self.bytes -= slot.bytes;
                slot.template = template;
                slot.bytes = bytes;
                slot.tick.store(now, Ordering::Relaxed);
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(TemplateSlot {
                    template,
                    bytes,
                    tick: AtomicU32::new(now),
                });
            }
        }
        self.bytes += bytes;
        let (entries, bytes) = (
            TEMPLATE_LIMITS[0].load(Ordering::Relaxed),
            TEMPLATE_LIMITS[1].load(Ordering::Relaxed),
        );
        while (self.map.len() > entries || self.bytes > bytes)
            && self.map.len() > 1
        {
            let victim = self
                .map
                .iter()
                .max_by_key(|(_, slot)| now.wrapping_sub(slot.tick.load(Ordering::Relaxed)))
                .map(|(id, _)| id.clone())
                .expect("more than one entry");
            let slot = self.map.remove(&victim).expect("present");
            self.bytes -= slot.bytes;
        }
    }
}

/// Lazily loaded templates (`structure/<namespace>/<path>.nbt`, gzip NBT) and
/// processor lists, shared by all generators of one resource bundle.
pub struct StructureResources {
    /// This resource bundle's own `Arc`, for the pool elements that fetch
    /// their templates through it (`jigsaw::TemplateRef`).
    this: std::sync::Weak<StructureResources>,
    /// The prepare-packed mirror of `worldgen/` (`skyline-data.bin`):
    /// template reads ask it first — its `structure/<id>.nbt` payloads are
    /// `SKYT` digests (pre-resolved palettes, ~1/3 the decode cost) — and
    /// fall back to the loose file when it doesn't have one.
    packed: Option<Arc<dyn PackedWorldgen>>,
    pub registry: &'static BlockRegistry,
    pub transforms: StateTransforms,
    dir: PathBuf,
    templates: RwLock<TemplateCache>,
    /// What assembly reads of the templates it has looked at (`template_meta`).
    metas: RwLock<HashMap<String, Arc<TemplateMeta>>>,
    /// The stamp each `template()` touch takes (the LRU's order).
    template_tick: AtomicU32,
    /// Data packs' template files, which win over the prepared ones.
    pack_templates: BTreeMap<String, Arc<[u8]>>,
    pub processor_lists: BTreeMap<String, String>,
    jigsaw_states: BTreeMap<BlockStateId, bool>,
    pub(crate) waterlogged: HashMap<BlockStateId, BlockStateId>,
    sources: BTreeSet<BlockStateId>,
    /// Block names in each block tag, for processors matching tags by block.
    pub block_tags: BTreeMap<String, BTreeSet<String>>,
}
impl std::fmt::Debug for StructureResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StructureResources")
    }
}
impl StructureResources {
    pub fn new(
        registry: &'static BlockRegistry,
        transforms: StateTransforms,
        dir: &Path,
        processor_lists: BTreeMap<String, String>,
        block_tags: &BTreeMap<String, Vec<String>>,
        packed: Option<Arc<dyn PackedWorldgen>>,
        pack_templates: BTreeMap<String, Arc<[u8]>>,
    ) -> Arc<Self> {
        let mut sources = BTreeSet::new();
        let fluids = crate::fluid::FluidStateTable::new(registry).ok();
        let mut jigsaw_states = BTreeMap::new();
        let mut waterlogged = HashMap::new();
        for id in 0..registry.len() {
            let id = BlockStateId(id as u32);
            let state = registry.state(id).unwrap();
            if state.name == "minecraft:jigsaw" {
                let orientation = state
                    .properties
                    .get("orientation")
                    .cloned()
                    .unwrap_or_default();
                let front = orientation.split('_').next().unwrap_or("");
                jigsaw_states.insert(id, !matches!(front, "up" | "down"));
            }
            if fluids
                .as_ref()
                .and_then(|table| table.get(id))
                .is_some_and(|fluid| fluid.is_source())
            {
                sources.insert(id);
            }
            // SlabBlock.placeLiquid refuses double slabs.
            let double_slab = state.name.ends_with("_slab")
                && state.properties.get("type").map(String::as_str) == Some("double");
            if !double_slab
                && state.properties.get("waterlogged").map(String::as_str) == Some("false")
            {
                let mut wet = state.clone();
                wet.properties.insert("waterlogged".into(), "true".into());
                if let Some(wet) = registry.id(&wet) {
                    waterlogged.insert(id, wet);
                }
            }
        }
        Arc::new_cyclic(|this| Self {
            this: this.clone(),
            packed,
            registry,
            transforms,
            dir: dir.to_owned(),
            templates: RwLock::new(TemplateCache {
                map: HashMap::new(),
                bytes: 0,
            }),
            metas: RwLock::new(HashMap::new()),
            template_tick: AtomicU32::new(0),
            pack_templates,
            processor_lists,
            jigsaw_states,
            waterlogged,
            sources,
            block_tags: block_tags
                .iter()
                .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
                .collect(),
        })
    }
    /// This bundle's `Arc` (always alive while `&self` is).
    pub(crate) fn arc(&self) -> Arc<StructureResources> {
        self.this.upgrade().expect("resources are built in an Arc")
    }
    /// Drop every cached template (the next `template()` loads it again) —
    /// unless [`set_template_cache_persistent`] kept the bounded cache:
    /// then the LRU's own bound is the only release it needs.
    pub fn release_templates(&self) {
        if KEEP_TEMPLATE_CACHE.load(Ordering::Relaxed) != 0 {
            return;
        }
        let mut cache = self.templates.write().unwrap();
        cache.map = HashMap::new();
        cache.bytes = 0;
    }
    /// Drop the cached [`TemplateMeta`]s (the next `template_meta` reads them again).
    pub fn release_metas(&self) {
        *self.metas.write().unwrap() = HashMap::new();
    }
    /// A template's size and jigsaw blocks, which is all that assembling a
    /// structure reads of it. Read from the file without decoding its
    /// blocks (`TemplateMeta::scan`); the full decode answers when the
    /// scan does not recognize the file, so the result is the same either
    /// way. Kept in a small map of its own: a meta is a few hundred bytes
    /// where the template is kilobytes, and re-reading a village's worth
    /// is what a cold start used to pay for on every assembly.
    pub fn template_meta(&self, id: &str) -> Arc<TemplateMeta> {
        let id: std::borrow::Cow<'_, str> = if id.contains(':') {
            id.into()
        } else {
            format!("minecraft:{id}").into()
        };
        if let Some(found) = self.metas.read().unwrap().get(id.as_ref()) {
            return found.clone();
        }
        let jigsaw_states = |state| self.jigsaw_horizontal(state);
        let scanned = self.scan_meta(&id, &jigsaw_states);
        let meta = Arc::new(scanned.unwrap_or_else(|| match self.template(&id) {
            Some(template) => TemplateMeta::of(&template, &jigsaw_states),
            None => TemplateMeta::of(&StructureTemplate::empty(), &jigsaw_states),
        }));
        let mut metas = self.metas.write().unwrap();
        if metas.len() >= META_LIMIT {
            metas.clear();
        }
        metas.insert(id.into_owned(), meta.clone());
        meta
    }
    /// A template's stored bytes: a data pack's file first (it wins over
    /// the prepared ones), then the packed mirror's
    /// `structure/<namespace>/<path>.nbt` (a `SKYT` digest when the pack
    /// digested it, else the file's own gzipped NBT), then the loose file
    /// under `dir`.
    fn template_bytes(&self, namespace: &str, path: &str) -> Option<Vec<u8>> {
        if !self.pack_templates.is_empty()
            && let Some(bytes) = self.pack_templates.get(&format!("{namespace}:{path}"))
        {
            return Some(bytes.to_vec());
        }
        if let Some(bytes) = self
            .packed
            .as_deref()
            .and_then(|src| src.bytes(&format!("structure/{namespace}/{path}.nbt")))
        {
            return Some(bytes);
        }
        let file = self.dir.join(namespace).join(format!("{path}.nbt"));
        horizon_core::prepared_files::read(&file).ok()
    }
    /// `bytes` as a template: the `SKYT` digest straight, else the gzipped
    /// NBT's decode — either is the same template, the digest just skips
    /// the parse.
    fn decode_template(&self, bytes: &[u8]) -> Option<StructureTemplate> {
        if StructureTemplate::is_digest(bytes) {
            return StructureTemplate::from_digest(bytes);
        }
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_end(&mut raw)
            .ok()?;
        let tag = nbt::read_named(&raw, Limits::templates()).ok()?.0.tag;
        Some(StructureTemplate::from_nbt(
            &tag,
            self.registry,
            &self.transforms,
        ))
    }
    fn scan_meta(
        &self,
        id: &str,
        jigsaw_states: &dyn Fn(BlockStateId) -> Option<bool>,
    ) -> Option<TemplateMeta> {
        let (namespace, path) = id.split_once(':')?;
        let bytes = self.template_bytes(namespace, path)?;
        if StructureTemplate::is_digest(&bytes) {
            return self
                .decode_template(&bytes)
                .map(|t| TemplateMeta::of(&t, jigsaw_states));
        }
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(bytes.as_slice())
            .read_to_end(&mut raw)
            .ok()?;
        drop(bytes);
        TemplateMeta::scan(&raw, self.registry, &self.transforms, jigsaw_states)
    }
    /// StructureTemplateManager.getOrCreate; missing templates are empty.
    pub fn template(&self, id: &str) -> Option<Arc<StructureTemplate>> {
        let id: std::borrow::Cow<'_, str> = if id.contains(':') {
            id.into()
        } else {
            format!("minecraft:{id}").into()
        };
        let id = id.as_ref();
        let tick = self.template_tick.fetch_add(1, Ordering::Relaxed);
        if let Some(found) = self.templates.read().unwrap().map.get(id) {
            found.tick.store(tick, Ordering::Relaxed);
            return found.template.clone();
        }
        let (namespace, path) = id.split_once(':').unwrap();
        let loaded = self.template_bytes(namespace, path).and_then(|bytes| {
            self.decode_template(&bytes)
        });
        // StructureTemplateManager.getOrCreate: missing templates are empty.
        let mut loaded = loaded.unwrap_or_else(StructureTemplate::empty);
        loaded.location = id.into();
        let loaded = Some(Arc::new(loaded));
        let bytes = loaded.as_ref().map_or(0, |t| t.heap_bytes());
        self.templates
            .write()
            .unwrap()
            .insert(id.to_owned(), loaded.clone(), bytes, tick);
        loaded
    }
    /// Whether the state's fluid is a source (water/lava level 0, waterlogged).
    pub fn source_fluid(&self, state: BlockStateId) -> bool {
        self.sources.contains(&state)
    }
    pub fn jigsaw_horizontal(&self, state: BlockStateId) -> Option<bool> {
        self.jigsaw_states.get(&state).copied()
    }
    pub fn processor_list(
        &self,
        json: &str,
        bindings: &super::block_feature::FeatureBlockBindings<'_>,
    ) -> Result<ProcessorList, super::DecodeError> {
        ProcessorList::decode(json, self, bindings)
    }
    pub fn is_jigsaw(&self, state: BlockStateId) -> bool {
        self.jigsaw_states.contains_key(&state)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn slot() -> Option<Arc<StructureTemplate>> {
        Some(Arc::new(StructureTemplate::empty()))
    }

    /// The template cache evicts the least recently touched entry past its
    /// bounds — a hit renews an entry's stamp — and a single too-big entry
    /// still fits.
    #[test]
    fn template_cache_evicts_least_recently_touched() {
        let mut cache = TemplateCache {
            map: HashMap::new(),
            bytes: 0,
        };
        for i in 0..TEMPLATE_CACHE_LIMIT {
            cache.insert(format!("minecraft:t{i}"), slot(), 0, i as u32);
        }
        assert_eq!(cache.map.len(), TEMPLATE_CACHE_LIMIT);
        // Renew the oldest, then insert over the limit: the second-oldest goes.
        cache
            .map
            .get("minecraft:t0")
            .unwrap()
            .tick
            .store(1000, Ordering::Relaxed);
        cache.insert("minecraft:new".into(), slot(), 0, 1001);
        assert_eq!(cache.map.len(), TEMPLATE_CACHE_LIMIT);
        assert!(cache.map.contains_key("minecraft:t0"));
        assert!(!cache.map.contains_key("minecraft:t1"));
        assert!(cache.map.contains_key("minecraft:new"));

        // The byte bound evicts likewise, and an oversized entry alone stays.
        let mut cache = TemplateCache {
            map: HashMap::new(),
            bytes: 0,
        };
        cache.insert("minecraft:a".into(), slot(), TEMPLATE_CACHE_BYTES / 2, 0);
        cache.insert("minecraft:b".into(), slot(), TEMPLATE_CACHE_BYTES / 2, 1);
        assert_eq!(cache.bytes, TEMPLATE_CACHE_BYTES);
        cache.insert("minecraft:c".into(), slot(), 1, 2);
        assert!(!cache.map.contains_key("minecraft:a"));
        assert!(cache.map.contains_key("minecraft:b"));
        cache.insert("minecraft:big".into(), slot(), TEMPLATE_CACHE_BYTES * 2, 3);
        assert_eq!(cache.map.len(), 1);
        assert!(cache.map.contains_key("minecraft:big"));
    }
}
