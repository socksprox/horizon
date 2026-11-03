//! Per-state gameplay facts exported from vanilla's own block-state accessors
//! (`gameplay-facts.json`), indexed by registry state ID.
mod decode;
use crate::Result;
use horizon_world::{
    level::{Behaviours, GameplayStateSource},
    registry::{BlockRegistry, BlockStateId},
};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

/// A JSON error naming the prepared file it came from, as
/// `horizon_world::json_file::parse` does for a slice.
fn named(path: &Path, e: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderShape {
    Model,
    Invisible,
    Animated,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fluid {
    None,
    Water,
    Lava,
}
#[derive(Clone, Debug)]
pub struct Sounds {
    pub volume: f32,
    pub pitch: f32,
    pub break_sound: Arc<str>,
    pub step: Arc<str>,
    pub place: Arc<str>,
    pub hit: Arc<str>,
    pub fall: Arc<str>,
}
#[derive(Clone, Debug)]
pub struct StateFacts {
    pub name: Arc<str>,
    pub emission: u8,
    pub dampening: u8,
    pub skylight_down: bool,
    pub shape_occlusion: bool,
    pub can_occlude: bool,
    pub solid_render: bool,
    pub hardness: f32,
    pub requires_tool: bool,
    pub replaceable: bool,
    pub random_ticks: bool,
    pub block_entity: bool,
    pub render_shape: RenderShape,
    pub full_collision: bool,
    pub friction: f32,
    pub explosion_resistance: f32,
    pub item: Arc<str>,
    pub sounds: Sounds,
    pub fluid: Fluid,
    /// Fluid amount 1..=8 (8 = source/falling surface), 0 when there is no fluid.
    pub fluid_level: u8,
    pub fluid_falling: bool,
    pub waterlogged: bool,
    /// `hasOffsetFunction`: the model is jittered per column.
    pub random_offset: bool,
    /// Whether the outline, not just the model, follows BlockState.getOffset.
    pub outline_offset: bool,
    /// `BlockState.isSolid()` (legacy solidity; fluid surfaces read it).
    pub legacy_solid: bool,
    /// `isFaceSturdy(.., SupportType.FULL)` by Direction ordinal.
    pub sturdy_full: u8,
    /// The block's class chain (`TransparentBlock`, `LeavesBlock`, ...), most specific first.
    pub classes: Arc<[Arc<str>]>,
    /// `getShape` (the outline used for picking) as boxes in block space,
    /// without the random offset; and its edges as LevelRenderer draws them.
    pub outline: Arc<[[f64; 6]]>,
    pub outline_edges: Arc<[[f64; 6]]>,
    /// `getMapColor(..).id` (0 = NONE).
    pub map_color: u8,
}
/// `MapColor.col` by id.
pub const MAP_COLORS: [u32; 62] = [
    0, 8368696, 16247203, 13092807, 16711680, 10526975, 10987431, 31744, 16777215, 10791096,
    9923917, 7368816, 4210943, 9402184, 16776437, 14188339, 11685080, 6724056, 15066419, 8375321,
    15892389, 5000268, 10066329, 5013401, 8339378, 3361970, 6704179, 6717235, 10040115, 1644825,
    16445005, 6085589, 4882687, 55610, 8476209, 7340544, 13742497, 10441252, 9787244, 7367818,
    12223780, 6780213, 10505550, 3746083, 8874850, 5725276, 8014168, 4996700, 4993571, 5001770,
    9321518, 2430480, 12398641, 9715553, 6035741, 1474182, 3837580, 5647422, 1356933, 6579300,
    14200723, 8365974,
];
/// The 12 edges of the unit cube (outline data predating the export).
pub fn cube_edges() -> Vec<[f64; 6]> {
    let mut edges = Vec::new();
    for a in 0..3 {
        for (u, v) in [(0., 0.), (1., 0.), (0., 1.), (1., 1.)] {
            let mut from = [0.0f64; 3];
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            from[b] = u;
            from[c] = v;
            let mut to = from;
            to[a] = 1.0;
            edges.push([from[0], from[1], from[2], to[0], to[1], to[2]]);
        }
    }
    edges
}
/// Which of the four `SignBlock` subclasses a block is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignForm {
    /// `CeilingHangingSignBlock` or `WallHangingSignBlock`.
    pub hanging: bool,
    /// `WallSignBlock` or `WallHangingSignBlock`.
    pub wall: bool,
}
impl SignForm {
    /// The block entity type its `newBlockEntity` makes.
    pub fn block_entity_type(self) -> &'static str {
        if self.hanging {
            "minecraft:hanging_sign"
        } else {
            "minecraft:sign"
        }
    }
}
impl StateFacts {
    pub fn has_class(&self, name: &str) -> bool {
        self.classes.iter().any(|c| &**c == name)
    }
    /// A `SignBlock`'s form, by class: a mod's sign has its template's,
    /// whatever its name.
    pub fn sign_form(&self) -> Option<SignForm> {
        self.has_class("SignBlock").then(|| SignForm {
            hanging: self.has_class("CeilingHangingSignBlock")
                || self.has_class("WallHangingSignBlock"),
            wall: self.has_class("WallSignBlock") || self.has_class("WallHangingSignBlock"),
        })
    }
}
/// Compact, owned behaviour inputs retained until the normal behaviour startup stage.
/// No borrowed gameplay JSON or process-wide cache survives the fact loader.
pub struct GameplayProjection {
    flags: u16,
    emission: u64,
    dampening: u64,
    hardness: f64,
    push: Arc<str>,
    volume: f64,
    pitch: f64,
    place: Arc<str>,
    break_sound: Arc<str>,
    fall: Arc<str>,
}

impl GameplayProjection {
    fn from_entry(entry: &decode::Entry, strings: &mut HashMap<String, Arc<str>>) -> Self {
        fn intern(strings: &mut HashMap<String, Arc<str>>, value: &str) -> Arc<str> {
            if let Some(existing) = strings.get(value) {
                return Arc::clone(existing);
            }
            let value_arc: Arc<str> = Arc::from(value);
            strings.insert(value.to_owned(), Arc::clone(&value_arc));
            value_arc
        }
        const FLAGS: [&str; 11] = [
            "replaceable",
            "random_ticks",
            "solid_render",
            "liquid",
            "block_entity",
            "ignited_by_lava",
            "can_occlude",
            "full_collision",
            "skylight_down",
            "requires_tool",
            "shape_occlusion",
        ];
        Self {
            flags: FLAGS.iter().enumerate().fold(0, |flags, (i, name)| {
                flags | ((u16::from(entry.flag(name))) << i)
            }),
            emission: entry.unsigned("emission"),
            dampening: entry.unsigned("dampening"),
            hardness: entry.number("hardness"),
            push: intern(strings, entry.text("push")),
            volume: entry.sound_number("volume"),
            pitch: entry.sound_number("pitch"),
            place: intern(strings, entry.sound_id("place")),
            break_sound: intern(strings, entry.sound_id("break")),
            fall: intern(strings, entry.sound_id("fall")),
        }
    }
}

impl GameplayStateSource for GameplayProjection {
    fn flag(&self, name: &str) -> bool {
        const FLAGS: [&str; 11] = [
            "replaceable",
            "random_ticks",
            "solid_render",
            "liquid",
            "block_entity",
            "ignited_by_lava",
            "can_occlude",
            "full_collision",
            "skylight_down",
            "requires_tool",
            "shape_occlusion",
        ];
        FLAGS
            .iter()
            .position(|field| *field == name)
            .is_some_and(|index| self.flags & (1 << index) != 0)
    }
    fn unsigned(&self, name: &str) -> u64 {
        match name {
            "emission" => self.emission,
            "dampening" => self.dampening,
            _ => 0,
        }
    }
    fn number(&self, name: &str) -> f64 {
        if name == "hardness" {
            self.hardness
        } else {
            0.0
        }
    }
    fn text(&self, name: &str) -> &str {
        if name == "push" { &self.push } else { "" }
    }
    fn sound_number(&self, name: &str) -> f64 {
        match name {
            "volume" => self.volume,
            "pitch" => self.pitch,
            _ => 1.0,
        }
    }
    fn sound_id(&self, name: &str) -> &str {
        match name {
            "place" => &self.place,
            "break" => &self.break_sound,
            "fall" => &self.fall,
            _ => "",
        }
    }
}

pub struct BlockFacts {
    pub states: Vec<StateFacts>,
    pub air: BlockStateId,
    /// Each sign block's `SignBlock.getWoodType` name.
    pub woods: HashMap<Arc<str>, Arc<str>>,
}
impl BlockFacts {
    pub fn load(root: &Path, registry: &BlockRegistry) -> Result<Self> {
        let behaviours: Value =
            horizon_world::json_file::read(&root.join("block-behaviours.json"))?;
        Self::load_with_behaviour_json(root, registry, &behaviours)
    }

    /// Load both tables from one resource root, sharing each JSON decode.
    /// No data is cached across calls: changes to prepared packs or mods are read again.
    pub fn load_pair(root: &Path, registry: &BlockRegistry) -> Result<(Self, Behaviours)> {
        let behaviours: Value =
            horizon_world::json_file::read(&root.join("block-behaviours.json"))?;
        Self::load_with_behaviours(root, registry, &behaviours)
    }

    /// Decode state facts once, retaining only the small owned fields needed later.
    /// `parallel`: the JSON validation pass may run on a thread of its own.
    pub fn load_with_projection(
        root: &Path,
        registry: &BlockRegistry,
        behaviours: &Value,
        parallel: bool,
    ) -> Result<(Self, Vec<GameplayProjection>)> {
        Self::load_with_callback(root, registry, behaviours, parallel, |entries| {
            let mut strings = HashMap::new();
            Ok(entries
                .iter()
                .map(|entry| GameplayProjection::from_entry(entry, &mut strings))
                .collect())
        })
    }

    pub fn load_with_behaviour_json(
        root: &Path,
        registry: &BlockRegistry,
        behaviours: &Value,
    ) -> Result<Self> {
        Ok(Self::load_with_callback(root, registry, behaviours, true, |_| Ok(()))?.0)
    }

    /// `load_with_behaviour_json` for small heaps (armv6k-nintendo-3ds):
    /// the parallel validation pass cannot live beside the decode, and
    /// the ~55 MB file cannot sit on the heap at all — both passes stream
    /// off the filesystem and each decoded entry folds into the facts
    /// table as it arrives, so the entry array is never resident either.
    pub fn load_with_behaviour_json_seq(
        root: &Path,
        registry: &BlockRegistry,
        behaviours: &Value,
    ) -> Result<Self> {
        let path = root.join("gameplay-facts.json");
        serde_json::from_reader::<_, decode::Validate>(BufReader::new(File::open(&path)?))
            .map(drop)?;
        let mut decode = Decode::new(registry, behaviours)?;
        let mut shapes = decode::Shapes::default();
        let mut present = false;
        let mut deserializer =
            serde_json::Deserializer::from_reader(BufReader::new(File::open(path)?));
        serde::Deserializer::deserialize_any(
            &mut deserializer,
            decode::Streamed {
                present: &mut present,
                sink: |index: usize, entry: decode::Entry| {
                    decode.entry_owned(&mut shapes, index, &entry)
                },
            },
        )?;
        if !present {
            return Err("missing gameplay state facts".into());
        }
        decode.finish()
    }

    /// Build both tables from the same behaviour document and gameplay decoder.
    pub fn load_with_behaviours(
        root: &Path,
        registry: &BlockRegistry,
        behaviours: &Value,
    ) -> Result<(Self, Behaviours)> {
        Self::load_with_callback(root, registry, behaviours, true, |entries| {
            Ok(Behaviours::load_with_gameplay(
                registry, root, behaviours, entries,
            )?)
        })
    }

    fn load_with_callback<T>(
        root: &Path,
        registry: &BlockRegistry,
        behaviours: &Value,
        parallel: bool,
        callback: impl FnOnce(&[decode::Entry]) -> Result<T>,
    ) -> Result<(Self, T)> {
        let path = root.join("gameplay-facts.json");
        if !parallel {
            // Streamed validation keeps the file off the heap; the decode
            // reads it a second time rather than retaining it.
            serde_json::from_reader::<_, decode::Validate>(BufReader::new(File::open(&path)?))
                .map_err(|e| named(&path, e))?;
            return Self::decode_with_callback(
                &path,
                BufReader::new(File::open(&path)?),
                registry,
                behaviours,
                callback,
            );
        }
        let bytes = horizon_world::json_file::read_bytes(&path)?;
        let validate =
            || horizon_world::json_file::parse::<decode::Validate>(&path, &bytes).map(drop);
        // The validation pass (a third of the load) only reports errors: it
        // reads the file beside the decode, and its error still wins.
        std::thread::scope(|scope| {
            let validated = scope.spawn(validate);
            let decoded =
                Self::decode_with_callback(&path, &bytes[..], registry, behaviours, callback);
            validated
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            decoded
        })
    }

    fn decode_with_callback<T>(
        path: &Path,
        reader: impl std::io::Read,
        registry: &BlockRegistry,
        behaviours: &Value,
        callback: impl FnOnce(&[decode::Entry]) -> Result<T>,
    ) -> Result<(Self, T)> {
        let raw: decode::Gameplay =
            serde_json::from_reader(reader).map_err(|e| named(path, e))?;
        let entries = raw
            .states
            .0
            .as_ref()
            .ok_or("missing gameplay state facts")?;
        if entries.len() != registry.len() {
            return Err("gameplay facts do not match the block registry".into());
        }
        let mut decode = Decode::new(registry, behaviours)?;
        let mut shapes = decode::Shapes::default();
        for (index, entry) in entries.iter().enumerate() {
            decode.entry_ref(&mut shapes, index, entry)?;
        }
        let extra = callback(entries)?;
        Ok((decode.finish()?, extra))
    }
    /// A state the registry doesn't have (a server with mods we lack sent it)
    /// has air's facts.
    #[inline]
    pub fn get(&self, id: BlockStateId) -> &StateFacts {
        self.states
            .get(id.0 as usize)
            .unwrap_or_else(|| &self.states[self.air.0 as usize])
    }
    /// `SignBlock.getWoodType`'s name for a sign block.
    pub fn wood_type(&self, block: &str) -> Option<&str> {
        self.woods.get(block).map(|wood| &**wood)
    }
    pub fn is_air(&self, id: BlockStateId) -> bool {
        matches!(
            &*self.get(id).name,
            "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
        )
    }
}

impl GameplayStateSource for decode::Entry {
    fn flag(&self, name: &str) -> bool {
        match name {
            "replaceable" => self.replaceable,
            "random_ticks" => self.random_ticks,
            "solid_render" => self.solid_render,
            "liquid" => self.liquid,
            "block_entity" => self.block_entity,
            "ignited_by_lava" => self.ignited_by_lava,
            "can_occlude" => self.can_occlude,
            "full_collision" => self.full_collision,
            "skylight_down" => self.skylight_down,
            "requires_tool" => self.requires_tool,
            "shape_occlusion" => self.shape_occlusion,
            _ => false,
        }
    }
    fn unsigned(&self, name: &str) -> u64 {
        match name {
            "emission" => self.emission.as_u64().unwrap_or(0),
            "dampening" => self.dampening.as_u64().unwrap_or(0),
            _ => 0,
        }
    }
    fn number(&self, name: &str) -> f64 {
        match name {
            "hardness" => self.hardness.as_f64().unwrap_or(0.0),
            _ => 0.0,
        }
    }
    fn text(&self, name: &str) -> &str {
        match name {
            "push" => self.push.as_deref().unwrap_or(""),
            _ => "",
        }
    }
    fn sound_number(&self, name: &str) -> f64 {
        match name {
            "volume" => self.sound.volume,
            "pitch" => self.sound.pitch,
            _ => 1.0,
        }
    }
    fn sound_id(&self, name: &str) -> &str {
        match name {
            "place" => self.sound.place.id.as_deref().unwrap_or(""),
            "break" => self.sound.r#break.id.as_deref().unwrap_or(""),
            "fall" => self.sound.fall.id.as_deref().unwrap_or(""),
            _ => "",
        }
    }
}

/// The per-load context `decode_with_callback`'s loop and
/// `load_with_behaviour_json_seq`'s streamed rows share: the behaviour
/// lookups (`behaviour_states`, `classes`), the interners (`names`, the
/// `Shapes` the callers pass), and the growing `states` table.
struct Decode<'a> {
    registry: &'a BlockRegistry,
    behaviour_states: &'a [Value],
    classes: HashMap<&'a str, Arc<[Arc<str>]>>,
    woods: HashMap<Arc<str>, Arc<str>>,
    names: HashMap<String, Arc<str>>,
    states: Vec<StateFacts>,
}

impl<'a> Decode<'a> {
    /// Class chains and per-state behaviour flags exported from the actual
    /// block objects; the per-row intern tables start empty.
    fn new(registry: &'a BlockRegistry, behaviours: &'a Value) -> Result<Self> {
        let behaviour_states = behaviours["states"]
            .as_array()
            .filter(|s| s.len() == registry.len())
            .ok_or("block behaviour facts do not match the block registry")?;
        let mut class_names: HashMap<String, Arc<str>> = HashMap::new();
        let classes: HashMap<&str, Arc<[Arc<str>]>> = behaviours["blocks"]
            .as_array()
            .ok_or("missing block behaviour classes")?
            .iter()
            .filter_map(|b| {
                let chain = b["class"]
                    .as_array()?
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|c| {
                        class_names
                            .entry(c.to_owned())
                            .or_insert_with(|| Arc::from(c))
                            .clone()
                    });
                Some((b["name"].as_str()?, chain.collect()))
            })
            .collect();
        let woods = behaviours["blocks"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|b| {
                b["class"]
                    .as_array()
                    .is_some_and(|chain| chain.iter().any(|c| c == "SignBlock"))
            })
            .filter_map(|b| {
                Some((
                    Arc::from(b["name"].as_str()?),
                    Arc::from(b["fields"]["type"]["name"].as_str()?),
                ))
            })
            .collect();
        Ok(Self {
            registry,
            behaviour_states,
            classes,
            woods,
            names: HashMap::new(),
            states: Vec::with_capacity(registry.len()),
        })
    }
    fn intern_str(&mut self, s: &str) -> Arc<str> {
        if let Some(name) = self.names.get(s) {
            return name.clone();
        }
        let name: Arc<str> = Arc::from(s);
        self.names.insert(s.to_owned(), name.clone());
        name
    }
    /// The air-aware outline fallbacks the `Shapes::intern*` calls take
    /// when a row carries no explicit geometry.
    fn outline_fallbacks(
        &self,
        index: usize,
    ) -> (
        impl FnOnce() -> Vec<[f64; 6]>,
        impl FnOnce() -> Vec<[f64; 6]>,
    ) {
        let air = self
            .registry
            .state(BlockStateId(index as u32))
            .is_some_and(|state| {
                matches!(
                    state.name.as_str(),
                    "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
                )
            });
        let cube = move || {
            if air {
                Vec::new()
            } else {
                vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
            }
        };
        let edges = move || if air { Vec::new() } else { cube_edges() };
        (cube, edges)
    }
    /// One row of `states` for the collected decode: the entry's raw text
    /// stays alive for the whole pass, so `Shapes::intern` can dedup on it.
    fn entry_ref<'b>(
        &mut self,
        shapes: &mut decode::Shapes<'b>,
        index: usize,
        entry: &'b decode::Entry,
    ) -> Result<()> {
        let (cube, edges) = self.outline_fallbacks(index);
        let outline = shapes.intern(entry.outline.as_deref(), cube)?;
        let outline_edges = shapes.intern(entry.outline_edges.as_deref(), edges)?;
        self.entry_facts(index, entry, outline, outline_edges)
    }
    /// One row for the streamed decode: the entry frees right after, so
    /// only the value-level shape interning applies.
    fn entry_owned(
        &mut self,
        shapes: &mut decode::Shapes<'_>,
        index: usize,
        entry: &decode::Entry,
    ) -> Result<()> {
        let (cube, edges) = self.outline_fallbacks(index);
        let outline = shapes.intern_owned(entry.outline.as_deref(), cube)?;
        let outline_edges = shapes.intern_owned(entry.outline_edges.as_deref(), edges)?;
        self.entry_facts(index, entry, outline, outline_edges)
    }
    /// `entry`'s `StateFacts` row, pushed in registry order.
    fn entry_facts(
        &mut self,
        index: usize,
        entry: &decode::Entry,
        outline: Arc<[[f64; 6]]>,
        outline_edges: Arc<[[f64; 6]]>,
    ) -> Result<()> {
        if index >= self.registry.len() {
            return Err("gameplay facts do not match the block registry".into());
        }
        let id = BlockStateId(index as u32);
        if entry.id != Some(index as u64) {
            return Err("gameplay facts are not in registry order".into());
        }
        let registry = self.registry;
        let state = registry.state(id).ok_or("missing registry state")?;
        let sound = &entry.sound;
        let prop = |k: &str| state.properties.get(k).map(String::as_str);
        let (fluid, fluid_level, fluid_falling) = match state.name.as_str() {
            "minecraft:water" | "minecraft:lava" => {
                let level: u8 = prop("level").and_then(|v| v.parse().ok()).unwrap_or(0);
                let kind = if state.name == "minecraft:water" {
                    Fluid::Water
                } else {
                    Fluid::Lava
                };
                // LiquidBlock: level 0 is the source; 1..7 flowing; >=8 falling.
                if level == 0 {
                    (kind, 8, false)
                } else if level >= 8 {
                    (kind, 8, true)
                } else {
                    (kind, 8 - level, false)
                }
            }
            "minecraft:bubble_column"
            | "minecraft:kelp"
            | "minecraft:kelp_plant"
            | "minecraft:seagrass"
            | "minecraft:tall_seagrass" => (Fluid::Water, 8, false),
            _ if prop("waterlogged") == Some("true") => (Fluid::Water, 8, false),
            _ => (Fluid::None, 0, false),
        };
        // The export holds MapColor.col; ids are positions in MAP_COLORS.
        let col = entry.map_color.unwrap_or(0) as u32;
        let map_color = if col == 0 {
            0
        } else {
            MAP_COLORS.iter().position(|c| *c == col).unwrap_or(0) as u8
        };
        let name = self.intern_str(&state.name);
        let item = self.intern_str(entry.item.as_deref().unwrap_or("minecraft:air"));
        let break_sound = self.intern_str(sound.r#break.id.as_deref().unwrap_or(""));
        let step = self.intern_str(sound.step.id.as_deref().unwrap_or(""));
        let place = self.intern_str(sound.place.id.as_deref().unwrap_or(""));
        let hit = self.intern_str(sound.hit.id.as_deref().unwrap_or(""));
        let fall = self.intern_str(sound.fall.id.as_deref().unwrap_or(""));
        let classes = self
            .classes
            .get(state.name.as_str())
            .cloned()
            .ok_or("missing block behaviour class chain")?;
        self.states.push(StateFacts {
            map_color,
            outline,
            outline_edges,
            name,
            emission: entry.emission.as_f64().unwrap_or(0.0) as u8,
            dampening: entry.dampening.as_f64().unwrap_or(0.0) as u8,
            skylight_down: entry.skylight_down,
            shape_occlusion: entry.shape_occlusion,
            can_occlude: entry.can_occlude,
            solid_render: entry.solid_render,
            hardness: entry.hardness.as_f64().unwrap_or(0.0) as f32,
            requires_tool: entry.requires_tool,
            replaceable: entry.replaceable,
            random_ticks: entry.random_ticks,
            block_entity: entry.block_entity,
            render_shape: match entry.render_shape.as_deref() {
                Some("INVISIBLE") => RenderShape::Invisible,
                Some("ENTITYBLOCK_ANIMATED") => RenderShape::Animated,
                _ => RenderShape::Model,
            },
            full_collision: entry.full_collision,
            friction: entry.friction as f32,
            explosion_resistance: entry.explosion_resistance as f32,
            item,
            sounds: Sounds {
                volume: sound.volume as f32,
                pitch: sound.pitch as f32,
                break_sound,
                step,
                place,
                hit,
                fall,
            },
            fluid,
            fluid_level,
            fluid_falling,
            waterlogged: prop("waterlogged") == Some("true"),
            random_offset: entry.offset,
            outline_offset: entry.outline_offset.unwrap_or(entry.offset),
            legacy_solid: self.behaviour_states[index][1].as_u64().unwrap_or(0) & 1 != 0,
            sturdy_full: (self.behaviour_states[index][0].as_u64().unwrap_or(0) & 63) as u8,
            classes,
        });
        Ok(())
    }
    fn finish(self) -> Result<BlockFacts> {
        if self.states.len() != self.registry.len() {
            return Err("gameplay facts do not match the block registry".into());
        }
        Ok(BlockFacts {
            states: self.states,
            air: self
                .registry
                .default_state("minecraft:air")
                .ok_or("missing air")?,
            woods: self.woods,
        })
    }
}
