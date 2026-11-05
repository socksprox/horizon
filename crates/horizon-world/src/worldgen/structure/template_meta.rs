//! What jigsaw assembly reads of a template: its size and its jigsaw blocks.
//!
//! Assembly asks every candidate of every pool for those two things, far
//! more often than anything needs a template's blocks (only placing a piece
//! does). Decoding a whole template — the NBT tree, then a `BlockInfo` for
//! each of its blocks — to answer them was most of a cold village's cost.
//! [`TemplateMeta::scan`] answers them from the file's bytes alone, walking
//! the NBT without building a tree, and returns `None` for anything it does
//! not recognize so that the caller falls back to the full decode, which
//! stays the reference: `TemplateMeta::of(&StructureTemplate)` and `scan`
//! agree on every template of the 26.3 data.
use super::{
    StateTransforms,
    template::{BlockInfo, JigsawInfo, StructureTemplate, read_state},
};
use crate::{
    nbt::{JavaString, Tag},
    registry::{BlockRegistry, BlockStateId},
};
use std::sync::Arc;

/// A template's size and, per palette, its jigsaw blocks (what
/// `StructureTemplate::jigsaws` returns). A jigsaw's `info.nbt` is not kept
/// by [`TemplateMeta::scan`]: assembly reads only the fields `JigsawInfo`
/// parses out of it.
#[derive(Debug)]
pub struct TemplateMeta {
    pub size: [i32; 3],
    palettes: Vec<Vec<JigsawInfo>>,
}

impl TemplateMeta {
    /// The meta of a template already decoded.
    pub fn of(
        template: &StructureTemplate,
        jigsaw_states: &dyn Fn(BlockStateId) -> Option<bool>,
    ) -> Self {
        Self {
            size: template.size,
            palettes: (0..template.palettes().len())
                .map(|palette| template.jigsaws(palette, jigsaw_states).to_vec())
                .collect(),
        }
    }
    /// How many palettes the template has (`getJigsaws` picks one of them).
    pub fn palette_count(&self) -> usize {
        self.palettes.len()
    }
    /// The jigsaw blocks of a palette, as `StructureTemplate::jigsaws`.
    pub fn jigsaws(&self, palette: usize) -> &[JigsawInfo] {
        self.palettes.get(palette).map_or(&[], Vec::as_slice)
    }
    /// An approximation of the heap this holds.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .palettes
                .iter()
                .map(|p| {
                    p.capacity() * std::mem::size_of::<JigsawInfo>()
                        + p.iter()
                            .map(|j| j.name.len() + j.pool.len() + j.target.len())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    /// The meta of the template whose uncompressed NBT is `raw` — the same
    /// as `TemplateMeta::of(&StructureTemplate::from_nbt(..))` — or `None`
    /// when the bytes hold anything this does not read the way `from_nbt`
    /// does (the caller then decodes the whole template).
    pub fn scan(
        raw: &[u8],
        registry: &BlockRegistry,
        transforms: &StateTransforms,
        jigsaw_states: &dyn Fn(BlockStateId) -> Option<bool>,
    ) -> Option<Self> {
        let mut scan = Scan { b: raw, o: 0 };
        if scan.u8()? != 10 {
            return None;
        }
        let root_name = scan.u16()? as usize;
        scan.take(root_name)?;
        let mut size = None;
        let mut palette = None;
        let mut palettes = None;
        let mut blocks = None;
        scan.compound(|scan, id, name| {
            let slot = match name {
                b"size" => {
                    if size.replace(scan.size(id)?).is_some() {
                        return None;
                    }
                    return Some(true);
                }
                b"palette" => &mut palette,
                b"palettes" => &mut palettes,
                b"blocks" => &mut blocks,
                _ => return Some(false),
            };
            // A key twice, or a list of another type: not a template this reads.
            if slot.is_some() || id != 9 {
                return None;
            }
            *slot = Some(scan.o);
            scan.skip(id, 0)?;
            Some(true)
        })?;
        // `from_nbt`: `palettes` (when it is a list) else `palette`, and a
        // template with neither has one empty palette.
        let mut states: Vec<Vec<Option<BlockStateId>>> = Vec::new();
        match (palettes, palette) {
            (Some(at), _) => {
                let mut scan = Scan { b: raw, o: at };
                let (element, count) = scan.list_head()?;
                if count > 0 && element != 9 {
                    return None;
                }
                for _ in 0..count {
                    states.push(scan.palette(registry, jigsaw_states)?);
                }
            }
            (None, Some(at)) => {
                states.push(Scan { b: raw, o: at }.palette(registry, jigsaw_states)?);
            }
            (None, None) => states.push(Vec::new()),
        }
        let mut found: Vec<Vec<(u8, [i32; 3], JigsawInfo)>> = states.iter().map(|_| Vec::new()).collect();
        if let Some(at) = blocks {
            let mut scan = Scan { b: raw, o: at };
            let (element, count) = scan.list_head()?;
            if count > 0 && element != 10 {
                return None;
            }
            for _ in 0..count {
                scan.block(&states, transforms, jigsaw_states, &mut found)?;
            }
        }
        Some(Self {
            size: size.unwrap_or([0; 3]),
            palettes: found
                .into_iter()
                .map(|mut list| {
                    // `from_nbt`'s order: class, then Y, X, Z (a stable sort).
                    list.sort_by_key(|(class, pos, _)| (*class, pos[1], pos[0], pos[2]));
                    list.into_iter().map(|(_, _, jigsaw)| jigsaw).collect()
                })
                .collect(),
        })
    }
}

/// A cursor over uncompressed NBT. Every read returns `None` past the end.
struct Scan<'a> {
    b: &'a [u8],
    o: usize,
}

/// Lists and compounds nest no deeper than this in a template; beyond it
/// the bytes are not read here.
const MAX_DEPTH: u32 = 64;

impl<'a> Scan<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.o.checked_add(len)?;
        let bytes = self.b.get(self.o..end)?;
        self.o = end;
        Some(bytes)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_be_bytes(self.take(2)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_be_bytes(self.take(4)?.try_into().ok()?))
    }
    fn count(&mut self) -> Option<usize> {
        usize::try_from(self.i32()?).ok()
    }
    /// A list's element type and length.
    fn list_head(&mut self) -> Option<(u8, usize)> {
        let element = self.u8()?;
        let count = self.count()?;
        if count > 0 && (element == 0 || element > 12) {
            return None;
        }
        // At least a byte a payload: no allocation or loop on a count the
        // bytes cannot hold.
        (count <= self.b.len() - self.o).then_some((element, count))
    }
    /// Skips the payload of a tag of type `id`.
    fn skip(&mut self, id: u8, depth: u32) -> Option<()> {
        if depth > MAX_DEPTH {
            return None;
        }
        let fixed = |id: u8| match id {
            1 => Some(1),
            2 => Some(2),
            3 | 5 => Some(4),
            4 | 6 => Some(8),
            _ => None,
        };
        match id {
            1..=6 => {
                self.take(fixed(id)?)?;
            }
            7 => {
                let len = self.count()?;
                self.take(len)?;
            }
            8 => {
                let len = self.u16()? as usize;
                self.take(len)?;
            }
            9 => {
                let (element, count) = self.list_head()?;
                match fixed(element) {
                    Some(size) => {
                        self.take(count.checked_mul(size)?)?;
                    }
                    None => {
                        for _ in 0..count {
                            self.skip(element, depth + 1)?;
                        }
                    }
                }
            }
            10 => {
                self.compound_at(depth + 1, |_, _, _| Some(false))?;
            }
            11 => {
                let len = self.count()?;
                self.take(len.checked_mul(4)?)?;
            }
            12 => {
                let len = self.count()?;
                self.take(len.checked_mul(8)?)?;
            }
            _ => return None,
        }
        Some(())
    }
    /// Walks a compound's entries (after its type byte and name): `entry`
    /// gets each key's type and name and returns whether it read the
    /// payload; one that returns `false` has it skipped. An empty key is
    /// refused: the tree reader unwraps a list element whose only key is
    /// empty, and this does not.
    fn compound(
        &mut self,
        entry: impl FnMut(&mut Self, u8, &'a [u8]) -> Option<bool>,
    ) -> Option<()> {
        self.compound_at(0, entry)
    }
    fn compound_at(
        &mut self,
        depth: u32,
        mut entry: impl FnMut(&mut Self, u8, &'a [u8]) -> Option<bool>,
    ) -> Option<()> {
        if depth > MAX_DEPTH {
            return None;
        }
        loop {
            let id = self.u8()?;
            if id == 0 {
                return Some(());
            }
            let len = self.u16()? as usize;
            let name = self.take(len)?;
            if name.is_empty() {
                return None;
            }
            if !entry(self, id, name)? {
                self.skip(id, depth)?;
            }
        }
    }
    /// A string payload as text, for ASCII only (the same in modified UTF-8).
    fn ascii(&mut self) -> Option<&'a str> {
        let len = self.u16()? as usize;
        let bytes = self.take(len)?;
        if bytes.iter().any(|&b| b == 0 || b >= 0x80) {
            return None;
        }
        std::str::from_utf8(bytes).ok()
    }
    /// A template's `size`: a list of three ints.
    fn size(&mut self, id: u8) -> Option<[i32; 3]> {
        if id != 9 {
            return None;
        }
        let (element, count) = self.list_head()?;
        if element != 3 || count != 3 {
            return None;
        }
        Some([self.i32()?, self.i32()?, self.i32()?])
    }
    /// A palette (a list of block-state compounds): for each entry, its
    /// state when it is a jigsaw block, else `None`.
    fn palette(
        &mut self,
        registry: &BlockRegistry,
        jigsaw_states: &dyn Fn(BlockStateId) -> Option<bool>,
    ) -> Option<Vec<Option<BlockStateId>>> {
        let (element, count) = self.list_head()?;
        if count > 0 && element != 10 {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.palette_entry(registry, jigsaw_states)?);
        }
        Some(out)
    }
    fn palette_entry(
        &mut self,
        registry: &BlockRegistry,
        jigsaw_states: &dyn Fn(BlockStateId) -> Option<bool>,
    ) -> Option<Option<BlockStateId>> {
        // `read_state`: `id` before `Name`, `properties` before `Properties`.
        let (mut id, mut name, mut lower, mut upper) = (None, None, None, None);
        self.compound(|scan, tag, key| {
            let slot = match (key, tag) {
                (b"id", 8) => {
                    id = Some(scan.ascii()?);
                    return Some(true);
                }
                (b"Name", 8) => {
                    name = Some(scan.ascii()?);
                    return Some(true);
                }
                (b"properties", 10) => &mut lower,
                (b"Properties", 10) => &mut upper,
                (b"id" | b"Name" | b"properties" | b"Properties", _) => return None,
                _ => return Some(false),
            };
            let mut properties = Vec::new();
            scan.compound(|scan, tag, key| {
                // Only string values are read as properties.
                if tag != 8 {
                    return Some(false);
                }
                properties.push((std::str::from_utf8(key).ok()?.to_owned(), scan.ascii()?.to_owned()));
                Some(true)
            })?;
            *slot = Some(properties);
            Some(true)
        })?;
        let Some(name) = id.or(name) else {
            return Some(None);
        };
        // Only a jigsaw block's state matters here: no other name reads as one.
        if name != "minecraft:jigsaw" {
            return Some(None);
        }
        let mut compound = crate::nbt::Compound::new();
        compound.insert(JavaString::from("id"), Tag::String(JavaString::from(name)));
        if let Some(properties) = lower.or(upper) {
            let mut map = crate::nbt::Compound::new();
            for (key, value) in properties {
                map.insert(JavaString::from(key.as_str()), Tag::String(JavaString::from(value.as_str())));
            }
            compound.insert(JavaString::from("properties"), Tag::Compound(map));
        }
        let state = read_state(registry, &Tag::Compound(compound));
        Some(jigsaw_states(state).map(|_| state))
    }
    /// One entry of `blocks`: if it is a jigsaw block under any palette,
    /// pushes its `JigsawInfo` to that palette's list.
    fn block(
        &mut self,
        palettes: &[Vec<Option<BlockStateId>>],
        transforms: &StateTransforms,
        jigsaw_states: &dyn Fn(BlockStateId) -> Option<bool>,
        found: &mut [Vec<(u8, [i32; 3], JigsawInfo)>],
    ) -> Option<()> {
        let (mut pos, mut state, mut nbt) = (None, 0i32, None);
        self.compound(|scan, id, key| {
            match (key, id) {
                (b"pos", 9) => {
                    let (element, count) = scan.list_head()?;
                    if element != 3 || count != 3 {
                        return None;
                    }
                    pos = Some([scan.i32()?, scan.i32()?, scan.i32()?]);
                }
                (b"state", 3) => state = scan.i32()?,
                (b"nbt", 10) => {
                    nbt = Some(scan.o);
                    scan.skip(10, 0)?;
                }
                (b"pos" | b"state" | b"nbt", _) => return None,
                _ => return Some(false),
            }
            Some(true)
        })?;
        let index = usize::try_from(state).ok();
        let jigsaws: Vec<(usize, BlockStateId)> = palettes
            .iter()
            .enumerate()
            .filter_map(|(i, palette)| Some((i, (*palette.get(index?)?)?)))
            .collect();
        if jigsaws.is_empty() {
            return Some(());
        }
        // `int_list(get(block, "pos"))`: a block without a position sits at 0.
        let pos = pos.unwrap_or([0; 3]);
        let info = match nbt {
            Some(at) => Scan { b: self.b, o: at }.jigsaw_fields()?,
            None => JigsawFields::default(),
        };
        for (palette, state) in jigsaws {
            let horizontal = jigsaw_states(state)?;
            // `from_nbt`'s classes: block entities, else full blocks, else the rest.
            let class = if nbt.is_some() {
                2
            } else if transforms.full_block(state) {
                0
            } else {
                1
            };
            found[palette].push((
                class,
                pos,
                JigsawInfo {
                    rollable: match info.joint.as_deref() {
                        Some("rollable") => true,
                        Some("aligned") => false,
                        _ => !horizontal,
                    },
                    name: info.name.clone(),
                    pool: info.pool.clone(),
                    target: info.target.clone(),
                    placement_priority: info.placement_priority,
                    selection_priority: info.selection_priority,
                    info: BlockInfo {
                        pos,
                        state,
                        nbt: None,
                    },
                },
            ));
        }
        Some(())
    }
    /// A jigsaw block entity's compound (at the cursor, after its type).
    fn jigsaw_fields(&mut self) -> Option<JigsawFields> {
        let mut fields = JigsawFields::default();
        self.compound(|scan, id, key| {
            match (key, id) {
                (b"joint", 8) => fields.joint = Some(scan.ascii()?.to_owned()),
                (b"name", 8) => fields.name = scan.ascii()?.into(),
                (b"pool", 8) => fields.pool = scan.ascii()?.into(),
                (b"target", 8) => fields.target = scan.ascii()?.into(),
                (b"placement_priority", 3) => fields.placement_priority = scan.i32()?,
                (b"selection_priority", 3) => fields.selection_priority = scan.i32()?,
                _ => return Some(false),
            }
            Some(true)
        })?;
        Some(fields)
    }
}

/// The fields `StructureTemplate::jigsaws` reads from a jigsaw block's NBT.
struct JigsawFields {
    joint: Option<String>,
    name: Arc<str>,
    pool: Arc<str>,
    target: Arc<str>,
    placement_priority: i32,
    selection_priority: i32,
}
impl Default for JigsawFields {
    fn default() -> Self {
        let empty: Arc<str> = "minecraft:empty".into();
        Self {
            joint: None,
            name: empty.clone(),
            pool: empty.clone(),
            target: empty,
            placement_priority: 0,
            selection_priority: 0,
        }
    }
}
