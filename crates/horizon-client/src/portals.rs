//! See-through portals (Horizon-only, off unless a world uses them).
//!
//! A `Portal` is a rectangular quad in one dimension paired with the quad its
//! view and travellers come out of in `dest_dim`. The transform between them
//! is the affine map of one quad's basis onto the other's: a point at
//! `center + u*w + v*h + d*n` maps to `dest_center + u*s_w*dest_w + v*s_h*dest_h
//! + d*s_n*dest_n`, where `s_w = |dest_w|/|w|` (and `s_n` the in-plane scale).
//! That single map drives the destination camera for rendering, the traveller's
//! position and velocity on crossing, and raycasts through the surface.
//!
//! The mod this follows is Immersive Portals (Fabric, MIT): a `Portal` entity
//! holds `axisW`/`axisH` and a `dest` quad; its teleport and render code apply
//! the same basis map. What differs here is that the pair lives in one record
//! per direction instead of two entities, and the destination quad is data,
//! not a second synced entity.

use horizon_core::geometry::Vec3;
use horizon_protocol::{
    buf::{Codec, Reader, Writer},
    packets::play::LevelChunkWithLight,
};

/// Wire format version of a synced `Portal` (`portal_net`).
pub const WIRE_VERSION: u8 = 1;
/// A remote dimension's chunk column on the wire (`horizon:remote_chunk`):
/// version byte, the level and dimension-type names, then the
/// LevelChunkWithLight body — the same bytes a vanilla server sends.
pub fn encode_remote_chunk(
    dim: &str,
    dim_type: &str,
    packet: &LevelChunkWithLight,
) -> Vec<u8> {
    let mut out = vec![WIRE_VERSION];
    put_str(&mut out, dim);
    put_str(&mut out, dim_type);
    let mut w = Writer::new();
    <LevelChunkWithLight as Codec<LevelChunkWithLight>>::write(packet, &mut w);
    out.extend_from_slice(&w.data);
    out
}

/// `encode_remote_chunk` with the column's body already encoded
/// (`ChunkEncoder::body_in` output).
pub fn encode_remote_chunk_body(dim: &str, dim_type: &str, body: Vec<u8>) -> Vec<u8> {
    let mut out = vec![WIRE_VERSION];
    put_str(&mut out, dim);
    put_str(&mut out, dim_type);
    out.extend_from_slice(&body);
    out
}

/// `decode_remote_chunk` reverses `encode_remote_chunk`.
pub fn decode_remote_chunk(data: &[u8]) -> Option<(String, String, LevelChunkWithLight)> {
    let mut cur = Cursor { data, at: 0 };
    if cur.u8()? != WIRE_VERSION {
        return None;
    }
    let dim = cur.str()?.to_owned();
    let dim_type = cur.str()?.to_owned();
    let mut r = Reader::new(&data[cur.at..]);
    let packet = <LevelChunkWithLight as Codec<LevelChunkWithLight>>::read(&mut r).ok()?;
    Some((dim, dim_type, packet))
}

/// `horizon:remote_forget`: version, dim, then the column's x and z.
pub fn encode_remote_forget(dim: &str, x: i32, z: i32) -> Vec<u8> {
    let mut out = vec![WIRE_VERSION];
    put_str(&mut out, dim);
    out.extend_from_slice(&x.to_le_bytes());
    out.extend_from_slice(&z.to_le_bytes());
    out
}

/// `decode_remote_forget` reverses `encode_remote_forget`.
pub fn decode_remote_forget(data: &[u8]) -> Option<(String, i32, i32)> {
    let mut cur = Cursor { data, at: 0 };
    if cur.u8()? != WIRE_VERSION {
        return None;
    }
    let dim = cur.str()?.to_owned();
    Some((dim, cur.i32()?, cur.i32()?))
}
/// Clientbound payload carrying a level's portal list.
pub const CHANNEL: &str = "horizon:portals";
/// Clientbound payload carrying a remote dimension's chunks (`remote_worlds`).
pub const REMOTE_CHUNK_CHANNEL: &str = "horizon:remote_chunk";
/// Clientbound payload carrying a remote dimension's entity positions.
pub const REMOTE_ENTITIES_CHANNEL: &str = "horizon:remote_entities";
/// Clientbound payload dropping a remote dimension's chunk column.
pub const REMOTE_FORGET_CHANNEL: &str = "horizon:remote_forget";
/// A traveller this far inside a portal's bounds already crossed last tick:
/// a second application would bounce them back and forth.
const CROSS_EPSILON: f64 = 1.0e-4;

#[derive(Clone, Debug)]
pub struct Portal {
    /// Stable per-level id; `remove` and updates name it.
    pub id: u32,
    /// Dimension it occupies and the one its `dest` quad is in, as the
    /// server's level names (`minecraft:overworld`).
    pub dim: String,
    pub dest_dim: String,
    /// Centre of the portal's quad.
    pub center: Vec3,
    /// Full-width edge and full-height edge of the quad; they may be any
    /// length (a 4x2 portal) and need not be axis aligned, but `w x h` must
    /// be nonzero. `normal` faces the side travellers enter from.
    pub w: Vec3,
    pub h: Vec3,
    /// The quad through-traffic and the view come out of.
    pub dest_center: Vec3,
    pub dest_w: Vec3,
    pub dest_h: Vec3,
    /// Mirror portals: the horizontal axis flips and the view shows the
    /// world reflected across the portal plane.
    pub mirror: bool,
    /// A lit nether-portal frame: shows through but vanilla's own travel
    /// (the stand-in timer) still owns the trip.
    pub auto: bool,
}

/// The four corners of a portal quad, for rendering.
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub corners: [Vec3; 4],
    /// Unit normal on the entry side.
    pub normal: Vec3,
    /// Distance of the quad's plane from the origin along `normal`.
    pub plane_d: f64,
}

impl Portal {
    /// Unit axes of the source quad (`None` on a degenerate portal).
    fn basis(&self) -> Option<(Vec3, Vec3, Vec3)> {
        let w = self.w;
        let h = self.h;
        if w.length_squared() < 1e-12 || h.length_squared() < 1e-12 {
            return None;
        }
        let n = w.cross(h);
        if n.length_squared() < 1e-12 {
            return None;
        }
        Some((w.normalize(), h.normalize(), n.normalize()))
    }
    fn dest_basis(&self) -> Option<(Vec3, Vec3, Vec3)> {
        let w = self.dest_w;
        let h = self.dest_h;
        if w.length_squared() < 1e-12 || h.length_squared() < 1e-12 {
            return None;
        }
        let n = w.cross(h);
        if n.length_squared() < 1e-12 {
            return None;
        }
        Some((w.normalize(), h.normalize(), n.normalize()))
    }
    /// `w`/`h` are orthogonal to the normal by construction (they span the
    /// plane); only each other may be non-orthogonal, which the dot
    /// coordinates still read correctly for a rectangle.
    fn in_plane_scale(&self) -> (f64, f64) {
        let (sw, sh) = (self.dest_w.length() / self.w.length(), self.dest_h.length() / self.h.length());
        (sw, sh)
    }
    /// The `n`-direction scale: the in-plane scale, so a 2x portal scales
    /// depth the same way it scales width and height.
    fn normal_scale(&self) -> f64 {
        let (sw, sh) = self.in_plane_scale();
        (sw + sh) / 2.0
    }

    /// `p` mapped through the portal (positions in `dim` space to `dest_dim`
    /// space). The normal component flips: a point behind the source plane
    /// maps in front of the destination's exit face, so a traveller enters
    /// the front and leaves the front. `mirror` additionally reflects the
    /// horizontal axis.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let Some((w, h, n)) = self.basis() else { return p };
        let Some((dw, dh, dn)) = self.dest_basis() else { return p };
        let s = p.subtract(self.center);
        let (sw, sh) = self.in_plane_scale();
        let sn = self.normal_scale();
        let u = s.dot(w) * sw * if self.mirror { -1.0 } else { 1.0 };
        let v = s.dot(h) * sh;
        let d = -s.dot(n) * sn;
        self.dest_center
            .add_vector(dw.scale(u))
            .add_vector(dh.scale(v))
            .add_vector(dn.scale(d))
    }
    /// A direction or velocity mapped through the portal (no translation).
    pub fn transform_dir(&self, dir: Vec3) -> Vec3 {
        let Some((w, h, n)) = self.basis() else { return dir };
        let Some((dw, dh, dn)) = self.dest_basis() else { return dir };
        let (sw, sh) = self.in_plane_scale();
        let sn = self.normal_scale();
        let u = dir.dot(w) * sw * if self.mirror { -1.0 } else { 1.0 };
        let v = dir.dot(h) * sh;
        let d = -dir.dot(n) * sn;
        dw.scale(u).add_vector(dh.scale(v)).add_vector(dn.scale(d))
    }
    /// `transform_dir` on the look vector, read back as degrees.
    /// Returns `(yaw, pitch)`.
    pub fn transform_yaw_pitch(&self, yaw: f32, pitch: f32) -> (f32, f32) {
        let (y, p) = (yaw.to_radians(), pitch.to_radians());
        let look = Vec3::new(
            f64::from(-y.sin() * p.cos()),
            f64::from(-p.sin()),
            f64::from(y.cos() * p.cos()),
        );
        let t = self.transform_dir(look);
        let flat = t.horizontal_distance();
        let yaw = (-t.x).atan2(t.z).to_degrees() as f32;
        let pitch = (-t.y).atan2(flat).to_degrees() as f32;
        (yaw, pitch)
    }

    /// Quad geometry: corners and the unit normal on the entry side.
    pub fn quad(&self) -> Option<Quad> {
        let (_, _, n) = self.basis()?;
        let (w2, h2) = (self.w.scale(0.5), self.h.scale(0.5));
        let c = self.center;
        Some(Quad {
            corners: [
                c.subtract(w2).subtract(h2),
                c.add_vector(w2).subtract(h2),
                c.add_vector(w2).add_vector(h2),
                c.subtract(w2).add_vector(h2),
            ],
            normal: n,
            plane_d: n.dot(c),
        })
    }
    /// The destination quad's geometry (what the view comes out of).
    pub fn dest_quad(&self) -> Option<Quad> {
        let (_, _, n) = self.dest_basis()?;
        let (w2, h2) = (self.dest_w.scale(0.5), self.dest_h.scale(0.5));
        let c = self.dest_center;
        Some(Quad {
            corners: [
                c.subtract(w2).subtract(h2),
                c.add_vector(w2).subtract(h2),
                c.add_vector(w2).add_vector(h2),
                c.subtract(w2).add_vector(h2),
            ],
            normal: n,
            plane_d: n.dot(c),
        })
    }

    /// Signed distance of `p` from the portal plane, positive on the entry
    /// side (`normal` points at travellers before they cross).
    pub fn side_distance(&self, p: Vec3) -> f64 {
        let Some((_, _, n)) = self.basis() else {
            return f64::MAX;
        };
        p.subtract(self.center).dot(n)
    }
    /// Whether `p` is inside the quad's rectangle when projected onto its
    /// plane (`margin` grows the bounds, for boxes that overlap the edge).
    pub fn inside(&self, p: Vec3, margin: f64) -> bool {
        let Some((w, h, _)) = self.basis() else {
            return false;
        };
        let s = p.subtract(self.center);
        s.dot(w).abs() <= self.w.length() / 2.0 + margin
            && s.dot(h).abs() <= self.h.length() / 2.0 + margin
    }
    /// Whether the `from -> to` segment crosses the portal plane from the
    /// entry side to the back, with the crossing point inside the quad
    /// (plus `margin`). `from` inside `CROSS_EPSILON` of the plane counts as
    /// already through: it does not re-cross.
    pub fn crossed(&self, from: Vec3, to: Vec3, margin: f64) -> bool {
        let d0 = self.side_distance(from);
        let d1 = self.side_distance(to);
        if d0 <= CROSS_EPSILON || d1 >= -CROSS_EPSILON {
            return false;
        }
        let t = d0 / (d0 - d1);
        self.inside(from.lerp(to, t), margin)
    }
    /// The camera for the view through this portal: `camera` and
    /// `yaw/pitch` transformed into `dest_dim` space.
    pub fn view_camera(&self, camera: Vec3, yaw: f32, pitch: f32) -> (Vec3, f32, f32) {
        let (yaw, pitch) = self.transform_yaw_pitch(yaw, pitch);
        (self.transform_point(camera), yaw, pitch)
    }
    /// Aabb covering the quad, for frustum culling and "is it near" checks.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let Some(q) = self.quad() else {
            return (self.center, self.center);
        };
        let mut min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
        for c in q.corners {
            min = Vec3::new(min.x.min(c.x), min.y.min(c.y), min.z.min(c.z));
            max = Vec3::new(max.x.max(c.x), max.y.max(c.y), max.z.max(c.z));
        }
        (min, max)
    }
}

/// The smallest enclosing sphere of a portal quad (`bounds` centre and
/// radius), used to decide which portals are worth a view each frame.
pub fn quad_radius(p: &Portal) -> f64 {
    (p.w.length().max(p.h.length())) / 2.0
}

/// The portal's transform as a `glam::DMat4`: `p' = L(p - c) + dc`, used to
/// transform entity vertex positions baked camera-relative so entities can
/// be drawn through the portal too.
pub fn transform_mat4(p: &Portal) -> Option<glam::DMat4> {
    let (w, h, n) = p.basis()?;
    let (dw, dh, dn) = p.dest_basis()?;
    let (sw, sh) = p.in_plane_scale();
    let sn = p.normal_scale();
    let m = if p.mirror { -1.0 } else { 1.0 };
    // Row-major in math terms, column-major in memory: the j-th column is
    // L applied to basis vector e_j, i.e. w.x * dw' + h.x * dh' + n.x * dn'.
    let col = |i: usize| -> glam::DVec4 {
        let s = [w, h, n];
        let d = [dw, dh, dn];
        let k = [sw * m, sh, -sn];
        glam::DVec4::new(
            (0..3).map(|j| d[j].x * k[j] * s[j].components()[i]).sum(),
            (0..3).map(|j| d[j].y * k[j] * s[j].components()[i]).sum(),
            (0..3).map(|j| d[j].z * k[j] * s[j].components()[i]).sum(),
            0.0,
        )
    };
    let l = glam::DMat4::from_cols(col(0), col(1), col(2), glam::DVec4::W);
    // p' = L(p) - L(c) + dc
    let lc = l.transform_point3(glam::DVec3::new(p.center.x, p.center.y, p.center.z));
    let t = glam::DVec3::new(
        p.dest_center.x - lc.x,
        p.dest_center.y - lc.y,
        p.dest_center.z - lc.z,
    );
    Some(glam::DMat4::from_translation(t) * l)
}

/// A wire encoding (`horizon:portals`): version byte, then
/// `u32` count and each portal's fields.
pub fn encode(portals: &[Portal]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + portals.len() * 140);
    out.push(WIRE_VERSION);
    out.extend_from_slice(&(portals.len() as u32).to_le_bytes());
    for p in portals {
        put_str(&mut out, &p.id.to_string());
        put_str(&mut out, &p.dim);
        put_str(&mut out, &p.dest_dim);
        for v in [
            p.center, p.w, p.h, p.dest_center, p.dest_w, p.dest_h,
        ] {
            for c in v.components() {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.push(u8::from(p.mirror) | (u8::from(p.auto) << 1));
    }
    out
}

/// `decode` reverses `encode`; `Err` is a malformed payload (ignored).
pub fn decode(data: &[u8]) -> Option<Vec<Portal>> {
    let mut cur = Cursor { data, at: 0 };
    if cur.u8()? != WIRE_VERSION {
        return None;
    }
    let count = cur.u32()? as usize;
    let mut portals = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let id: u32 = cur.str()?.parse().ok()?;
        let dim = cur.str()?.to_owned();
        let dest_dim = cur.str()?.to_owned();
        let mut f64s = [0.0f64; 18];
        for c in &mut f64s {
            *c = cur.f64()?;
        }
        let v = |i: usize| Vec3::new(f64s[i], f64s[i + 1], f64s[i + 2]);
        let flags = cur.u8()?;
        portals.push(Portal {
            id,
            dim,
            dest_dim,
            center: v(0),
            w: v(3),
            h: v(6),
            dest_center: v(9),
            dest_w: v(12),
            dest_h: v(15),
            mirror: flags & 1 != 0,
            auto: flags & 2 != 0,
        });
    }
    Some(portals)
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}
struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let s = self.data.get(self.at..end)?;
        self.at = end;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<&'a str> {
        let len = u16::from_le_bytes(self.take(2)?.try_into().ok()?) as usize;
        std::str::from_utf8(self.take(len)?).ok()
    }
}

/// A portal pair where each side's entry face points away from the other
/// (you walk in one face and out of the other's far face): `b` is `a`'s
/// destination and `a` is `b`'s. `half_d` pushes each centre this far along
/// its normal so the traveller lands clear of the plane they exit.
pub fn pair(
    mut a: Portal,
    mut b: Portal,
    next_id: &mut u32,
) -> (Portal, Portal) {
    a.id = *next_id;
    *next_id += 1;
    b.id = *next_id;
    *next_id += 1;
    let a_to_b = Portal {
        id: a.id,
        dim: a.dim.clone(),
        dest_dim: b.dim.clone(),
        center: a.center,
        w: a.w,
        h: a.h,
        dest_center: b.center,
        dest_w: b.w,
        dest_h: b.h,
        mirror: a.mirror,
        auto: a.auto,
    };
    let b_to_a = Portal {
        id: b.id,
        dim: b.dim,
        dest_dim: a.dim,
        center: b.center,
        w: b.w,
        h: b.h,
        dest_center: a.center,
        dest_w: a.w,
        dest_h: a.h,
        mirror: b.mirror,
        auto: b.auto,
    };
    (a_to_b, b_to_a)
}

/// `pair` for a one-direction portal only (`b` stays a wall-less exit).
pub fn one_way(mut p: Portal, dest: Portal) -> Portal {
    p.dest_center = dest.center;
    p.dest_w = dest.w;
    p.dest_h = dest.h;
    p.dest_dim = dest.dim;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_portal() -> Portal {
        // A 2x2 portal at the origin facing +z, to a copy of itself at
        // (0, 0, 10) facing -z (you enter +z, you exit -z... see pair()).
        Portal {
            id: 1,
            dim: "minecraft:overworld".into(),
            dest_dim: "minecraft:overworld".into(),
            center: Vec3::new(0.0, 64.0, 0.0),
            w: Vec3::new(2.0, 0.0, 0.0),
            h: Vec3::new(0.0, 2.0, 0.0),
            dest_center: Vec3::new(0.0, 64.0, 10.0),
            dest_w: Vec3::new(-2.0, 0.0, 0.0),
            dest_h: Vec3::new(0.0, 2.0, 0.0),
            mirror: false,
        }
    }

    #[test]
    fn transform_preserves_plane_positions() {
        let p = flat_portal();
        // Centre maps to centre.
        let c = p.transform_point(p.center);
        assert!(c.distance(p.dest_center) < 1e-9, "{c:?}");
        // A point on the +x edge maps to the dest's +w edge (w' = -x).
        let e = p.transform_point(Vec3::new(1.0, 64.0, 0.0));
        assert!(e.distance(Vec3::new(-1.0, 64.0, 10.0)) < 1e-9, "{e:?}");
        // A point 1m in front (+z) of the source maps behind the
        // destination's normal (d flips: n' = (-x)x(+y) = -z, so d'=-1 →
        // +z side): the space you see through the portal is the space you
        // would emerge into.
        let f = p.transform_point(Vec3::new(0.0, 64.0, 1.0));
        assert!(f.distance(Vec3::new(0.0, 64.0, 11.0)) < 1e-9, "{f:?}");
        // And 1m behind the source maps in front of the destination.
        let b = p.transform_point(Vec3::new(0.0, 64.0, -1.0));
        assert!(b.distance(Vec3::new(0.0, 64.0, 9.0)) < 1e-9, "{b:?}");
    }

    #[test]
    fn crossing_is_one_way_within_bounds() {
        let p = flat_portal();
        // Entering front (+z) to back crosses.
        assert!(p.crossed(
            Vec3::new(0.5, 64.0, 0.5),
            Vec3::new(0.5, 64.0, -0.5),
            0.0
        ));
        // Back to front is not "entering" (you arrive on the far side).
        assert!(!p.crossed(
            Vec3::new(0.5, 64.0, -0.5),
            Vec3::new(0.5, 64.0, 0.5),
            0.0
        ));
        // Outside the quad's x bounds.
        assert!(!p.crossed(
            Vec3::new(5.0, 64.0, 0.5),
            Vec3::new(5.0, 64.0, -0.5),
            0.0
        ));
        // Outside its y bounds.
        assert!(!p.crossed(
            Vec3::new(0.0, 70.0, 0.5),
            Vec3::new(0.0, 70.0, -0.5),
            0.0
        ));
        // Margin widens the accepted rectangle.
        assert!(p.crossed(
            Vec3::new(1.5, 64.0, 0.5),
            Vec3::new(1.5, 64.0, -0.5),
            0.6
        ));
    }

    #[test]
    fn yaw_rotates_with_the_portal() {
        let p = flat_portal();
        // Walking in (-z, yaw 180): the dest's front face (n' = -z) opens
        // toward -z, so you come out still walking -z — straight through
        // the doorway.
        let (yaw, pitch) = p.transform_yaw_pitch(180.0, 0.0);
        assert!((yaw.abs() - 180.0).abs() < 1e-4, "{yaw}");
        assert!(pitch.abs() < 1e-4);
        // Facing +x (yaw -90): the u axis is kept, w' = -x, so the look
        // comes out -x (yaw 90) — the doorway's left-right flip.
        let (yaw, _) = p.transform_yaw_pitch(-90.0, 0.0);
        assert!((yaw - 90.0).abs() < 1e-3, "{yaw}");
    }

    #[test]
    fn wire_round_trip() {
        let p = flat_portal();
        let q = decode(&encode(&[p.clone()])).unwrap();
        assert_eq!(q.len(), 1);
        let q = &q[0];
        assert_eq!(q.id, 1);
        assert_eq!(q.dim, "minecraft:overworld");
        assert_eq!(q.dest_w, p.dest_w);
        assert_eq!(q.dest_h, p.dest_h);
        assert!(!q.mirror);
        assert!(decode(&[]).is_none());
        assert!(decode(&[9, 0, 0]).is_none());
    }

    #[test]
    fn self_portal_is_a_mirror() {
        let mut p = flat_portal();
        // A mirror: destination is the same quad — the flipped normal
        // component alone is a physical reflection across the plane.
        p.dest_center = p.center;
        p.dest_w = p.w;
        p.dest_h = p.h;
        let r = p.transform_point(Vec3::new(1.0, 64.0, 1.0));
        assert!(r.distance(Vec3::new(1.0, 64.0, -1.0)) < 1e-9, "{r:?}");
        // `mirror` adds a horizontal flip on top (a left-right swap, not a
        // physical mirror): for odd effects, not for a looking glass.
        p.mirror = true;
        let r = p.transform_point(Vec3::new(1.0, 64.0, 1.0));
        assert!(r.distance(Vec3::new(-1.0, 64.0, -1.0)) < 1e-9, "{r:?}");
    }

    #[test]
    fn scaled_portal_grows_distances() {
        let mut p = flat_portal();
        p.dest_w = Vec3::new(-4.0, 0.0, 0.0);
        p.dest_h = Vec3::new(0.0, 4.0, 0.0);
        // 1m in front → 2m past the dest plane on its far side (flipped
        // normal, in-plane scale 2): (0,64,10) + (-z)·(-2) = (0,64,12).
        let f = p.transform_point(Vec3::new(0.0, 64.0, 1.0));
        assert!(f.distance(Vec3::new(0.0, 64.0, 12.0)) < 1e-9, "{f:?}");
    }
}
