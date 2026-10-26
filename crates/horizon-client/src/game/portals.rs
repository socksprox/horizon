//! See-through portals, game side (`horizon_client::portals` is the math):
//! the level's portal list, player crossings, the views the renderer needs
//! each frame, and the `horizon:portals` sync for remote clients.
//!
//! The list lives in the level (`game/levels.rs` parks it with the world):
//! servers mutate it and push it to guests, remote clients only read it.

use super::{Dim, Game, commands::R};
use horizon_client::command::{
    args::Arg,
    Context,
};
use horizon_client::{
    gfx::{self, portal::View},
    portals::{self, Portal},
    text::Text,
};
use horizon_core::geometry::Vec3;
use horizon_protocol::packets::{configuration as cfg, play};
use horizon_world::nbt::{Compound, Tag};

/// How far past a portal's bounds a crossing still counts: the player's
/// 0.6m box, halved, plus a step.
const CROSS_MARGIN: f64 = 0.5;
/// A traveller re-crossing within this many ticks is ignored: a pair puts
/// you just past the far plane, and one tick of lag could send you back.
pub(super) const CROSS_COOLDOWN: u8 = 10;
/// The bounding-box centre height used for crossing checks.
pub(super) const CROSS_HEIGHT: f64 = 0.9;
/// Portals render and teleport within this distance of the player (blocks).
const MAX_DISTANCE: f64 = 96.0;

impl Game<'_> {
    /// Portals of the dimension the level in place is.
    pub(super) fn dim_portals(&self) -> impl Iterator<Item = &Portal> {
        let dim = self.dim.id();
        self.imm_portals.iter().filter(move |p| p.dim == dim)
    }
    /// All portals whose far side is `dim` (their views need its world).
    pub(super) fn portals_into(&self, dim: Dim) -> impl Iterator<Item = &Portal> {
        let dim = dim.id();
        self.imm_portals.iter().filter(move |p| p.dest_dim == dim)
    }

    /// `/portal`-family bookkeeping when a portal is added or removed: sync
    /// every guest and the owner, and keep ids unique per level.
    pub(super) fn portals_changed(&mut self) {
        self.portal_sync_dirty = true;
    }
    /// Push the level's portal list to every placed guest when it changed
    /// or they have not been sent it yet (`tick_world`); clients filter
    /// it to their own dimension.
    pub(super) fn sync_portals(&mut self) {
        if self.lan.is_none() {
            self.portal_sync_dirty = false;
            return;
        }
        let changed = self.portal_sync_dirty;
        let conns: Vec<u32> = self
            .lan
            .as_ref()
            .map(|h| {
                h.hub
                    .guests
                    .iter()
                    .filter(|g| g.placed)
                    .map(|g| g.conn)
                    .collect()
            })
            .unwrap_or_default();
        for conn in conns {
            if !changed && self.portal_synced.contains(&conn) {
                continue;
            }
            let packet = self.portals_packet();
            if let Some(g) = self
                .lan
                .as_ref()
                .and_then(|h| h.hub.guests.iter().find(|g| g.conn == conn))
            {
                g.send(&packet);
            }
            self.portal_synced.insert(conn);
        }
        self.portal_sync_dirty = false;
    }
    /// The encoded full list for a client (`horizon:portals`); a per-level
    /// snapshot, sent whole on every change (portals are few).
    pub(super) fn portals_packet(&self) -> play::Clientbound {
        play::Clientbound::CustomPayload(cfg::CustomPayload {
            channel: portals::CHANNEL.into(),
            data: portals::encode(&self.imm_portals),
        })
    }
    /// A crossing into another dimension (`tick_portal_crossing`). Remote
    /// clients wait for the server: it sees the same crossing from the
    /// move packets and runs the level teleport (`tick_guests`). In-frame,
    /// this Game is both sides and moves itself.
    pub(super) fn request_portal_travel(&mut self, id: u32) {
        let Some(p) = self.imm_portals.iter().find(|p| p.id == id).cloned() else {
            return;
        };
        if self.remote.is_some() || self.serving.is_some() {
            return;
        }
        let dim = Dim::from_id(&p.dest_dim);
        if let Err(e) = self.enter_level(None, dim) {
            log::error!("portal travel to {}: {e}", p.dest_dim);
            return;
        }
        if let Some(player) = self.player.as_mut() {
            player.position = p.transform_point(player.position);
            player.velocity = p.transform_dir(player.velocity);
        }
        let (yaw, pitch) = p.transform_yaw_pitch(self.yaw, self.pitch);
        self.yaw = yaw;
        self.pitch = pitch;
    }
    /// A remote client's portals list, wholesale (what the server sent).
    pub(super) fn set_remote_portals(&mut self, list: Vec<Portal>) {
        self.imm_portals = list;
        self.portal_sync_dirty = false;
    }

    /// `data/horizon_imm_portals.dat` in the world folder (like
    /// `nether_portal.rs`'s arrivals file).
    fn imm_portals_path(&self) -> std::path::PathBuf {
        self.world_root
            .join("data")
            .join("horizon_imm_portals.dat")
    }
    /// A level's portals when it opens (`open_level`): every level keeps
    /// its list in the one file, tagged by dimension.
    pub(super) fn load_imm_portals(&self, dim: Dim) -> Vec<Portal> {
        let Some(root) = horizon_client::save::read_dat(&self.imm_portals_path())
        else {
            return Vec::new();
        };
        let Some(Tag::List(list)) = root.get("portals") else {
            return Vec::new();
        };
        list.iter()
            .filter_map(portal_from_tag)
            .filter(|p| p.dim == dim.id())
            .collect()
    }
    /// Every level's portals (`save_game_metadata`): the open level's
    /// plus each parked level's.
    pub(super) fn save_imm_portals(&mut self) {
        if self.remote.is_some() {
            return;
        }
        let mut list: Vec<Tag> = self.imm_portals.iter().map(portal_to_tag).collect();
        for p in self.levels.parked_portals() {
            list.push(portal_to_tag(p));
        }
        let root = Tag::compound([("portals", Tag::List(list))]);
        if let Err(e) = horizon_client::save::write_dat(&self.imm_portals_path(), root)
        {
            log::warn!("Horizon: saving portals failed: {e}");
        }
    }

    /// Whether `from -> to` (bounding-box centres) entered a portal in
    /// `dim` this tick. An associated function so callers inside the
    /// `self.player` reborrow can pass `&self.imm_portals` — the fields are
    /// disjoint, which a `&self` receiver would not let the borrow checker
    /// see.
    pub(super) fn crossed_in<'p>(
        portals: &'p [Portal],
        dim: &str,
        from: Vec3,
        to: Vec3,
    ) -> Option<&'p Portal> {
        portals
            .iter()
            .find(|p| p.dim == dim && p.crossed(from, to, CROSS_MARGIN))
    }
    /// A cross-dimension crossing queued inside the player borrow is run
    /// once it is out (`apply_portal_travel` early in the tick).
    pub(super) fn apply_portal_travel(&mut self) {
        if let Some(id) = self.portal_travel_request.take() {
            self.request_portal_travel(id);
        }
    }

    /// What the renderer draws through this frame's visible portals: one
    /// `View` each, nearest first, `world` telling which resident world
    /// holds the far side (0 while remote worlds are unstreamed).
    pub(super) fn portal_views(
        &mut self,
        camera: Vec3,
        yaw: f32,
        pitch: f32,
        fov_degrees: f32,
        bob: glam::Mat4,
        depth_far: f32,
        aspect: f32,
    ) -> Vec<View<'static>> {
        let main_vp = horizon_client::gfx::world_view_projection(
            aspect, fov_degrees, yaw, pitch, bob, depth_far,
        );
        let dim = self.dim.id();
        // Remote worlds exist before the borrow: a cross-dimension view's
        // `world` resolves to one even before its chunks land.
        let dest_dims: Vec<String> = self
            .imm_portals
            .iter()
            .filter(|p| p.dim == dim && p.dest_dim != dim)
            .map(|p| p.dest_dim.clone())
            .collect();
        for d in dest_dims {
            self.ensure_remote_world(&d, &d);
        }
        let mut by_distance: Vec<&Portal> = self
            .imm_portals
            .iter()
            .filter(|p| p.dim == dim)
            .filter(|p| p.side_distance(camera) > -0.25)
            .filter(|p| {
                let (min, max) = p.bounds();
                let cx = (min.x + max.x) / 2.0;
                let cy = (min.y + max.y) / 2.0;
                let cz = (min.z + max.z) / 2.0;
                camera.distance_squared(Vec3::new(cx, cy, cz))
                    < (MAX_DISTANCE * MAX_DISTANCE) + portals::quad_radius(p)
            })
            .collect();
        by_distance.sort_by_key(|p| {
            let (min, max) = p.bounds();
            let cx = (min.x + max.x) / 2.0;
            let cy = (min.y + max.y) / 2.0;
            let cz = (min.z + max.z) / 2.0;
            camera
                .distance_squared(Vec3::new(cx, cy, cz))
                .to_bits()
        });
        let mut views = Vec::new();
        for p in by_distance {
            if views.len() >= gfx::portal::MAX_VIEWS {
                break;
            }
            let Some(quad) = p.quad() else { continue };
            // Quad corners relative to the main camera; cull by the main
            // frustum so off-screen portals cost no extra pass.
            let corners: [[f32; 3]; 4] = std::array::from_fn(|i| {
                let c = quad.corners[i].subtract(camera);
                [c.x as f32, c.y as f32, c.z as f32]
            });
            if !Self::quad_in_frustum(&corners, main_vp) {
                continue;
            }
            // The destination camera, and the destination plane folded
            // into its projection as the near clip.
            let (dcam, dyaw, dpitch) = p.view_camera(camera, yaw, pitch);
            let view = horizon_client::camera::view_matrix(dyaw, dpitch);
            let Some(dq) = p.dest_quad() else {
                continue;
            };
            let n = dq.normal;
            // Keep fragments past the dest plane from the camera: in
            // camera-relative space `n . c > n . (dc - cam)`; in view
            // space the normal rotates by `view` and the offset is
            // `n . (dc - cam)`.
            let cam_rel = Vec3::new(dq.corners[0].x, dq.corners[0].y, dq.corners[0].z)
                .lerp(
                    Vec3::new(dq.corners[2].x, dq.corners[2].y, dq.corners[2].z),
                    0.5,
                )
                .subtract(dcam);
            let rn = view.transform_vector3(glam::Vec3::new(
                n.x as f32,
                n.y as f32,
                n.z as f32,
            ));
            let d = -(n.x * cam_rel.x + n.y * cam_rel.y + n.z * cam_rel.z) as f32;
            let clip = glam::Vec4::new(rn.x, rn.y, rn.z, d);
            // The clip plane folds into the bare projection; bob and the
            // view's own rotation apply after it.
            let view_proj = horizon_client::gfx::portal::oblique(
                horizon_client::gfx::world_projection(fov_degrees, aspect, depth_far),
                clip,
            ) * bob
                * horizon_client::camera::view_matrix(dyaw, dpitch);
            let world = self.portal_world_index(&p.dest_dim);
            if world == usize::MAX {
                // Cross-dimension view before `remote_worlds` streams the
                // far side: draw nothing rather than the wrong world.
                continue;
            }
            views.push(View {
                corners,
                camera: glam::DVec3::new(dcam.x, dcam.y, dcam.z),
                view_proj,
                sky: None,
                skybox: self.dim == Dim::Overworld && self.settings.horizon.sky,
                entities: &[],
                world,
            });
        }
        views
    }

    /// Which resident world a `View::world` draws: the current one (0), or
    /// a remote world (created on demand so a new portal shows sky before
    /// its chunks land).
    fn portal_world_index(&self, dest_dim: &str) -> usize {
        if dest_dim == self.dim.id() {
            0
        } else {
            self.remote_worlds
                .iter()
                .position(|w| w.dim == dest_dim)
                .map_or(usize::MAX, |i| i + 1)
        }
    }

    /// `/portal spawn|mirror|floor|list|remove [id]` (command/vanilla.rs).
    /// Spawning puts the portal ahead of the player, facing them; pairs
    /// link both ways so either face is an entrance.
    pub(super) fn cmd_portal(&mut self, id: &str, ctx: &Context) -> R<i32> {
        match id {
            "portal.list" => {
                let mut lines = Vec::new();
                if self.imm_portals.is_empty() {
                    lines.push("No portals.".to_owned());
                }
                for p in &self.imm_portals {
                    let (c, d) = (p.center, p.dest_center);
                    lines.push(format!(
                        "#{} {} ({:.1} {:.1} {:.1}) -> {} ({:.1} {:.1} {:.1})",
                        p.id, p.dim, c.x, c.y, c.z, p.dest_dim, d.x, d.y, d.z
                    ));
                }
                let count = self.imm_portals.len() as i32;
                for line in lines {
                    self.feedback(Text::literal(line));
                }
                Ok(count)
            }
            "portal.remove" => {
                let before = self.imm_portals.len();
                if let Some(Arg::Int(want)) = ctx.get("id") {
                    self.imm_portals.retain(|p| p.id != *want as u32);
                } else if let Some(player) = &self.player {
                    // Nearest portal within 4 blocks of the player's feet.
                    let feet = player.position;
                    let near = self
                        .imm_portals
                        .iter()
                        .filter(|p| p.dim == self.dim.id())
                        .map(|p| {
                            let (lo, hi) = p.bounds();
                            let c = (lo.add_vector(hi.subtract(lo))).scale(0.5);
                            (p.id, feet.distance_squared(c))
                        })
                        .filter(|(_, d)| *d < 16.0)
                        .min_by_key(|(_, d)| d.to_bits())
                        .map(|(id, _)| id);
                    if let Some(id) = near {
                        self.imm_portals.retain(|p| p.id != id);
                    }
                }
                let removed = before - self.imm_portals.len();
                if removed > 0 {
                    self.portals_changed();
                }
                self.feedback(Text::literal(format!("Removed {removed} portal(s).")));
                Ok(removed as i32)
            }
            _ => {
                let Some(player) = &self.player else {
                    self.feedback(Text::literal("No player."));
                    return Ok(0);
                };
                let feet = player.position;
                let yaw = self.yaw;
                let dim = self.dim.id().to_owned();
                let mut next = self
                    .imm_portals
                    .iter()
                    .map(|p| p.id)
                    .max()
                    .unwrap_or(0)
                    + 1;
                let spawned = match id {
                    "portal.spawn" => {
                        let (a, b) = portal_pair(feet, yaw, &dim, &mut next);
                        self.imm_portals.extend([a, b]);
                        2
                    }
                    "portal.mirror" => {
                        self.imm_portals.push(mirror_portal(feet, yaw, &dim, next));
                        1
                    }
                    "portal.floor" => {
                        let (a, b) = floor_pair(feet, yaw, &dim, &mut next);
                        self.imm_portals.extend([a, b]);
                        2
                    }
                    _ => 0,
                };
                if spawned > 0 {
                    self.portals_changed();
                    self.feedback(Text::literal(format!("Spawned {spawned} portal(s).")));
                }
                Ok(spawned)
            }
        }
    }

    /// Frustum test for a quad's camera-relative corners under `view_proj`:
    /// false only when every corner is outside the same clip side.
    fn quad_in_frustum(corners: &[[f32; 3]; 4], view_proj: glam::Mat4) -> bool {
        let mut out = [true; 6];
        for c in corners {
            let v = view_proj * glam::Vec4::new(c[0], c[1], c[2], 1.0);
            out[0] &= v.x < -v.w;
            out[1] &= v.x > v.w;
            out[2] &= v.y < -v.w;
            out[3] &= v.y > v.w;
            out[4] &= v.z < 0.0;
            out[5] &= v.z > v.w;
        }
        !out.iter().all(|&b| b)
    }
}

/// A portal as NBT for `horizon_imm_portals.dat`.
fn portal_to_tag(p: &Portal) -> Tag {
    let mut t = Compound::new();
    t.insert("id".into(), Tag::Int(p.id as i32));
    t.insert("dim".into(), Tag::String(p.dim.as_str().into()));
    t.insert("dest".into(), Tag::String(p.dest_dim.as_str().into()));
    let put = |t: &mut Compound, key: &str, v: Vec3| {
        for (k, c) in v.components().iter().enumerate() {
            t.insert(format!("{key}{k}").as_str().into(), Tag::Double(c.to_bits()));
        }
    };
    put(&mut t, "c", p.center);
    put(&mut t, "w", p.w);
    put(&mut t, "h", p.h);
    put(&mut t, "dc", p.dest_center);
    put(&mut t, "dw", p.dest_w);
    put(&mut t, "dh", p.dest_h);
    t.insert("mirror".into(), Tag::Int(i32::from(p.mirror)));
    Tag::Compound(t)
}
/// `portal_from_tag` reverses `portal_to_tag`; malformed entries drop out.
fn portal_from_tag(t: &Tag) -> Option<Portal> {
    let str_tag = |k: &str| horizon_client::save::text(t.get(k));
    let int = |k: &str| match t.get(k) {
        Some(Tag::Int(v)) => Some(*v),
        _ => None,
    };
    let vec = |key: &str| -> Option<Vec3> {
        let c = |i: usize| match t.get(&format!("{key}{i}")) {
            Some(Tag::Double(v)) => Some(f64::from_bits(*v)),
            _ => None,
        };
        Some(Vec3::new(c(0)?, c(1)?, c(2)?))
    };
    Some(Portal {
        id: int("id")? as u32,
        dim: str_tag("dim")?,
        dest_dim: str_tag("dest")?,
        center: vec("c")?,
        w: vec("w")?,
        h: vec("h")?,
        dest_center: vec("dc")?,
        dest_w: vec("dw")?,
        dest_h: vec("dh")?,
        mirror: int("mirror")? != 0,
    })
}

/// Horizontal unit forward and right for `yaw` (Minecraft's: yaw 0 south).
fn yaw_frame(yaw: f32) -> (Vec3, Vec3) {
    let y = f64::from(yaw).to_radians();
    (
        Vec3::new(-y.sin(), 0.0, y.cos()),
        Vec3::new(-y.cos(), 0.0, -y.sin()),
    )
}
/// `/portal spawn`: two 2x3 verticals 16 apart, linked both ways.
fn portal_pair(feet: Vec3, yaw: f32, dim: &str, next: &mut u32) -> (Portal, Portal) {
    let (fwd, right) = yaw_frame(yaw);
    let up = Vec3::new(0.0, 1.5, 0.0);
    let a = Portal {
        id: 0,
        dim: dim.to_owned(),
        dest_dim: dim.to_owned(),
        // Facing the player, two blocks ahead: its normal points at them.
        center: feet.add_vector(fwd.scale(2.0)).add_vector(up),
        w: right.scale(2.0),
        h: Vec3::new(0.0, 3.0, 0.0),
        dest_center: Vec3::ZERO,
        dest_w: Vec3::ZERO,
        dest_h: Vec3::ZERO,
        mirror: false,
    };
    let b = Portal {
        center: a.center.add_vector(fwd.scale(16.0)),
        w: a.w.scale(-1.0),
        ..a.clone()
    };
    portals::pair(a, b, next)
}
/// `/portal mirror`: the quad's destination is itself, so walking in puts
/// you back out the same face and the view reflects the world behind you.
fn mirror_portal(feet: Vec3, yaw: f32, dim: &str, id: u32) -> Portal {
    let (fwd, right) = yaw_frame(yaw);
    let up = Vec3::new(0.0, 1.5, 0.0);
    let center = feet.add_vector(fwd.scale(2.0)).add_vector(up);
    Portal {
        id,
        dim: dim.to_owned(),
        dest_dim: dim.to_owned(),
        center,
        w: right.scale(2.0),
        h: Vec3::new(0.0, 3.0, 0.0),
        dest_center: center,
        dest_w: right.scale(2.0),
        dest_h: Vec3::new(0.0, 3.0, 0.0),
        // Same basis on both sides: the transform reduces to reflecting
        // through the portal's own plane — a mirror, not a flip.
        mirror: false,
    }
}
/// `/portal floor`: a 2x2 opening in the ground ahead; falling in drops
/// you out of a matching opening 15 up, 16 ahead (and back up through it).
fn floor_pair(feet: Vec3, yaw: f32, dim: &str, next: &mut u32) -> (Portal, Portal) {
    let (fwd, right) = yaw_frame(yaw);
    let a = Portal {
        id: 0,
        dim: dim.to_owned(),
        dest_dim: dim.to_owned(),
        center: feet.add_vector(fwd.scale(2.0)),
        // Flat: `w x h` = right x forward = +y, the face you fall into.
        w: right.scale(2.0),
        h: fwd.scale(2.0),
        dest_center: Vec3::ZERO,
        dest_w: Vec3::ZERO,
        dest_h: Vec3::ZERO,
        mirror: false,
    };
    let b = Portal {
        center: a
            .center
            .add_vector(fwd.scale(16.0))
            .add_vector(Vec3::new(0.0, 15.0, 0.0)),
        // Swapped axes make `w x h` face down: you fall out its underside.
        w: a.h,
        h: a.w,
        ..a.clone()
    };
    portals::pair(a, b, next)
}
