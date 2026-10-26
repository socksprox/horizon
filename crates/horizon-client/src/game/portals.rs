//! See-through portals, game side (`horizon_client::portals` is the math):
//! the level's portal list, player crossings, the views the renderer needs
//! each frame, and the `horizon:portals` sync for remote clients.
//!
//! The list lives in the level (`game/levels.rs` parks it with the world):
//! servers mutate it and push it to guests, remote clients only read it.

use super::{Dim, Game};
use horizon_client::{
    gfx::{self, portal::View},
    portals::{self, Portal},
};
use horizon_core::geometry::Vec3;
use horizon_protocol::packets::{configuration as cfg, play};

/// A remote dimension the client holds for a portal's view (`world`
/// index + 1 in `View`); chunks stream in with `remote_worlds`.
pub(super) struct RemoteDim {
    pub dim: String,
}

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
    /// The encoded full list for a client (`horizon:portals`); a per-level
    /// snapshot, sent whole on every change (portals are few).
    pub(super) fn portals_packet(&self) -> play::Clientbound {
        play::Clientbound::CustomPayload(cfg::CustomPayload {
            channel: portals::CHANNEL.into(),
            data: portals::encode(&self.imm_portals),
        })
    }
    /// A crossing into another dimension (`tick_portal_crossing`): the
    /// server's travel machinery moves the player and the client follows
    /// (`server_travel`); filled in with cross-dimension portals.
    pub(super) fn request_portal_travel(&mut self, _portal: u32) {}
    /// A remote client's portals list, wholesale (what the server sent).
    pub(super) fn set_remote_portals(&mut self, list: Vec<Portal>) {
        self.imm_portals = list;
        self.portal_sync_dirty = false;
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
            let view_proj = horizon_client::gfx::portal::oblique(
                horizon_client::gfx::world_view_projection(
                    aspect, fov_degrees, dyaw, dpitch, bob, depth_far,
                ),
                clip,
            );
            let world = self.portal_world_index(&p.dest_dim);
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
    /// a streamed remote world. Remote worlds come with the cross-
    /// dimension milestone; until then every view draws the current one.
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
