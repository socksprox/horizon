//! See-through portals (Horizon-only): each visible portal renders the
//! world again from the camera its transform puts on the far side, into an
//! offscreen target, and the portal's quad samples that texture at the
//! screen position it covers — the same approach as Immersive Portals'
//! `RendererUsingFrameBuffer`, on wgpu.
//!
//! Views are capped (`MAX_VIEWS`) and never recursive: a portal rendered
//! inside another portal's view draws nothing. Geometry on the camera's
//! side of the destination plane is removed by folding the destination
//! plane into the view's projection (`oblique`), the classic near-plane
//! clip, instead of a per-fragment discard.

use super::{Globals, Renderer};
use crate::environment::Sky;
use glam::{DVec3, Mat4, Vec4};

/// The most portal views one frame renders: each costs a second world pass.
pub const MAX_VIEWS: usize = 4;

/// One portal the frame wants drawn, the far side rendered as `view`.
pub struct View {
    /// The quad's corners relative to the main camera (the drawn surface).
    pub corners: [[f32; 3]; 4],
    /// Where the destination camera sits in the destination world.
    pub camera: DVec3,
    /// `world_projection * bob * view * oblique`, computed by the game.
    pub view_proj: Mat4,
    /// The destination's sky (fog colour and sky colour) when it differs
    /// from the current world's; the frame's own values are used for None.
    pub sky: Option<Sky>,
    /// Whether the destination's sky shows at all (`frame.skybox` for it).
    pub skybox: bool,
    /// Entities baked for this view, camera-relative to `camera`.
    pub entities: Vec<crate::mesher::Vertex>,
    /// Which resident world to draw: the current one (0), or a remote
    /// world the client also holds (`remote` index + 1).
    pub world: usize,
    /// The outer view this view shows through, if this is a nested
    /// portal-in-portal view: its index in the frame's view list. Inner
    /// views render first and the parent's pass samples their target.
    /// `corners` then hold the inner quad in the parent's camera space.
    pub parent: Option<usize>,
}

/// `self.portal_targets` slots that survive until a resize; one per `View`.
pub(super) struct Target {
    pub color: wgpu::Texture,
    /// The quad's bind group: main globals + this target's colour.
    pub group: wgpu::BindGroup,
    /// The world-pass bind group: this view's own globals + the shared
    /// atlas and lightmap (rebuilt when they are, with `self.group`).
    pub world_group: wgpu::BindGroup,
    /// This view's `Globals`.
    pub globals: wgpu::Buffer,
    pub color_view: wgpu::TextureView,
    pub depth_view: wgpu::TextureView,
}

/// Near-plane clipping by folding a plane into the projection (Lengyel's
/// oblique projection, for wgpu's 0..1 clip depth). `plane` is the clip
/// plane in view space `(n.x, n.y, n.z, d)`: fragments with `n . x + d < 0`
/// are clipped — the camera must be on that side, so `d` is made negative.
pub fn oblique(proj: Mat4, plane: Vec4) -> Mat4 {
    let c = if plane.w < 0.0 { plane } else { -plane };
    let sign = |x: f32| if x >= 0.0 { 1.0f32 } else { -1.0f32 };
    // Where the camera's near and far planes land in clip z, probed
    // through `proj` itself — correct for every depth convention
    // (0..1, -1..1, and this renderer's reversed z).
    let probe = |z: f32| {
        let v = proj * Vec4::new(0.0, 0.0, z, 1.0);
        v.z / v.w
    };
    let (nz, fz) = (probe(-0.05), probe(-1.0e6));
    // Q: the frustum's near-plane corner in camera space, un-projected.
    let q = proj.inverse() * Vec4::new(sign(c.x), sign(c.y), nz, 1.0);
    let c2 = c * (2.0 / c.dot(q));
    // The z-producing column becomes the clip plane; `fz` keeps the far
    // clip where it was (Terathon's `+1` for the -1..1 convention).
    Mat4::from_cols(
        proj.x_axis,
        proj.y_axis,
        Vec4::new(c2.x, c2.y, c2.z + fz, c2.w),
        proj.w_axis,
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct QuadVertex {
    pub position: [f32; 3],
}

impl Renderer {
    /// Lazily sized offscreen targets, one per `View` this frame asks for.
    /// Recreated when the surface size changed since they were made.
    pub(super) fn portal_targets(&mut self, count: usize) -> usize {
        let count = count.min(MAX_VIEWS);
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        if self
            .portal_targets
            .first()
            .is_some_and(|t| t.color.width() != w || t.color.height() != h)
        {
            self.portal_targets.clear();
        }
        while self.portal_targets.len() < count {
            let color = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Portal view"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Portal depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let globals = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Portal globals"),
                size: std::mem::size_of::<Globals>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let color_view = color.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Portal quad"),
                layout: &self.portal_quad_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.globals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sprite_sampler),
                    },
                ],
            });
            let world_group = self.portal_world_group(&globals);
            self.portal_targets.push(Target {
                color,
                group,
                world_group,
                globals,
                color_view,
                depth_view,
            });
        }
        count
    }

    /// The bind group the world pipelines use inside a portal view: that
    /// view's `Globals` with the shared sprite atlas and lightmap.
    fn portal_world_group(&self, globals: &wgpu::Buffer) -> wgpu::BindGroup {
        super::terrain_group(
            &self.device,
            "Portal world",
            &self.terrain_layout,
            globals,
            &self.portal_sprite_views,
            &self.sprite_sampler,
            &self.lightmap_view,
            &self.lightmap_sampler,
        )
    }

    /// The atlas or lightmap changed: the portal views' world bind groups
    /// hold the same textures `self.group` does.
    pub(super) fn rebuild_portal_groups(&mut self) {
        for i in 0..self.portal_targets.len() {
            let globals = self.portal_targets[i].globals.clone();
            self.portal_targets[i].world_group = self.portal_world_group(&globals);
        }
    }

    /// Draw one portal view's world into its target: sky, then the
    /// destination's resident sections solid then (unsorted) translucent —
    /// the camera the translucency sort ran for is the main view's.
    pub(super) fn draw_portal_world(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        i: usize,
        view: &View,
        base: &Globals,
        visible: &[super::resident::Visible],
        views: &[View],
    ) {
        let Some(target) = self.portal_targets.get(i) else {
            return;
        };
        let mut globals = *base;
        globals.view_proj = view.view_proj.to_cols_array_2d();
        if let Some(sky) = &view.sky {
            globals.fog_color = [sky.fog_color[0], sky.fog_color[1], sky.fog_color[2], 1.0];
            globals.sky_color = [sky.sky_color[0], sky.sky_color[1], sky.sky_color[2], 1.0];
        }
        self.uploads
            .buffer(encoder, &target.globals, 0, bytemuck::bytes_of(&globals));
        // Section draws for this view: solids' facing runs first, every
        // translucent after them so it composites over the solids.
        let mut draws = std::mem::take(&mut self.portal_draws);
        let mut pages = std::mem::take(&mut self.portal_pages);
        let mut origins = std::mem::take(&mut self.portal_origins);
        draws.clear();
        pages.clear();
        origins.clear();
        let eye = super::projection_center(view.view_proj).filter(|_| self.sides_on);
        let translucent_from;
        {
            let sections = self.sections_for(view.world);
            for (n, v) in visible.iter().enumerate() {
                let s = sections.get(v.slot);
                origins.push(super::section_instance(v.origin));
                super::facing_runs(&s.sides, eye, v.origin, |first, quads| {
                    draws.push([
                        quads * 6,
                        1,
                        0,
                        (s.start + u64::from(first) * 4) as u32,
                        n as u32,
                    ]);
                    pages.push((s.page, None));
                });
            }
            translucent_from = draws.len();
            for (n, v) in visible.iter().enumerate() {
                let s = sections.get(v.slot);
                if s.translucent > 0 {
                    draws.push([
                        s.translucent * 6,
                        1,
                        0,
                        (s.start + u64::from(s.solid) * 4) as u32,
                        n as u32,
                    ]);
                    pages.push((s.page, None));
                }
            }
        }
        self.portal_draws = draws;
        self.portal_pages = pages;
        self.portal_origins = origins;
        if !self.portal_origins.is_empty() {
            if self.portal_origins.len() as u64 > self.portal_instance_capacity {
                self.portal_instance_capacity =
                    (self.portal_origins.len() as u64).next_power_of_two();
                self.portal_instances = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Portal section origins"),
                    size: self.portal_instance_capacity * 16,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            self.uploads.buffer(
                encoder,
                &self.portal_instances,
                0,
                bytemuck::cast_slice(&self.portal_origins),
            );
        }
        // The view's entities upload before the pass holds `encoder`.
        let entity_quads = (view.entities.len() / 4) as u64;
        if entity_quads > 0 {
            if entity_quads > self.portal_entity_capacity {
                self.portal_entity_capacity = entity_quads.next_power_of_two();
                self.portal_entities = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Portal entity vertices"),
                    size: self.portal_entity_capacity
                        * std::mem::size_of::<crate::mesher::Vertex>() as u64
                        * 4,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            self.uploads.buffer(
                encoder,
                &self.portal_entities,
                0,
                bytemuck::cast_slice(&view.entities),
            );
        }
        let fog = globals.fog_color;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Portal view"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(fog[0]),
                        g: f64::from(fog[1]),
                        b: f64::from(fog[2]),
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, &target.world_group, &[]);
        if view.skybox {
            pass.set_pipeline(&self.sky);
            pass.draw(0..3, 0..1);
        }
        if self.portal_draws.is_empty() {
            return;
        }
        pass.set_pipeline(&self.solid);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(1, self.portal_instances.slice(..));
        self.draw_direct(&mut pass, 0, translucent_from);
        pass.set_pipeline(&self.translucent);
        self.draw_direct(&mut pass, translucent_from, self.portal_draws.len());
        // Entities through the portal: a second bake for this view.
        if entity_quads > 0 {
            pass.set_pipeline(&self.entity_pipeline);
            pass.set_bind_group(0, &self.entity_group, &[]);
            pass.set_vertex_buffer(0, self.portal_entities.slice(..));
            pass.draw_indexed(0..(entity_quads as u32) * 6, 0, 0..1);
            pass.set_bind_group(0, &target.world_group, &[]);
        }
        // Portal-in-portal: inner views rendered into their targets already
        // (view passes run in reverse index order, deepest first); the
        // inner quad samples that target through this pass's camera.
        for (j, inner) in views.iter().enumerate() {
            if inner.parent != Some(i) {
                continue;
            }
            let Some(itarget) = self.portal_targets.get(j) else {
                continue;
            };
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Portal inner quad"),
                layout: &self.portal_quad_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: target.globals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&itarget.color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sprite_sampler),
                    },
                ],
            });
            pass.set_pipeline(&self.portal_quad);
            pass.set_bind_group(0, &group, &[]);
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(0, self.portal_quads.slice(..));
            pass.draw_indexed(0..6, (j * 4) as i32, 0..1);
            pass.set_bind_group(0, &target.world_group, &[]);
        }
    }

    /// Draw entries into a pass without the indirect buffer: the same
    /// `(page, index_page)` run grouping as `draw_sections`, one
    /// `draw_indexed` per entry (portal views are few and small).
    fn draw_direct(&self, pass: &mut wgpu::RenderPass<'_>, first: usize, end: usize) {
        let (draws, pages) = (&self.portal_draws[..], &self.portal_pages[..]);
        let (mut i, end) = (first, end.min(draws.len()));
        while i < end {
            let (page, index_page) = pages[i];
            let run = pages[i..end]
                .iter()
                .take_while(|&&p| p == (page, index_page))
                .count();
            pass.set_vertex_buffer(0, self.arena.buffer(page).slice(..));
            let indices = index_page.map_or(&self.indices, |p| self.sorted_indices.buffer(p));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            for d in &draws[i..i + run] {
                pass.draw_indexed(d[2]..d[2] + d[0], d[3] as i32, d[4]..d[4] + 1);
            }
            i += run;
        }
    }
}
