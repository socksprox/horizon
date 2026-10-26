//! Remote dimensions the client holds for see-through portals
//! (`game/portals.rs`): a portal whose destination is another dimension
//! renders that level's world, so the server streams the chunks around
//! each portal's exit to clients (`horizon:remote_chunk` /
//! `horizon:remote_forget`), and the client meshes them beside its own.
//!
//! Each `RemoteDim` runs the same `Streaming` a remote level does, with a
//! small fixed radius centred on the portal's exit; its meshes go to
//! `Renderer.portal_worlds[i]`, which `portal::View::world` names.

use super::{Dim, Game, remote::NoTerrain};
use horizon_client::{chunk::SECTIONS, portals, streaming::Streaming, world::World};
use horizon_protocol::packets::{configuration as cfg, play};
use std::{collections::HashSet, sync::Arc};

/// Chunks around a portal's exit the client sees through it (radius).
const REMOTE_RADIUS: i32 = 4;
/// Columns streamed per server tick across all guests, so a portal far
/// away does not stall the hub.
const SENDS_PER_TICK: usize = 32;

/// A dimension held beside the remote level for portal views.
pub(super) struct RemoteDim {
    /// Its level name (`minecraft:the_nether`), matching `Portal.dest_dim`.
    pub dim: String,
    /// Its dimension type for the chunk decoder (`remote_chunk`).
    pub dim_type: String,
    pub world: World,
    pub streaming: Streaming,
    /// Entities near the portal exits (`horizon:remote_entities`), drawn
    /// as markers inside this dimension's portal views.
    pub entities: Vec<portals::RemoteEntity>,
}

impl Game<'_> {
    /// `remote_worlds`' index of `dim`, creating it (empty world, remote
    /// streaming around 0,0 until a portal gives a centre).
    pub(super) fn ensure_remote_world(&mut self, dim: &str, dim_type: &str) -> usize {
        if let Some(i) = self.remote_worlds.iter().position(|w| w.dim == dim) {
            return i;
        }
        let mut streaming = Streaming::new(
            Arc::new(NoTerrain),
            self.pack.context.clone(),
            self.res.threads(),
        );
        streaming.remote = true;
        streaming.radius = REMOTE_RADIUS;
        self.remote_worlds.push(RemoteDim {
            dim: dim.to_owned(),
            dim_type: dim_type.to_owned(),
            world: World::new(self.res.facts().air, 0),
            streaming,
            entities: Vec::new(),
        });
        self.remote_worlds.len() - 1
    }

    /// A `horizon:remote_chunk` payload: the column joins its world and
    /// its sections queue for meshing.
    pub(super) fn remote_world_chunk(&mut self, data: &[u8]) {
        let Some((dim, dim_type, packet)) = portals::decode_remote_chunk(data) else {
            return;
        };
        let i = self.ensure_remote_world(&dim, &dim_type);
        // The destination's chunk is the streaming centre (and view radius).
        if let Some(p) = self
            .imm_portals
            .iter()
            .find(|p| p.dest_dim == self.remote_worlds[i].dim)
        {
            self.remote_worlds[i].streaming.center = (
                (p.dest_center.x.floor() as i32) >> 4,
                (p.dest_center.z.floor() as i32) >> 4,
            );
        }
        let d = self.remote.as_ref().and_then(|r| {
            r.registries
                .id("minecraft:dimension_type", &dim_type)
                .map(|id| r.registries.dimension(id))
        });
        match d {
            Some(d) => match self.remote_chunk_for(&packet, &d) {
                Ok(chunk) => {
                    let rw = &mut self.remote_worlds[i];
                    rw.streaming.insert_remote(&mut rw.world, chunk);
                }
                Err(e) => log::warn!("Horizon: remote chunk {dim}: {e}"),
            },
            None => log::warn!("Horizon: remote chunk for unknown {dim_type}"),
        }
    }

    /// `horizon:remote_entities` payloads give remote portal views their
    /// entities: the parked level's mobs plus the guests inside it,
    /// around each portal exit, at five times a second.
    pub(super) fn stream_remote_entities(&mut self) {
        if self.ticks % 4 != 0 {
            return;
        }
        let active = self.dim.id().to_owned();
        // Each portal into another dimension wants that side's entities.
        let want: Vec<(Dim, horizon_core::geometry::Vec3)> = self
            .imm_portals
            .iter()
            .filter(|p| p.dim == active && p.dest_dim != active)
            .map(|p| (Dim::from_id(&p.dest_dim), p.dest_center))
            .collect();
        if want.is_empty() {
            return;
        }
        let Some(host) = self.lan.as_mut() else {
            return;
        };
        let radius = (REMOTE_RADIUS * 16) as f64;
        for (dim, center) in want {
            let mut entities = Vec::new();
            for m in self.levels.parked_mobs(dim) {
                let d = (m.position.x - center.x)
                    .abs()
                    .max((m.position.z - center.z).abs());
                if d <= radius {
                    entities.push(portals::RemoteEntity {
                        pos: [m.position.x, m.position.y, m.position.z],
                        yaw: m.yaw,
                        pitch: 0.0,
                        kind: 0,
                    });
                }
            }
            for g in &host.hub.guests {
                if g.placed && g.dimension == dim.id() {
                    entities.push(portals::RemoteEntity {
                        pos: g.position,
                        yaw: g.yaw,
                        pitch: g.pitch,
                        kind: 1,
                    });
                }
            }
            let dim_name = dim.id().to_owned();
            let data = portals::encode_remote_entities(&dim_name, &entities);
            for g in &host.hub.guests {
                if g.placed && g.dimension == active {
                    g.send(&play::Clientbound::CustomPayload(cfg::CustomPayload {
                        channel: portals::REMOTE_ENTITIES_CHANNEL.into(),
                        data: data.clone(),
                    }));
                }
            }
        }
    }

    /// A `horizon:remote_entities` payload: the entity markers a remote
    /// world shows inside its portal views this tick.
    pub(super) fn remote_world_entities(&mut self, data: &[u8]) {
        let Some((dim, entities)) = portals::decode_remote_entities(data) else {
            return;
        };
        let i = self.ensure_remote_world(&dim, &dim);
        self.remote_worlds[i].entities = entities;
    }

    /// A `horizon:remote_forget` payload: drop the column and its meshes.
    pub(super) fn remote_world_forget(&mut self, data: &[u8]) {
        let Some((dim, x, z)) = portals::decode_remote_forget(data) else {
            return;
        };
        let Some(i) = self.remote_worlds.iter().position(|w| w.dim == dim) else {
            return;
        };
        let rw = &mut self.remote_worlds[i];
        rw.streaming.forget(&mut rw.world, x, z);
        if let Some(renderer) = self.renderer.as_mut() {
            for sy in 0..SECTIONS as i32 {
                renderer.remove_portal_section((x, sy, z), i + 1);
            }
        }
    }

    /// Mesh finished sections of every remote world into its resident
    /// (`update_world`, after the remote level's own streaming).
    pub(super) fn update_remote_worlds(&mut self) {
        for (i, rw) in self.remote_worlds.iter_mut().enumerate() {
            if rw.world.dirty_sections.is_empty() {
                continue;
            }
            let (uploads, _) = rw.streaming.update(&mut rw.world);
            if let Some(renderer) = self.renderer.as_mut() {
                for (key, mesh) in uploads {
                    renderer.upload_portal(key, &mesh, i + 1);
                }
            }
        }
        // Remote worlds no portal reaches any more are dropped.
        let active = self.dim.id();
        let needed: HashSet<String> = self
            .imm_portals
            .iter()
            .filter(|p| p.dim == active && p.dest_dim != active)
            .map(|p| p.dest_dim.clone())
            .collect();
        let mut at = 0;
        while at < self.remote_worlds.len() {
            if needed.contains(&self.remote_worlds[at].dim) {
                at += 1;
                continue;
            }
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.clear_portal_world(at);
            }
            self.remote_worlds.remove(at);
        }
    }

    /// Server side (`tick_world`): every guest gets the chunks around the
    /// exit of each portal whose far side is a parked level, once per
    /// column they have not been sent.
    pub(super) fn stream_remote_worlds(&mut self) {
        let active = self.dim.id();
        // Columns this level's edits touched are stale in any guest's
        // remote copy of it (another level streams it through a portal):
        // forget them so the send below re-pushes (`block_log` is this
        // tick's changes, before `broadcast_block_changes` drains it).
        if let Some(log) = &self.world.block_log
            && !log.is_empty()
        {
            let touched: HashSet<(i32, i32)> = log.iter().map(|p| (p.0 >> 4, p.2 >> 4)).collect();
            let dim = active.to_owned();
            for sent in self.remote_sent.values_mut() {
                sent.retain(|(d, x, z)| *d != dim || !touched.contains(&(*x, *z)));
            }
        }
        // (destination dim, chunk x, chunk z) around every exit.
        let want: Vec<(Dim, i32, i32)> = self
            .imm_portals
            .iter()
            .filter(|p| p.dim == active && p.dest_dim != active)
            .map(|p| {
                (
                    Dim::from_id(&p.dest_dim),
                    (p.dest_center.x.floor() as i32) >> 4,
                    (p.dest_center.z.floor() as i32) >> 4,
                )
            })
            .collect();
        if want.is_empty() {
            return;
        }
        // Exits stay open (`levels_to_tick` keeps them alive and
        // `update_dedicated_streaming` aims their streamer at the exit).
        for &(dim, _, _) in &want {
            if let Err(e) = self.ensure_level(dim) {
                log::warn!("Horizon: portal exit {dim:?}: {e}");
            }
        }
        let mut sent = SENDS_PER_TICK;
        for (dim, cx, cz) in want {
            let Some(world) = self.levels.parked_world(dim) else {
                continue;
            };
            let columns: Vec<(i32, i32)> = (cx - REMOTE_RADIUS..=cx + REMOTE_RADIUS)
                .flat_map(|x| (cz - REMOTE_RADIUS..=cz + REMOTE_RADIUS).map(move |z| (x, z)))
                .filter(|k| world.chunks.contains_key(k))
                .collect();
            if columns.is_empty() {
                continue;
            }
            let Some(host) = self.lan.as_mut() else {
                return;
            };
            // Guests in another level get this level's blocks live.
            let d = horizon_protocol::registry::DimensionType::vanilla(dim.id());
            host.hub
                .encoder
                .set_dimension(d.min_y, d.height, d.has_skylight);
            let dim_name = dim.id().to_owned();
            for g in &host.hub.guests {
                if sent == 0 {
                    break;
                }
                // Only viewers on the portal's side stream its far level.
                if !g.placed || g.dimension != active {
                    continue;
                }
                let seen = self.remote_sent.entry(g.conn).or_default();
                for &(x, z) in &columns {
                    if sent == 0 {
                        break;
                    }
                    let key = (dim_name.clone(), x, z);
                    if seen.contains(&key) {
                        continue;
                    }
                    let Some(chunk) = world.chunks.get(&(x, z)) else {
                        continue;
                    };
                    let packet = host.hub.encoder.encode(chunk, &world.block_entities);
                    let data = portals::encode_remote_chunk(&dim_name, &dim_name, &packet);
                    g.send(&play::Clientbound::CustomPayload(cfg::CustomPayload {
                        channel: portals::REMOTE_CHUNK_CHANNEL.into(),
                        data,
                    }));
                    seen.insert(key);
                    sent -= 1;
                }
            }
            let a = horizon_protocol::registry::DimensionType::vanilla(active);
            host.hub
                .encoder
                .set_dimension(a.min_y, a.height, a.has_skylight);
        }
    }
}
