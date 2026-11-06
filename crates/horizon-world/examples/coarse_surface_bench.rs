//! `GeneratorWorker::coarse_surface` accuracy and cost against the generated
//! terrain of the same chunks (the far ring's height lattice).
//!
//! coarse_surface_bench <client resource dir> [--seed S] [--center CX CZ] [--radius R]
use horizon_core::position::ChunkPos;
use horizon_world::worldgen::generator::{GeneratorData, OverworldGenerator, ProtoChunk};
use std::{path::PathBuf, time::Instant};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let resources = PathBuf::from(args.next().ok_or("usage: coarse_surface_bench <resources>")?);
    let (mut seed, mut center, mut radius) = (0i64, (0, 0), 3);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => seed = args.next().ok_or("--seed S")?.parse()?,
            "--center" => {
                center = (
                    args.next().ok_or("--center CX CZ")?.parse()?,
                    args.next().ok_or("--center CX CZ")?.parse()?,
                )
            }
            "--radius" => radius = args.next().ok_or("--radius R")?.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let registry = Box::leak(Box::new(horizon_world::level::data::load_registry(
        &resources,
    )?));
    let data = GeneratorData::load(&resources.join("worldgen"), registry)?;
    let generator =
        OverworldGenerator::for_dimension(data, seed, "minecraft:normal", "minecraft:overworld")?;
    let mut worker = generator.worker();
    // Warm the worker's caches once.
    worker.coarse_surface(0, 0, 8, 3, 2)?;
    run(&mut worker, seed, center.0, center.1, radius);
    Ok(())
}

/// Terrain top (ocean floor) of a generated chunk column: the first block
/// below the surface that is neither air nor water.
fn ground(proto: &ProtoChunk, x: i32, z: i32) -> (i32, bool) {
    let top = proto.surface_height(x, z);
    let mut y = top - 1;
    while y > -64 {
        let name = proto.block_name(x, y, z);
        if name != "minecraft:air" && name != "minecraft:water" {
            break;
        }
        y -= 1;
    }
    (y + 1, y + 1 < top)
}

/// Signed errors (coarse - generated): mean/p90/max of |error|, and how many
/// are over 8 blocks too high (the generator carved the surface away: caves,
/// ravines, underwater canyons) or too low (the lattice missed an overhang).
fn report(tag: &str, errors: &mut Vec<i32>) {
    let (high, low) = (
        errors.iter().filter(|&&e| e > 8).count(),
        errors.iter().filter(|&&e| e < -8).count(),
    );
    let mut abs: Vec<i32> = errors.iter().map(|e| e.abs()).collect();
    abs.sort_unstable();
    let n = abs.len().max(1);
    let mean = abs.iter().map(|&e| f64::from(e)).sum::<f64>() / n as f64;
    println!(
        "{tag:<34} n={n:<6} mean {mean:5.2}  median {:3}  p90 {:3}  p99 {:3}  max {:3}  >8 high: {high}  >8 low: {low}",
        abs[n / 2],
        abs[n * 9 / 10],
        abs[n * 99 / 100],
        abs[n - 1],
    );
}

fn run(worker: &mut horizon_world::worldgen::generator::GeneratorWorker, _seed: i64, cx: i32, cz: i32, radius: i32) {
    let (mut t_proto, mut t4, mut t8, mut chunks) = (0.0, 0.0, 0.0, 0);
    let (mut lattice4, mut lattice8, mut recon4, mut recon8) = (vec![], vec![], vec![], vec![]);
    let mut biome_same = (0, 0);
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let pos = ChunkPos::new(cx + dx, cz + dz);
            let (x0, z0) = (pos.x * 16, pos.z * 16);
            let t = Instant::now();
            let proto = worker.generate_proto(pos).expect("proto");
            t_proto += t.elapsed().as_secs_f64() * 1e3;
            let mut truth = [[0i32; 16]; 16];
            for lz in 0..16 {
                for lx in 0..16 {
                    truth[lz][lx] = ground(&proto, x0 + lx as i32, z0 + lz as i32).0;
                }
            }
            for (step, side) in [(4, 5usize), (8, 3usize)] {
                // A tile owns its low-corner lattice points; the 4-grid costs
                // 16 columns amortised (the high edge is its neighbours').
                let owned = side - 1;
                let t = Instant::now();
                let cols = worker
                    .coarse_surface(x0, z0, step, side, 2)
                    .expect("coarse");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                // Cost as if only the owned columns were sampled.
                let scale = (owned * owned) as f64 / (side * side) as f64;
                if step == 4 { t4 += ms * scale; } else { t8 += ms * scale; }
                let lat = if step == 4 { &mut lattice4 } else { &mut lattice8 };
                for lz in 0..side {
                    for lx in 0..side {
                        let (px, pz) = (lx * step as usize, lz * step as usize);
                        if px < 16 && pz < 16 {
                            let signed = cols[lz * side + lx].height - truth[pz][px];
                            if step == 4 && signed < -8 && std::env::var_os("OUTLIERS").is_some() {
                                println!(
                                    "outlier chunk {},{} at +{px},+{pz}: coarse {} truth {} (coarse - truth {signed})",
                                    pos.x, pos.z, cols[lz * side + lx].height, truth[pz][px]
                                );
                            }
                            lat.push(signed);
                        }
                    }
                }
                // Bilinear reconstruction of every block column of the chunk.
                let rec = if step == 4 { &mut recon4 } else { &mut recon8 };
                for lz in 0..16usize {
                    for lx in 0..16usize {
                        let (fx, fz) = (lx as f32 / step as f32, lz as f32 / step as f32);
                        let (ix, iz) = (fx.floor() as usize, fz.floor() as usize);
                        let (tx, tz) = (fx - ix as f32, fz - iz as f32);
                        let h = |i: usize, j: usize| cols[j * side + i].height as f32;
                        let top = h(ix, iz) * (1. - tx) + h(ix + 1, iz) * tx;
                        let bottom = h(ix, iz + 1) * (1. - tx) + h(ix + 1, iz + 1) * tx;
                        let r = top * (1. - tz) + bottom * tz;
                        rec.push(r.round() as i32 - truth[lz][lx]);
                    }
                }
                if step == 8 {
                    biome_same.1 += 1;
                    if proto.biome_id_at(x0 + 4, cols[0].height.min(300), z0 + 4) == cols[0].biome {
                        biome_same.0 += 1;
                    }
                }
            }
            chunks += 1;
        }
    }
    let n = f64::from(chunks);
    println!("chunks {chunks}: generate_proto {:.1} ms/chunk (host)", t_proto / n);
    println!("coarse 4-block grid (16 owned columns) {:.2} ms/chunk", t4 / n);
    println!("coarse 8-block grid ( 4 owned columns) {:.2} ms/chunk", t8 / n);
    report("lattice points, 4-grid", &mut lattice4);
    report("lattice points, 8-grid", &mut lattice8);
    report("all columns bilinear from 4-grid", &mut recon4);
    report("all columns bilinear from 8-grid", &mut recon8);
    println!("biome at corner matches proto's: {}/{}", biome_same.0, biome_same.1);
}