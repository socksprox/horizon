// A portal's surface: the world rendered through it (gfx/portal.rs) is
// sampled where the quad covers the screen, so the quad's own colour comes
// entirely from the portal view texture, not the block atlas.
struct Globals {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog: vec4<f32>,
    sky_color: vec4<f32>,
    misc: vec4<f32>,
    inverse_view: mat4x4<f32>,
    sunrise: vec4<f32>,
    sun: vec4<f32>,
    fog_ends: vec4<f32>,
    end_portal_sprites: array<vec4<f32>, 2>,
    time: vec4<f32>,
    crumbling_sprite: vec4<f32>,
    outline_params: vec4<f32>,
    dynamic_lights: array<vec4<f32>, 16>,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var portal_tex: texture_2d<f32>;
@group(0) @binding(2) var portal_sampler: sampler;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex
fn vs_portal(@location(0) pos: vec3<f32>) -> Out {
    var o: Out;
    o.clip = globals.view_proj * vec4<f32>(pos, 1.0);
    // Screen UV of this fragment: the portal texture is a full-size view,
    // sampled 1:1 where the quad projects.
    let ndc = o.clip.xy / o.clip.w;
    o.uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return o;
}
@fragment
fn fs_portal(in: Out) -> @location(0) vec4<f32> {
    return textureSample(portal_tex, portal_sampler, in.uv);
}
