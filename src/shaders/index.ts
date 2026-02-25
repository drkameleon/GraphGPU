// ============================================================
// graphGPU - WGSL Shaders
// ============================================================

/** Node vertex + fragment shader (instanced quads with SDF circles + selection halo) */
export const NODE_SHADER = /* wgsl */ `

struct Camera {
  col0: vec4f,
  col1: vec4f,
  col2: vec4f,
};

struct FrameUniforms {
  camera: Camera,
  viewport: vec2f,
  time: f32,
  nodeScale: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
  @location(1) uv: vec2f,
  @location(2) selection: f32,
};

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
  vec2f(-1, 1),  vec2f(1, -1), vec2f(1, 1),
);

fn applyCamera(pos: vec2f) -> vec2f {
  let cam = frame.camera;
  return vec2f(
    pos.x * cam.col0.x + pos.y * cam.col1.x + cam.col2.x,
    pos.x * cam.col0.y + pos.y * cam.col1.y + cam.col2.y,
  );
}

fn getCameraZoom() -> f32 {
  return length(vec2f(frame.camera.col0.x, frame.camera.col0.y));
}

@vertex
fn vs_node(
  @builtin(vertex_index) vi: u32,
  @location(0) center: vec2f,
  @location(1) color: vec4f,
  @location(2) size: f32,
  @location(3) selection: f32,
) -> VSOut {
  let q = QUAD[vi];
  let cameraZoom = getCameraZoom();

  // World-space sizing
  let worldSize = size * frame.nodeScale * 0.01;

  // Halo expansion when selected (thin outline)
  let haloExpand = selection * 0.15;
  let finalWorldSize = worldSize * (1.0 + haloExpand);

  let projected = applyCamera(center);
  let ndcSize = vec2f(
    finalWorldSize * cameraZoom,
    finalWorldSize * cameraZoom * (frame.viewport.x / frame.viewport.y)
  );
  let pos = projected + q * ndcSize;

  var out: VSOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.color = color;
  out.uv = q;
  out.selection = selection;
  return out;
}

@fragment
fn fs_node(in: VSOut) -> @location(0) vec4f {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }

  let aa = fwidth(dist);
  let edge = smoothstep(1.0, 1.0 - aa * 2.0, dist);

  // Flat fill - no glow, no 3D shading
  let baseColor = in.color.rgb;
  var finalColor = baseColor;

  // Thin dark border at the edge
  let borderStart = 0.85;
  let borderMask = smoothstep(borderStart - aa, borderStart, dist);
  finalColor = mix(finalColor, baseColor * 0.6, borderMask * 0.4);

  // Selection: contrast-adaptive outline ring in the expanded area
  if (in.selection > 0.5) {
    let innerR = 1.0 / (1.0 + 0.15);  // matches haloExpand
    let ring = smoothstep(innerR - aa, innerR, dist) * smoothstep(1.0, 1.0 - aa * 2.0, dist);
    // Choose halo color based on node brightness: dark halo on light nodes, light on dark
    let lum = dot(baseColor, vec3f(0.2126, 0.7152, 0.0722));
    let haloColor = select(vec3f(1.0, 1.0, 1.0), vec3f(0.1, 0.1, 0.15), lum > 0.45);
    finalColor = mix(finalColor, haloColor, ring);
  }

  return vec4f(finalColor * edge, in.color.a * edge);
}
`;

/**
 * Edge shader - instanced quads between node pairs.
 */
export const EDGE_SHADER = /* wgsl */ `

struct Camera {
  col0: vec4f,
  col1: vec4f,
  col2: vec4f,
};

struct FrameUniforms {
  camera: Camera,
  viewport: vec2f,
  time: f32,
  edgeOpacity: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) alpha: f32,
  @location(1) vEdge: f32,
  @location(2) color: vec3f,
};

fn applyCamera(pos: vec2f) -> vec2f {
  let cam = frame.camera;
  return vec2f(
    pos.x * cam.col0.x + pos.y * cam.col1.x + cam.col2.x,
    pos.x * cam.col0.y + pos.y * cam.col1.y + cam.col2.y,
  );
}

// Per-vertex data for edge quad (6 verts = 2 tris)
// t: position along edge (0=source, 1=target)
// s: perpendicular offset (-1=left, +1=right)
var<private> ET: array<f32, 6> = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
var<private> ES: array<f32, 6> = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);

@vertex
fn vs_edge(
  @builtin(vertex_index) vi: u32,
  @location(0) srcPos: vec2f,
  @location(1) tgtPos: vec2f,
  @location(2) alpha: f32,
  @location(3) edgeColor: vec3f,
) -> VSOut {
  let t = ET[vi];
  let s = ES[vi];

  let srcProj = applyCamera(srcPos);
  let tgtProj = applyCamera(tgtPos);
  let dir = tgtProj - srcProj;
  let len = length(dir);

  var out: VSOut;

  if (len < 0.0001) {
    out.position = vec4f(srcProj, 0.0, 1.0);
    out.alpha = 0.0;
    out.vEdge = 0.0;
    out.color = edgeColor;
    return out;
  }

  let fwd = dir / len;
  let perp = vec2f(-fwd.y, fwd.x);

  // Width: 3px minimum on screen, grows with zoom
  let minWidth = 3.0 / frame.viewport.y;
  let zoomWidth = 0.004 * length(vec2f(frame.camera.col0.x, frame.camera.col0.y));
  let w = max(minWidth, zoomWidth);

  let pos = mix(srcProj, tgtProj, t) + perp * s * w;

  out.position = vec4f(pos, 0.0, 1.0);
  out.alpha = alpha * frame.edgeOpacity;
  out.vEdge = s;  // raw -1 to +1, interpolated across quad face
  out.color = edgeColor;
  return out;
}

@fragment
fn fs_edge(in: VSOut) -> @location(0) vec4f {
  let d = abs(in.vEdge);  // 0 at center of edge, 1 at border
  let aa = fwidth(d);
  let edgeFade = smoothstep(1.0, 1.0 - aa * 3.0, d);
  return vec4f(in.color, in.alpha * edgeFade);
}
`;

/** GPU Compute shader for force-directed layout */
export const FORCE_COMPUTE_SHADER = /* wgsl */ `

struct ForceParams {
  nodeCount: u32,
  edgeCount: u32,
  repulsion: f32,
  attraction: f32,
  gravity: f32,
  damping: f32,
  deltaTime: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(3) var<storage, read> edges: array<vec2u>;
@group(0) @binding(4) var<storage, read> sizes: array<f32>;

@compute @workgroup_size(64)
fn cs_repulsion(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }

  let pos_i = positions[i];
  var force = vec2f(0.0, 0.0);

  for (var j = 0u; j < params.nodeCount; j++) {
    if (i == j) { continue; }
    let delta = pos_i - positions[j];
    let dist = max(length(delta), 0.001);
    force += (delta / dist) * params.repulsion / (dist * dist);
  }

  force -= pos_i * params.gravity;
  velocities[i] = (velocities[i] + force * params.deltaTime) * params.damping;
}

@compute @workgroup_size(64)
fn cs_attraction(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.edgeCount) { return; }

  let edge = edges[i];
  let delta = positions[edge.y] - positions[edge.x];
  let dist = max(length(delta), 0.001);
  let force = (delta / dist) * (dist - 1.0) * params.attraction;

  velocities[edge.x] += force * params.deltaTime;
  velocities[edge.y] -= force * params.deltaTime;
}

@compute @workgroup_size(64)
fn cs_integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  if (sizes[i] <= 0.0) { return; }
  positions[i] += velocities[i] * params.deltaTime;
}
`;
