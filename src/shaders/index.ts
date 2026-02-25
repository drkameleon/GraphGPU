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

  // Selection: ring outline using node's own color, shifted for contrast
  if (in.selection > 0.5) {
    let innerR = 1.0 / (1.0 + 0.15);  // matches haloExpand
    let ring = smoothstep(innerR - aa * 1.5, innerR, dist) * smoothstep(1.0, 1.0 - aa * 3.0, dist);
    // Use node's own color: lighter version for dark nodes, darker for bright nodes
    let lum = dot(baseColor, vec3f(0.2126, 0.7152, 0.0722));
    // Lighten: mix toward white but keep color identity
    let lighter = mix(baseColor, vec3f(1.0), 0.55);
    // Darken: multiply down but keep saturation
    let darker = baseColor * 0.35;
    var haloColor = select(lighter, darker, lum > 0.5);
    finalColor = mix(finalColor, haloColor, ring * 0.95);
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
  edgeOpacity: f32,
  nodeScale: f32,
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

  // Width: equal to projected node radius (deliberately oversized for testing)
  let cameraZoom = length(vec2f(frame.camera.col0.x, frame.camera.col0.y));
  let nodeWorldSize = frame.nodeScale * 0.01;
  let projectedNodeR = nodeWorldSize * cameraZoom;
  let minWidth = 2.5 / frame.viewport.y;
  let w = max(minWidth, projectedNodeR * 1.0);

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

/** GPU Compute shader for vis.js-style force-directed layout */
export const FORCE_COMPUTE_SHADER = /* wgsl */ `

struct ForceParams {
  nodeCount: u32,
  edgeCount: u32,
  gravitationalConstant: f32,  // negative = repulsive (default: -0.25)
  springLength: f32,           // rest length (default: 0.2)
  springConstant: f32,         // Hooke stiffness (default: 0.06)
  centralGravity: f32,         // pull toward origin (default: 0.012)
  damping: f32,                // viscous friction (default: 0.18)
  timestep: f32,               // integration dt (default: 0.35)
  maxVelocity: f32,            // velocity clamp (default: 0.06)
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(3) var<storage, read> edges: array<vec2u>;
@group(0) @binding(4) var<storage, read> sizes: array<f32>;

// Forces buffer: accumulated per node, reset each step
@group(0) @binding(5) var<storage, read_write> forces: array<vec2f>;

// ── Pass 1: Reset forces to zero ──
@compute @workgroup_size(64)
fn cs_reset_forces(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  forces[i] = vec2f(0.0, 0.0);
}

// ── Pass 2: Gravitational repulsion (N-body, O(n²)) ──
// vis.js Barnes-Hut: F = G * m1 * m2 / dist³ (dividing by dist³ gives direction)
@compute @workgroup_size(64)
fn cs_repulsion(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  if (sizes[i] <= 0.0) { return; }

  let pos_i = positions[i];
  var force = vec2f(0.0, 0.0);
  let G = params.gravitationalConstant;

  for (var j = 0u; j < params.nodeCount; j++) {
    if (i == j || sizes[j] <= 0.0) { continue; }
    var delta = pos_i - positions[j];
    var dist = length(delta);
    if (dist < 0.0001) {
      // Jitter to prevent zero-distance singularity
      delta = vec2f(f32(i % 7u) * 0.01 - 0.03, f32(j % 7u) * 0.01 - 0.03);
      dist = length(delta);
    }
    // Gravitational force: F = G / dist² (G is negative → repulsive)
    let forceMag = G / (dist * dist);
    force -= (delta / dist) * forceMag;
  }

  forces[i] += force;
}

// ── Pass 3: Spring attraction along edges (Hooke's law) ──
// vis.js: springForce = springConstant * (springLength - distance) / distance
@compute @workgroup_size(64)
fn cs_attraction(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= params.edgeCount) { return; }

  let edge = edges[e];
  let src = edge.x;
  let tgt = edge.y;
  if (sizes[src] <= 0.0 || sizes[tgt] <= 0.0) { return; }

  let delta = positions[src] - positions[tgt];
  let dist = max(length(delta), 0.0001);

  // Hooke's law with rest length
  let springForce = params.springConstant * (params.springLength - dist) / dist;
  let fx = delta * springForce;

  // Note: data races on force writes are acceptable for force-directed layout.
  // The small errors are equivalent to noise and don't affect convergence.
  forces[src] += fx;
  forces[tgt] -= fx;
}

// ── Pass 4: Central gravity ──
// vis.js: gravityForce = centralGravity / distance (toward origin)
@compute @workgroup_size(64)
fn cs_gravity(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  if (sizes[i] <= 0.0) { return; }

  let pos = positions[i];
  let toCenter = -pos;
  let dist = max(length(toCenter), 0.0001);
  let gForce = params.centralGravity / dist;
  forces[i] += toCenter * gForce;
}

// ── Pass 5: Velocity integration + position update ──
// vis.js: a = (F - damping*v) / mass; v += a * dt; x += v * dt
@compute @workgroup_size(64)
fn cs_integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  if (sizes[i] <= 0.0) { return; }

  let dt = params.timestep;
  let damp = params.damping;
  let maxV = params.maxVelocity;

  var vel = velocities[i];
  let f = forces[i];

  // Acceleration = (force - damping * velocity) / mass (mass=1)
  let acc = f - damp * vel;
  vel += acc * dt;

  // Clamp velocity
  vel = clamp(vel, vec2f(-maxV), vec2f(maxV));

  // Update position
  positions[i] += vel * dt;
  velocities[i] = vel;
}
`;
