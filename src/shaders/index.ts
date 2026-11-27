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

  // Straight (non-premultiplied) alpha: the blend multiplies by alpha,
  // so the color must not be scaled by the edge factor as well
  return vec4f(finalColor, in.color.a * edge);
}
`;

/**
 * Edge shader - instanced quads between node pairs.
 *
 * Each instance only carries its endpoint ids and color; node
 * positions are read from the same storage buffer the nodes are
 * drawn from, so moving nodes never requires re-uploading edges.
 * Instances with color alpha 0 are removed slots and get culled.
 *
 * FIX: The perpendicular vector is now computed in screen-space
 * (aspect-corrected) so that edge width is uniform regardless of
 * the edge's angle.  Previously, the perpendicular was computed
 * directly in NDC, where X and Y have different scales (due to
 * aspectRatio), causing horizontal and vertical edges to appear
 * at different widths.
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
  edgeWidth: f32,   // fixed width in device pixels, 0 = scale with nodes
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) alpha: f32,
  @location(1) vEdge: f32,                            // signed distance from the center line, in pixels
  @location(2) color: vec3f,
  @location(3) @interpolate(flat) halfWidth: f32,     // line half-width, in pixels
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
  @location(0) ends: vec2u,
  @location(1) packedColor: vec4f,
) -> VSOut {
  let t = ET[vi];
  let s = ES[vi];
  let edgeColor = packedColor.rgb;

  var out: VSOut;

  // Removed edge slot: emit a degenerate vertex outside the clip volume
  if (packedColor.a == 0.0) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    out.alpha = 0.0;
    out.vEdge = 0.0;
    out.color = edgeColor;
    out.halfWidth = 0.0;
    return out;
  }

  let srcProj = applyCamera(positions[ends.x]);
  let tgtProj = applyCamera(positions[ends.y]);

  // ── Compute perpendicular in screen-space to fix aspect distortion ──
  let aspect = frame.viewport.x / frame.viewport.y;

  // Stretch NDC → square screen-space
  let screenSrc = vec2f(srcProj.x * aspect, srcProj.y);
  let screenTgt = vec2f(tgtProj.x * aspect, tgtProj.y);
  let screenDir = screenTgt - screenSrc;
  let screenLen = length(screenDir);

  if (screenLen < 0.0001) {
    out.position = vec4f(srcProj, 0.0, 1.0);
    out.alpha = 0.0;
    out.vEdge = 0.0;
    out.color = edgeColor;
    out.halfWidth = 0.0;
    return out;
  }

  let screenFwd = screenDir / screenLen;
  let screenPerp = vec2f(-screenFwd.y, screenFwd.x);

  // Convert perpendicular back to NDC space
  let ndcPerp = vec2f(screenPerp.x / aspect, screenPerp.y);

  // Width: based on projected node radius
  let cameraZoom = length(vec2f(frame.camera.col0.x, frame.camera.col0.y));
  let nodeWorldSize = frame.nodeScale * 0.01;
  let projectedNodeR = nodeWorldSize * cameraZoom;
  let minWidth = 2.5 / frame.viewport.y;

  // A perpendicular offset of 1 (in these units) spans viewport.y / 2
  // pixels. Grow the quad by 1px so the antialiasing ramp fits outside.
  let pxPerUnit = frame.viewport.y * 0.5;
  var halfWidthPx = max(minWidth, projectedNodeR) * pxPerUnit;
  if (frame.edgeWidth > 0.0) {
    halfWidthPx = frame.edgeWidth * 0.5;
  }
  let quadHalf = halfWidthPx + 1.0;

  let pos = mix(srcProj, tgtProj, t) + ndcPerp * s * (quadHalf / pxPerUnit);

  out.position = vec4f(pos, 0.0, 1.0);
  out.alpha = frame.edgeOpacity;
  out.vEdge = s * quadHalf;
  out.color = edgeColor;
  out.halfWidth = halfWidthPx;
  return out;
}

@fragment
fn fs_edge(in: VSOut) -> @location(0) vec4f {
  // Coverage of this pixel by the line, with a 1px ramp at its border.
  // Measured in pixels, so thin lines keep the same brightness wherever
  // they fall relative to the pixel grid (no shimmering while moving).
  let coverage = clamp(in.halfWidth + 0.5 - abs(in.vEdge), 0.0, 1.0);
  return vec4f(in.color, in.alpha * coverage);
}
`;

/**
 * GPU Compute shader for vis.js-style force-directed layout.
 *
 * Two passes per step:
 *   cs_forces     one thread per node: repulsion (all pairs, tiled
 *                 through workgroup memory), springs (the node's own
 *                 neighbor list, so no two threads write the same
 *                 node) and central gravity
 *   cs_integrate  velocity + position update
 */
const FORCE_PARAMS = /* wgsl */ `
struct ForceParams {
  nodeCount: u32,
  neighborCount: u32,
  gravitationalConstant: f32,  // negative = repulsive (default: -0.25)
  springLength: f32,           // rest length (default: 0.2)
  springConstant: f32,         // Hooke stiffness (default: 0.06)
  centralGravity: f32,         // pull toward origin (default: 0.012)
  damping: f32,                // viscous friction (default: 0.18)
  timestep: f32,               // integration dt (default: 0.35)
  maxVelocity: f32,            // velocity clamp (default: 0.06)
  gridDim: u32,                // grid repulsion: cells per side
  gridMinX: f32,               // grid repulsion: origin
  gridMinY: f32,
  cellSize: f32,               // grid repulsion: cell size (world units)
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};
`;

/** Springs (from node i's own neighbor list) + central gravity */
const SPRINGS_AND_GRAVITY = /* wgsl */ `
fn springsAndGravity(i: u32, pos_i: vec2f) -> vec2f {
  var force = vec2f(0.0, 0.0);
  // vis.js: springForce = springConstant * (springLength - distance) / distance
  let end = offsets[i + 1u];
  for (var e = offsets[i]; e < end; e++) {
    let j = neighbors[e];
    if (sizes[j] <= 0.0) { continue; }
    let delta = pos_i - positions[j];
    let dist = max(length(delta), 0.0001);
    force += delta * (params.springConstant * (params.springLength - dist) / dist);
  }
  // vis.js: gravityForce = centralGravity / distance
  let toCenter = -pos_i;
  let dc = max(length(toCenter), 0.0001);
  force += toCenter * (params.centralGravity / dc);
  return force;
}
`;

/** Deterministic unit vector for a coincident pair, antisymmetric in (a, b) */
const PAIR_DIRECTION = /* wgsl */ `
fn pairDirection(a: u32, b: u32) -> vec2f {
  let lo = min(a, b);
  let hi = max(a, b);
  var h = (lo * 0x9E3779B1u) ^ ((hi + 1u) * 0x85EBCA77u);
  h = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
  h = h ^ (h >> 13u);
  let angle = f32(h) / 4294967296.0 * 6.28318530718;
  let sign = select(-1.0, 1.0, a == lo);
  return vec2f(cos(angle), sin(angle)) * sign;
}
`;

export const FORCE_COMPUTE_SHADER = /* wgsl */ `
${FORCE_PARAMS}

@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
// Adjacency (CSR): node i's neighbors are neighbors[offsets[i] .. offsets[i + 1]]
@group(0) @binding(3) var<storage, read> neighbors: array<u32>;
@group(0) @binding(4) var<storage, read> sizes: array<f32>;
@group(0) @binding(5) var<storage, read_write> forces: array<vec2f>;
@group(0) @binding(6) var<storage, read> offsets: array<u32>;

const TILE: u32 = 64u;
var<workgroup> tilePos: array<vec2f, 64>;
var<workgroup> tileActive: array<u32, 64>;

${PAIR_DIRECTION}
${SPRINGS_AND_GRAVITY}

// ── Pass 1: all forces on node i ──
@compute @workgroup_size(64)
fn cs_forces(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let i = gid.x;
  let n = params.nodeCount;
  let inRange = i < n;

  var pos_i = vec2f(0.0, 0.0);
  var isActive = false;
  if (inRange) {
    pos_i = positions[i];
    isActive = sizes[i] > 0.0;
  }

  // Repulsion: F = G / dist² (G < 0 → push apart), over all nodes,
  // one tile of positions at a time through workgroup memory.
  // Every thread takes part in loading tiles (barriers need uniform flow).
  var force = vec2f(0.0, 0.0);
  let G = params.gravitationalConstant;
  let tiles = (n + TILE - 1u) / TILE;
  for (var t = 0u; t < tiles; t++) {
    let j = t * TILE + lid.x;
    if (j < n) {
      tilePos[lid.x] = positions[j];
      tileActive[lid.x] = select(0u, 1u, sizes[j] > 0.0);
    } else {
      tileActive[lid.x] = 0u;
    }
    workgroupBarrier();

    if (isActive) {
      for (var k = 0u; k < TILE; k++) {
        let jj = t * TILE + k;
        if (jj == i || tileActive[k] == 0u) { continue; }
        var delta = pos_i - tilePos[k];
        var dist = length(delta);
        if (dist < 0.0001) {
          // Coincident: push apart along a fixed per-pair direction
          dist = 0.01;
          delta = pairDirection(i, jj) * dist;
        }
        force -= (delta / dist) * (G / (dist * dist));
      }
    }
    workgroupBarrier();
  }

  if (!inRange) { return; }
  if (!isActive) {
    forces[i] = vec2f(0.0, 0.0);
    return;
  }

  forces[i] = force + springsAndGravity(i, pos_i);
}

// ── Pass 2: Velocity integration + position update ──
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

// ============================================================
// Grid-approximated repulsion (large graphs)
// ============================================================
// Per step: bin nodes into a gridDim² grid (counting sort: count,
// scan, scatter), sum each cell's mass & center of mass, then per
// node: exact repulsion from nodes in the surrounding 5×5 cells,
// and each farther cell's center of mass for the rest.
// Each kernel binds only what it needs (≤ 8 storage buffers).
// ============================================================

const GRID_NEAR = 2; // cells: exact repulsion within ±2 cells

const CELL_OF = /* wgsl */ `
fn cellCoords(p: vec2f) -> vec2i {
  let g = i32(params.gridDim);
  let c = vec2i(floor((p - vec2f(params.gridMinX, params.gridMinY)) / params.cellSize));
  return clamp(c, vec2i(0, 0), vec2i(g - 1, g - 1));
}
`;

/** 1. Count nodes per cell */
export const GRID_COUNT_SHADER = /* wgsl */ `
${FORCE_PARAMS}
${CELL_OF}
@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> sizes: array<f32>;
@group(0) @binding(3) var<storage, read_write> cellOf: array<u32>;
@group(0) @binding(4) var<storage, read_write> cellCount: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_grid_count(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  if (sizes[i] <= 0.0) {
    cellOf[i] = 0xFFFFFFFFu;
    return;
  }
  let c = cellCoords(positions[i]);
  let cell = u32(c.y) * params.gridDim + u32(c.x);
  cellOf[i] = cell;
  atomicAdd(&cellCount[cell], 1u);
}
`;

/** 2. Exclusive prefix sum of the counts (one workgroup) */
export const GRID_SCAN_SHADER = /* wgsl */ `
${FORCE_PARAMS}
@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read_write> cellCursor: array<atomic<u32>>;

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_grid_scan(@builtin(local_invocation_id) lid: vec3u) {
  let cells = params.gridDim * params.gridDim;
  let per = (cells + 255u) / 256u;
  let first = min(lid.x * per, cells);
  let last = min(first + per, cells);

  var sum = 0u;
  for (var k = first; k < last; k++) { sum += atomicLoad(&cellCount[k]); }
  partial[lid.x] = sum;
  workgroupBarrier();

  // Inclusive scan of the 256 partial sums (Hillis-Steele)
  for (var offset = 1u; offset < 256u; offset = offset * 2u) {
    var v = partial[lid.x];
    if (lid.x >= offset) { v += partial[lid.x - offset]; }
    workgroupBarrier();
    partial[lid.x] = v;
    workgroupBarrier();
  }

  var running = partial[lid.x] - sum;
  for (var k = first; k < last; k++) {
    cellStart[k] = running;
    atomicStore(&cellCursor[k], running);
    running += atomicLoad(&cellCount[k]);
  }
  if (lid.x == 255u) { cellStart[cells] = partial[255]; }
}
`;

/** 3. Place each node in its cell's slot range */
export const GRID_SCATTER_SHADER = /* wgsl */ `
${FORCE_PARAMS}
@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read> cellOf: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted: array<u32>;

@compute @workgroup_size(64)
fn cs_grid_scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  let cell = cellOf[i];
  if (cell == 0xFFFFFFFFu) { return; }
  sorted[atomicAdd(&cellCursor[cell], 1u)] = i;
}
`;

/** 4. Mass & center of mass per cell (and reset counts for the next step) */
export const GRID_MASS_SHADER = /* wgsl */ `
${FORCE_PARAMS}
@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> sorted: array<u32>;
@group(0) @binding(4) var<storage, read_write> cellMass: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> cellCount: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_grid_mass(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= params.gridDim * params.gridDim) { return; }
  let first = cellStart[c];
  let last = cellStart[c + 1u];
  var sum = vec2f(0.0, 0.0);
  for (var k = first; k < last; k++) { sum += positions[sorted[k]]; }
  let m = f32(last - first);
  cellMass[c] = vec4f(select(vec2f(0.0, 0.0), sum / m, m > 0.0), m, 0.0);
  atomicStore(&cellCount[c], 0u);
}
`;

/** 5. Forces: near cells exact, far cells by center of mass, + springs & gravity */
export const GRID_FORCES_SHADER = /* wgsl */ `
${FORCE_PARAMS}
${CELL_OF}
${PAIR_DIRECTION}
@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> sizes: array<f32>;
@group(0) @binding(3) var<storage, read> neighbors: array<u32>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> forces: array<vec2f>;
@group(0) @binding(6) var<storage, read> cellStart: array<u32>;
@group(0) @binding(7) var<storage, read> sorted: array<u32>;
@group(0) @binding(8) var<storage, read> cellMass: array<vec4f>;
${SPRINGS_AND_GRAVITY}

const NEAR: i32 = ${GRID_NEAR};

@compute @workgroup_size(64)
fn cs_grid_forces(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  if (sizes[i] <= 0.0) {
    forces[i] = vec2f(0.0, 0.0);
    return;
  }

  let pos_i = positions[i];
  let own = cellCoords(pos_i);
  let g = i32(params.gridDim);
  let G = params.gravitationalConstant;
  var force = vec2f(0.0, 0.0);

  for (var cy = 0; cy < g; cy++) {
    for (var cx = 0; cx < g; cx++) {
      let c = u32(cy * g + cx);
      if (abs(cx - own.x) <= NEAR && abs(cy - own.y) <= NEAR) {
        // Near: exact, node by node
        let end = cellStart[c + 1u];
        for (var k = cellStart[c]; k < end; k++) {
          let j = sorted[k];
          if (j == i) { continue; }
          var delta = pos_i - positions[j];
          var dist = length(delta);
          if (dist < 0.0001) {
            dist = 0.01;
            delta = pairDirection(i, j) * dist;
          }
          force -= (delta / dist) * (G / (dist * dist));
        }
      } else {
        // Far: the whole cell as one mass at its center of mass
        let cm = cellMass[c];
        if (cm.z == 0.0) { continue; }
        let delta = pos_i - cm.xy;
        let dist = max(length(delta), 0.0001);
        force -= (delta / dist) * (G * cm.z / (dist * dist));
      }
    }
  }

  forces[i] = force + springsAndGravity(i, pos_i);
}
`;
