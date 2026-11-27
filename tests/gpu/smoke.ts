// ============================================================
// graphGPU - GPU smoke test (real WebGPU device)
// ============================================================
// Renders with the real Renderer on a real GPU (Deno's WebGPU)
// and checks actual pixels, plus that no validation errors
// occur. Catches shader/pipeline mistakes the fake device can't.
//
// Usage:
//   npm run test:gpu
// ============================================================

import { Graph } from '../../src/core/Graph';
import { Renderer } from '../../src/core/Renderer';
import { GPUForceLayout } from '../../src/layout/GPUForceLayout';

// Deno has no animation frames
const g = globalThis as Record<string, unknown>;
g.requestAnimationFrame ??= (cb: (t: number) => void) => setTimeout(() => cb(performance.now()), 4);
g.cancelAnimationFrame ??= (id: number) => clearTimeout(id);

const SIZE = 256;
const BG: [number, number, number, number] = [0, 0, 0, 1];

let failures = 0;
function check(name: string, ok: boolean, detail = ''): void {
    console.log(`${ok ? '✓' : '✗'} ${name}${detail ? `  (${detail})` : ''}`);
    if (!ok) failures++;
}

// -----------------------------------------------------------
// A headless "canvas" backed by a real texture
// -----------------------------------------------------------

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
    // In CI the adapter is required (GPU_REQUIRED=1): fail instead of skipping
    console.log('No WebGPU adapter available' + (Deno.env.get('GPU_REQUIRED') ? '' : ', skipping'));
    Deno.exit(Deno.env.get('GPU_REQUIRED') ? 1 : 0);
}
const info = adapter.info ?? {};
console.log(`Adapter: ${[info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / ') || 'unknown'}\n`);

let device!: GPUDevice;
let target!: GPUTexture;
const format = navigator.gpu.getPreferredCanvasFormat();

const context = {
    configure(cfg: GPUCanvasConfiguration) {
        device = cfg.device;

        // Deno only accepts views here (browsers & the spec take raw
        // ArrayBuffers too): wrap them, offsets/sizes are bytes either way
        const queue = device.queue;
        const writeBuffer = queue.writeBuffer.bind(queue);
        queue.writeBuffer = ((buf: GPUBuffer, offset: number, data: BufferSource, dataOffset = 0, size?: number) =>
            data instanceof ArrayBuffer
                ? writeBuffer(buf, offset, new Uint8Array(data), dataOffset, size)
                : writeBuffer(buf, offset, data as ArrayBufferView, dataOffset, size)
        ) as typeof queue.writeBuffer;
        device.addEventListener('uncapturederror', (e) => {
            check('no uncaptured GPU errors', false, (e as GPUUncapturedErrorEvent).error.message);
        });
        target = device.createTexture({
            size: [SIZE, SIZE],
            format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
    },
    getCurrentTexture: () => target,
};

const canvas = {
    width: SIZE,
    height: SIZE,
    clientWidth: SIZE,
    clientHeight: SIZE,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: SIZE, height: SIZE }),
    getContext: (type: string) => (type === 'webgpu' ? context : null),
} as unknown as HTMLCanvasElement;

async function readPixel(x: number, y: number): Promise<[number, number, number]> {
    const bytesPerRow = 256;
    const buffer = device.createBuffer({ size: bytesPerRow, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture: target, origin: [x, y] },
        { buffer, bytesPerRow },
        [1, 1],
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(buffer.getMappedRange().slice(0, 4));
    buffer.unmap();
    buffer.destroy();
    // Normalize BGRA → RGB
    return format.startsWith('bgra') ? [px[2], px[1], px[0]] : [px[0], px[1], px[2]];
}

/** World coords → pixel (aspect 1, zoom 1: world [-1, 1] spans the canvas) */
const toPixel = (wx: number, wy: number): [number, number] => [
    Math.round((wx + 1) / 2 * SIZE),
    Math.round((1 - wy) / 2 * SIZE),
];

/** Brightest of 3 vertically adjacent pixels (lines are thin & antialiased) */
async function sample(wx: number, wy: number): Promise<[number, number, number]> {
    const [x, y] = toPixel(wx, wy);
    let best: [number, number, number] = [0, 0, 0];
    for (const dy of [-1, 0, 1]) {
        const c = await readPixel(x, Math.min(SIZE - 1, Math.max(0, y + dy)));
        if (c[0] + c[1] + c[2] > best[0] + best[1] + best[2]) best = c;
    }
    return best;
}

const isBackground = (c: number[]) => c[0] < 10 && c[1] < 10 && c[2] < 10;
const fmt = (c: number[]) => `rgb(${c.join(',')})`;

// -----------------------------------------------------------
// Scene: two nodes and an edge between them
// -----------------------------------------------------------

const graph = new Graph();
const a = graph.addNode({ tag: 'n', x: -0.6, y: 0, color: '#ff0000' });
const b = graph.addNode({ tag: 'n', x: 0.6, y: 0, color: '#0000ff' });
const e = graph.addEdge({ tag: 'link', source: a, target: b });
graph.edgeColors.set([0, 1, 0], e * 3);

const renderer = new Renderer(graph, {
    canvas,
    pixelRatio: 1,
    nodeSize: 1,
    edgeOpacity: 1,
    background: BG,
});

check('renderer initializes', await renderer.init());

device.pushErrorScope('validation');
renderer.renderFrame(16);
await device.queue.onSubmittedWorkDone();
const initErr = await device.popErrorScope();
check('first frame has no validation errors', initErr === null, initErr?.message);

// Edge midpoint is on the edge; node centers are the nodes
const mid = await sample(0, 0);
check('edge is drawn between its nodes', mid[1] > 150 && mid[0] < 30, fmt(mid));
const nodeA = await sample(-0.6, 0);
check('node is drawn with its color', nodeA[0] > 150 && nodeA[2] < 100, fmt(nodeA));
const empty = await sample(0, 0.8);
check('background stays empty', isBackground(empty), fmt(empty));

// -----------------------------------------------------------
// Moving a node: the edge must follow it (positions on the GPU)
// -----------------------------------------------------------

graph.setNodePosition(a, -0.6, 0.6);
graph.setNodePosition(b, 0.6, 0.6);
renderer.renderFrame(32);

const oldMid = await sample(0, 0);
check('edge left its old position', isBackground(oldMid), fmt(oldMid));
const newMid = await sample(0, 0.6);
check('edge follows moved nodes', newMid[1] > 150, fmt(newMid));
// Same thin line, different sub-pixel position: same brightness
check('thin edges keep their brightness when moving', Math.abs(newMid[1] - mid[1]) < 40, `${mid[1]} vs ${newMid[1]}`);

// -----------------------------------------------------------
// Removing the edge hides it; adding many nodes/edges works
// -----------------------------------------------------------

// A fixed edge width: the line gets about that many pixels thick
renderer.setEdgeWidth(9);
renderer.renderFrame(40);
{
    const [x, y] = toPixel(0, 0.6);
    let thick = 0;
    for (let dy = -10; dy <= 10; dy++) if ((await readPixel(x, y + dy))[1] > 128) thick++;
    check('edgeWidth sets the drawn width', thick >= 8 && thick <= 10, `${thick}px`);
}
renderer.setEdgeWidth(0);

// A selected edge is drawn wider than a normal one
{
    const [x, y] = toPixel(0, 0.6);
    const thickness = async () => {
        let t = 0;
        for (let dy = -10; dy <= 10; dy++) if ((await readPixel(x, y + dy))[1] > 128) t++;
        return t;
    };
    renderer.renderFrame(42);
    const normal = await thickness();
    renderer.setEdgeSelection(e, true);
    renderer.renderFrame(44);
    const selected = await thickness();
    renderer.setEdgeSelection(e, false);
    check('selected edges are drawn wider', selected >= normal * 2, `${normal}px → ${selected}px`);
}

graph.removeEdge(e);
renderer.renderFrame(48);
const gone = await sample(0, 0.6);
check('removed edge is not drawn', isBackground(gone), fmt(gone));

device.pushErrorScope('validation');
for (let i = 0; i < 5000; i++) graph.addNode({ tag: 't', x: Math.random() * 2 - 1, y: Math.random() * 2 - 1 });
for (let i = 0; i < 10000; i++) {
    graph.addEdge({ tag: 'r', source: Math.floor(Math.random() * 5000), target: Math.floor(Math.random() * 5000) });
}
renderer.renderFrame(64);
await device.queue.onSubmittedWorkDone();
const growErr = await device.popErrorScope();
check('growing buffers has no validation errors', growErr === null, growErr?.message);

// -----------------------------------------------------------
// GPU layout writing straight into the renderer's buffer
// -----------------------------------------------------------

const before = graph.positions.slice(0, graph.nodeSlots * 2);
device.pushErrorScope('validation');
const layout = new GPUForceLayout(device, graph, { stepsPerFrame: 2, maxIterations: 100000 }, renderer);
await layout.start();
for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 8));
    if (renderer.isDirty()) renderer.renderFrame(100 + i * 16);
}
// Nodes added while running end up in the shared buffer too
for (let i = 0; i < 300; i++) graph.addNode({ tag: 'late', x: 0, y: 0 });
for (let i = 0; i < 10; i++) {
    await new Promise(r => setTimeout(r, 8));
    if (renderer.isDirty()) renderer.renderFrame(600 + i * 16);
}
layout.stop();
await new Promise(r => setTimeout(r, 100));
await device.queue.onSubmittedWorkDone();
const layoutErr = await device.popErrorScope();
check('GPU layout + render has no validation errors', layoutErr === null, layoutErr?.message);

let moved = 0;
for (let i = 0; i < before.length; i++) if (Math.abs(graph.positions[i] - before[i]) > 1e-4) moved++;
check('GPU layout moved the nodes (CPU copy updated)', moved > before.length * 0.5, `${moved}/${before.length} coords`);

let finite = true;
for (const v of graph.positions.subarray(0, graph.nodeSlots * 2)) if (!Number.isFinite(v)) finite = false;
check('positions stay finite', finite);

// After stop, the renderer draws the CPU positions again: they must match the GPU's
device.pushErrorScope('validation');
graph.setNodePosition(0, 0, 0);
renderer.renderFrame(2000);
await device.queue.onSubmittedWorkDone();
const afterErr = await device.popErrorScope();
check('rendering after the GPU layout has no validation errors', afterErr === null, afterErr?.message);

layout.destroy();

// -----------------------------------------------------------
// Physics: one GPU step vs a plain CPU implementation
// -----------------------------------------------------------
// With velocity starting at 0 and no clamping, one step moves each
// node by exactly F·dt², so the GPU's force per node can be read off
// the position change and compared to the reference.

{
    const N = 400;
    const P = { G: -0.25, L: 0.2, K: 0.06, C: 0.012, dt: 0.1 };
    const pg = new Graph({ maxNodes: N, maxEdges: 1000 });
    let seed = 7;
    const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    for (let i = 0; i < N; i++) pg.addNode({ tag: 'n', x: rand() * 2 - 1, y: rand() * 2 - 1 });
    pg.setNodePosition(N - 1, pg.positions[(N - 2) * 2], pg.positions[(N - 2) * 2 + 1]); // coincident pair
    for (let i = 1; i <= 150; i++) pg.addEdge({ tag: 'hub', source: 0, target: i }); // hub
    for (let i = 0; i < 300; i++) {
        pg.addEdge({ tag: 'r', source: Math.floor(rand() * N), target: Math.floor(rand() * N) });
    }
    pg.addEdge({ tag: 'loop', source: 5, target: 5 });                  // self-loop (ignored)
    const removed = pg.addEdge({ tag: 'gone', source: 3, target: 4 });
    pg.removeEdge(removed);

    const x0 = Array.from(pg.positions.subarray(0, N * 2));

    // CPU reference (same formulas as the shader)
    const ref = new Float64Array(N * 2);
    for (let i = 0; i < N; i++) {
        let fx = 0, fy = 0;
        for (let j = 0; j < N; j++) {
            if (j === i) continue;
            const dx = x0[i * 2] - x0[j * 2], dy = x0[i * 2 + 1] - x0[j * 2 + 1];
            const d = Math.hypot(dx, dy);
            if (d < 0.0001) continue; // coincident: direction is arbitrary, checked separately
            const mag = P.G / (d * d);
            fx -= (dx / d) * mag;
            fy -= (dy / d) * mag;
        }
        const dc = Math.max(Math.hypot(x0[i * 2], x0[i * 2 + 1]), 0.0001);
        fx += -x0[i * 2] * (P.C / dc);
        fy += -x0[i * 2 + 1] * (P.C / dc);
        ref[i * 2] = fx;
        ref[i * 2 + 1] = fy;
    }
    for (const id of pg.activeEdgeIds()) {
        const a = pg.edgeIndices[id * 2], b = pg.edgeIndices[id * 2 + 1];
        if (a === b) continue;
        const dx = x0[a * 2] - x0[b * 2], dy = x0[a * 2 + 1] - x0[b * 2 + 1];
        const d = Math.max(Math.hypot(dx, dy), 0.0001);
        const sf = P.K * (P.L - d) / d;
        ref[a * 2] += dx * sf; ref[a * 2 + 1] += dy * sf;
        ref[b * 2] -= dx * sf; ref[b * 2 + 1] -= dy * sf;
    }

    device.pushErrorScope('validation');
    const physics = new GPUForceLayout(device, pg, {
        gravitationalConstant: P.G, springLength: P.L, springConstant: P.K,
        centralGravity: P.C, damping: 0, timestep: P.dt, maxVelocity: 1e9,
    });
    await physics.step(1);
    const physicsErr = await device.popErrorScope();
    check('GPU physics step has no validation errors', physicsErr === null, physicsErr?.message);

    const gpuForce = (i: number, k: number) => (pg.positions[i * 2 + k] - x0[i * 2 + k]) / (P.dt * P.dt);
    let worst = 0, worstId = -1;
    for (let i = 0; i < N; i++) {
        if (i === N - 2 || i === N - 1) continue; // coincident pair: has an extra push
        const err = Math.hypot(gpuForce(i, 0) - ref[i * 2], gpuForce(i, 1) - ref[i * 2 + 1]);
        const rel = err / Math.max(1e-3, Math.hypot(ref[i * 2], ref[i * 2 + 1]));
        if (rel > worst) { worst = rel; worstId = i; }
    }
    check('GPU forces match the CPU reference', worst < 0.02, `worst relative error ${(worst * 100).toFixed(2)}% at node ${worstId}`);

    // The hub's spring forces must all arrive (no lost updates)
    const hubSprings = (() => {
        let fx = 0, fy = 0;
        for (let i = 1; i <= 150; i++) {
            const dx = x0[0] - x0[i * 2], dy = x0[1] - x0[i * 2 + 1];
            const d = Math.max(Math.hypot(dx, dy), 0.0001);
            const sf = P.K * (P.L - d) / d;
            fx += dx * sf; fy += dy * sf;
        }
        return Math.hypot(fx, fy);
    })();
    const hubErr = Math.hypot(gpuForce(0, 0) - ref[0], gpuForce(0, 1) - ref[1]) / Math.hypot(ref[0], ref[1]);
    check('the hub gets all 150 spring forces', hubErr < 0.02, `error ${(hubErr * 100).toFixed(2)}%, spring total ${hubSprings.toFixed(3)}`);

    const pair = Math.hypot(
        pg.positions[(N - 2) * 2] - pg.positions[(N - 1) * 2],
        pg.positions[(N - 2) * 2 + 1] - pg.positions[(N - 1) * 2 + 1],
    );
    check('coincident nodes are pushed apart', pair > 0 && Number.isFinite(pair), `distance ${pair.toExponential(2)}`);
    physics.destroy();
}

// -----------------------------------------------------------
// Grid-approximated repulsion (large graphs) vs exact reference
// -----------------------------------------------------------

{
    const N = 20000;
    const P = { G: -0.25, L: 0.2, K: 0.06, C: 0.012, dt: 0.1 };
    const pg = new Graph({ maxNodes: N, maxEdges: N });
    let seed = 99;
    const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    for (let i = 0; i < N; i++) pg.addNode({ tag: 'n', x: rand() * 2 - 1, y: rand() * 2 - 1 });
    for (let i = 0; i < N; i++) {
        pg.addEdge({ tag: 'r', source: Math.floor(rand() * N), target: Math.floor(rand() * N) });
    }
    const x0 = Float64Array.from(pg.positions.subarray(0, N * 2));

    // Exact reference: O(n²) repulsion + springs + gravity
    const ref = new Float64Array(N * 2);
    for (let i = 0; i < N; i++) {
        const xi = x0[i * 2], yi = x0[i * 2 + 1];
        let fx = 0, fy = 0;
        for (let j = 0; j < N; j++) {
            if (j === i) continue;
            const dx = xi - x0[j * 2], dy = yi - x0[j * 2 + 1];
            const d2 = dx * dx + dy * dy;
            if (d2 < 1e-8) continue;
            const d = Math.sqrt(d2);
            const mag = P.G / d2;
            fx -= (dx / d) * mag;
            fy -= (dy / d) * mag;
        }
        const dc = Math.max(Math.hypot(xi, yi), 0.0001);
        ref[i * 2] = fx - xi * (P.C / dc);
        ref[i * 2 + 1] = fy - yi * (P.C / dc);
    }
    for (const id of pg.activeEdgeIds()) {
        const a = pg.edgeIndices[id * 2], b = pg.edgeIndices[id * 2 + 1];
        if (a === b) continue;
        const dx = x0[a * 2] - x0[b * 2], dy = x0[a * 2 + 1] - x0[b * 2 + 1];
        const d = Math.max(Math.hypot(dx, dy), 0.0001);
        const sf = P.K * (P.L - d) / d;
        ref[a * 2] += dx * sf; ref[a * 2 + 1] += dy * sf;
        ref[b * 2] -= dx * sf; ref[b * 2 + 1] -= dy * sf;
    }

    device.pushErrorScope('validation');
    const approx = new GPUForceLayout(device, pg, {
        gravitationalConstant: P.G, springLength: P.L, springConstant: P.K,
        centralGravity: P.C, damping: 0, timestep: P.dt, maxVelocity: 1e9,
        approximateRepulsion: true,
    });
    await approx.step(1);
    const gridErr = await device.popErrorScope();
    check('grid repulsion has no validation errors', gridErr === null, gridErr?.message);

    // Error relative to the typical force (plain relative error blows up
    // where forces nearly cancel)
    let meanForce = 0;
    for (let i = 0; i < N; i++) meanForce += Math.hypot(ref[i * 2], ref[i * 2 + 1]);
    meanForce /= N;
    const errors: number[] = [];
    for (let i = 0; i < N; i++) {
        const gx = (pg.positions[i * 2] - x0[i * 2]) / (P.dt * P.dt);
        const gy = (pg.positions[i * 2 + 1] - x0[i * 2 + 1]) / (P.dt * P.dt);
        errors.push(Math.hypot(gx - ref[i * 2], gy - ref[i * 2 + 1]) / meanForce);
    }
    errors.sort((a, b) => a - b);
    const median = errors[Math.floor(N * 0.5)], p99 = errors[Math.floor(N * 0.99)];
    console.log(`  grid error vs exact: median ${(median * 100).toFixed(2)}%, p99 ${(p99 * 100).toFixed(2)}%, max ${(errors[N - 1] * 100).toFixed(2)}% (of the mean force)`);
    check('grid repulsion stays close to exact', median < 0.005 && p99 < 0.01, `median ${(median * 100).toFixed(2)}%, p99 ${(p99 * 100).toFixed(2)}%`);
    approx.destroy();
}

// Destroys the device too, so it goes last
renderer.destroy();

console.log(failures === 0 ? '\nGPU smoke test passed' : `\n${failures} check(s) failed`);
Deno.exit(failures === 0 ? 0 : 1);
