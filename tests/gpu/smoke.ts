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

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    console.log('No WebGPU adapter available, skipping');
    Deno.exit(0);
}

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
// Thin edges are dimmer when they straddle pixel rows (existing AA
// behavior), so only require clearly-not-background here
check('edge is drawn between its nodes', mid[1] > 20 && mid[0] < 20, fmt(mid));
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
check('edge follows moved nodes', newMid[1] > 100, fmt(newMid));

// -----------------------------------------------------------
// Removing the edge hides it; adding many nodes/edges works
// -----------------------------------------------------------

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
renderer.destroy();

console.log(failures === 0 ? '\nGPU smoke test passed' : `\n${failures} check(s) failed`);
Deno.exit(failures === 0 ? 0 : 1);
