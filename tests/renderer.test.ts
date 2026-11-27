// @vitest-environment happy-dom
// ============================================================
// graphGPU - Renderer tests (fake GPU device)
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GraphGPU } from '../src/index';
import { gpu, installFakeWebGPU, installFakeFrames, flushFrames, pendingFrames, FakeBuffer } from './fake-gpu';

let canvas: HTMLCanvasElement;
let g: GraphGPU;

beforeEach(async () => {
    canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
        left: 0, top: 0, right: 800, bottom: 600,
        width: 800, height: 600, x: 0, y: 0,
        toJSON: () => ({}),
    } as DOMRect);

    installFakeWebGPU(canvas);
    installFakeFrames();

    g = new GraphGPU({ canvas, pixelRatio: 1 });
    for (let i = 0; i < 10; i++) g.put('n', { name: `n${i}` });
    for (let i = 1; i < 10; i++) g.link(i - 1, 'e', i);
    expect(await g.init()).toBe(true);
});

afterEach(() => {
    g.destroy();
    canvas.remove();
    vi.unstubAllGlobals();
});

/** Render passes submitted while running `frames` frames */
async function framesDrawn(frames: number, action?: () => void): Promise<number> {
    action?.();
    const before = gpu.renderPasses;
    await flushFrames(frames);
    return gpu.renderPasses - before;
}

describe('Renderer: render on demand', () => {
    it('draws the first frame, then idles', async () => {
        expect(await framesDrawn(1)).toBe(1);
        expect(await framesDrawn(30)).toBe(0);
        // ...while still waiting for changes
        expect(pendingFrames()).toBe(1);
    });

    it('draws nodes and edges', async () => {
        await flushFrames(1);
        // 6 vertices per instanced quad: 9 edges, 10 nodes
        expect(gpu.draws).toEqual([[6, 9], [6, 10]]);
    });

    it('redraws once after a camera change', async () => {
        await flushFrames(1);
        expect(await framesDrawn(10, () => g.setView(0.5, 0.5, 2))).toBe(1);
    });

    it('redraws once after a graph change', async () => {
        await flushFrames(1);
        expect(await framesDrawn(10, () => g.put('n', { name: 'new' }))).toBe(1);
        expect(await framesDrawn(10, () => g.unput(0))).toBe(1);
    });

    it('redraws once after an appearance change', async () => {
        await flushFrames(1);
        const changes: (() => void)[] = [
            () => g.setBackground([1, 1, 1, 1]),
            () => g.setNodeSize(3),
            () => g.setEdgeOpacity(0.5),
            () => g.setLabelsVisible(false),
            () => g.setPalette('vibrant'),
            () => g.resize(),
            () => g.requestRender(),
            () => g.markDirty(),
        ];
        for (const change of changes) {
            expect(await framesDrawn(5, change)).toBe(1);
        }
    });

    it('redraws once after a selection change', async () => {
        await flushFrames(1);
        const renderer = (g as unknown as { renderer: { setSelection(id: number, on: boolean): void } }).renderer;
        expect(await framesDrawn(5, () => renderer.setSelection(2, true))).toBe(1);
    });

    it('draws every frame while the layout runs, and idles after', async () => {
        await flushFrames(1);
        g.startLayout({ minVelocity: 0 });
        // The layout ticks in its own frame callback, after the renderer's,
        // so its first tick shows up one frame later
        expect(await framesDrawn(10)).toBeGreaterThanOrEqual(9);
        g.stopLayout();
        await flushFrames(1);
        expect(await framesDrawn(10)).toBe(0);
    });

    it('does not start a second loop on resume()', async () => {
        g.resume();
        g.resume();
        expect(pendingFrames()).toBe(1);
        g.pause();
        expect(pendingFrames()).toBe(0);
        g.resume();
        expect(await framesDrawn(1)).toBe(1);
    });
});

describe('Renderer: edges', () => {
    // Edge instances are 12 bytes: src, tgt, packed color
    const edgeWrites = () => gpu.writes.filter(w => w.bytes > 0 && w.bytes % 12 === 0 && w.bytes / 12 === g.getGraph().edgeSlots);
    const edgeBuffer = () => edgeWrites().at(-1)?.buffer as FakeBuffer;

    it('uploads 12 bytes per edge: endpoints and packed color', async () => {
        await flushFrames(1);
        const data = new Uint32Array(edgeBuffer().data, 0, 9 * 3);
        const edge = g.getGraph().getEdge(3)!;
        expect(data[3 * 3]).toBe(edge.source);
        expect(data[3 * 3 + 1]).toBe(edge.target);
        expect(data[3 * 3 + 2] >>> 24).toBe(255); // visible
    });

    it('does not re-upload edges when nodes move', async () => {
        await flushFrames(1);
        const before = edgeWrites().length;
        g.getGraph().setNodePosition(0, 0.3, 0.3);
        expect(await framesDrawn(1)).toBe(1);
        g.startLayout({ minVelocity: 0 });
        await flushFrames(5);
        g.stopLayout();
        expect(edgeWrites().length).toBe(before);
    });

    it('re-uploads edges when they change', async () => {
        await flushFrames(1);
        const count = () => gpu.writes.filter(w => w.buffer === edgeBuffer()).length;
        let before = count();
        g.link(0, 'e', 5);
        await flushFrames(1);
        expect(count()).toBe(before + 1);

        before = count();
        g.setPalette('vibrant');
        await flushFrames(1);
        expect(count()).toBe(before + 1);
    });

    it('hides removed edges instead of compacting', async () => {
        await flushFrames(1);
        const ids = g.edges(4).map(e => e.id);
        g.unlink(ids[0]);
        await flushFrames(1);
        const data = new Uint32Array(edgeBuffer().data, 0, 9 * 3);
        expect(data[ids[0] * 3 + 2] >>> 24).toBe(0); // alpha 0 → culled in shader
        // Still one instance per slot
        expect(gpu.draws.at(-2)).toEqual([6, 9]);
    });
});

describe('Renderer: GPU layout with shared positions', () => {
    interface Internals {
        renderer: { gpuPositions: FakeBuffer; externalPositions: boolean };
    }
    const internals = () => (g as unknown as Internals).renderer;
    const positionWrites = () => gpu.writes.filter(w => w.buffer === internals().gpuPositions);
    const gpuPositions = () => new Float32Array(internals().gpuPositions.data);

    it('never uploads CPU positions while the GPU layout runs', async () => {
        await flushFrames(1);
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 1000 });
        await flushFrames(1);
        const before = positionWrites().length;

        expect(await framesDrawn(10)).toBe(10);
        expect(positionWrites().length).toBe(before);
        expect(gpu.errors).toEqual([]);
        g.stopGPULayout();
    });

    it('keeps the CPU copy of positions up to date in the background', async () => {
        await flushFrames(1);
        const x0 = g.positions[0];
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 1000 });
        await flushFrames(5);
        // The fake compute moves every node by +1 per step
        expect(g.positions[0]).toBeGreaterThan(x0 + 2);
        g.stopGPULayout();
    });

    it('sends dragged nodes to the GPU as 8-byte writes', async () => {
        await flushFrames(1);
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 1000 });
        await flushFrames(1);

        const before = positionWrites().length;
        g.getGraph().setNodePosition(3, 42, -42);
        await flushFrames(1);
        const writes = positionWrites().slice(before);
        expect(writes).toEqual([{ buffer: internals().gpuPositions, bytes: 8 }]);
        g.stopGPULayout();
    });

    it('syncs the CPU copy and hands positions back on stop', async () => {
        await flushFrames(1);
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 1000 });
        await flushFrames(3);
        g.stopLayout();
        await flushFrames(2);

        expect(internals().externalPositions).toBe(false);
        expect(Array.from(g.positions.subarray(0, 20))).toEqual(Array.from(gpuPositions().subarray(0, 20)));
    });

    it('stops by itself at maxIterations and hands positions back', async () => {
        await flushFrames(1);
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 3 });
        await flushFrames(6);
        expect(internals().externalPositions).toBe(false);
    });

    it('carries GPU positions over when the buffer grows', async () => {
        await flushFrames(1);
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 1000 });
        await flushFrames(2);
        const node0 = gpuPositions()[0];

        for (let i = 0; i < 600; i++) g.put('late', { name: `l${i}` });
        await flushFrames(1);
        // Same node, carried over to the new buffer, then stepped on
        expect(gpuPositions()[0]).toBeGreaterThanOrEqual(node0);
        expect(gpuPositions().length).toBeGreaterThanOrEqual(610 * 2);
        expect(gpu.errors).toEqual([]);
        g.stopGPULayout();
    });
});

describe('Renderer: node uploads', () => {
    interface Internals {
        renderer: { gpuPositions: FakeBuffer; gpuColors: FakeBuffer; gpuSizes: FakeBuffer; gpuSelection: FakeBuffer };
    }
    const bufs = () => (g as unknown as Internals).renderer;
    const writesTo = (b: FakeBuffer) => gpu.writes.filter(w => w.buffer === b).length;
    const snapshot = () => {
        const r = bufs();
        return [writesTo(r.gpuPositions), writesTo(r.gpuColors), writesTo(r.gpuSizes), writesTo(r.gpuSelection)];
    };
    const diff = (a: number[], b: number[]) => b.map((v, i) => v - a[i]);

    it('only uploads positions when nodes move', async () => {
        await flushFrames(1);
        const before = snapshot();
        g.getGraph().setNodePosition(1, 0.5, 0.5);
        await flushFrames(1);
        // positions, colors, sizes, selection
        expect(diff(before, snapshot())).toEqual([1, 0, 0, 0]);

        const mid = snapshot();
        g.startLayout({ minVelocity: 0 });
        await flushFrames(4);
        g.stopLayout();
        const after = diff(mid, snapshot());
        expect(after[0]).toBeGreaterThan(0);
        expect(after.slice(1)).toEqual([0, 0, 0]);
    });

    it('uploads colors and sizes when they change', async () => {
        await flushFrames(1);
        let before = snapshot();
        g.getGraph().setNodeColor(2, [1, 0, 0, 1]);
        await flushFrames(1);
        expect(diff(before, snapshot()).slice(1, 3)).toEqual([1, 1]);

        before = snapshot();
        g.put('n', { name: 'new' });
        await flushFrames(1);
        expect(diff(before, snapshot()).slice(0, 3)).toEqual([1, 1, 1]);
    });

    it('only uploads selection when it changes', async () => {
        await flushFrames(1);
        const before = snapshot();
        (g as unknown as { renderer: { setSelection(id: number, on: boolean): void } }).renderer.setSelection(1, true);
        await flushFrames(1);
        expect(diff(before, snapshot())).toEqual([0, 0, 0, 1]);
    });

    it('uploads everything after markDirty()', async () => {
        await flushFrames(1);
        const before = snapshot();
        g.markDirty();
        await flushFrames(1);
        expect(diff(before, snapshot())).toEqual([1, 1, 1, 1]);
    });
});

describe('Renderer: events', () => {
    it('emits render:frame with stats for every drawn frame', async () => {
        const frames: { fps: number; frameTime: number; nodeCount: number; edgeCount: number }[] = [];
        g.on<typeof frames[number]>('render:frame', (e) => frames.push(e.target));
        await flushFrames(3); // only the first one draws
        expect(frames).toHaveLength(1);
        expect(frames[0].nodeCount).toBe(10);
        expect(frames[0].edgeCount).toBe(9);
        expect(frames[0].frameTime).toBeGreaterThanOrEqual(0);
    });

    it('emits layout events for the GPU layout too', async () => {
        const log: string[] = [];
        g.on('layout:start', (e) => log.push(`start:${(e.target as { gpu: boolean }).gpu}`));
        g.on('layout:tick', () => log.push('tick'));
        g.on('layout:stop', () => log.push('stop'));
        await flushFrames(1);
        await g.startGPULayout({ stepsPerFrame: 1, maxIterations: 1000 });
        await flushFrames(2);
        g.stopLayout();
        expect(log[0]).toBe('start:true');
        expect(log.filter(e => e === 'tick').length).toBeGreaterThanOrEqual(2);
        expect(log.at(-1)).toBe('stop');
    });
});
