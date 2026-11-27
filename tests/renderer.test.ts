// @vitest-environment happy-dom
// ============================================================
// graphGPU - Renderer tests (fake GPU device)
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GraphGPU } from '../src/index';
import { gpu, installFakeWebGPU, installFakeFrames, flushFrames, pendingFrames } from './fake-gpu';

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
