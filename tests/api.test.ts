// @vitest-environment happy-dom
// ============================================================
// graphGPU - Public API tests (no GPU needed)
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GraphGPU } from '../src/index';

let frames = new Map<number, FrameRequestCallback>();
let nextFrameId = 1;

function flushFrames(count: number): void {
    for (let i = 0; i < count && frames.size > 0; i++) {
        const pending = [...frames.values()];
        frames = new Map();
        for (const cb of pending) cb(0);
    }
}

let canvas: HTMLCanvasElement;
let g: GraphGPU;

beforeEach(() => {
    frames = new Map();
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
        const id = nextFrameId++;
        frames.set(id, cb);
        return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });

    canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    g = new GraphGPU({ canvas });
    for (let i = 0; i < 20; i++) g.put('n', { name: `n${i}` });
    for (let i = 1; i < 20; i++) g.link(i - 1, 'e', i);
});

afterEach(() => {
    canvas.remove();
    vi.unstubAllGlobals();
});

describe('GraphGPU: layout lifecycle', () => {
    it('replaces the running layout instead of stacking loops', () => {
        g.startLayout({ minVelocity: 0 });
        g.startLayout({ minVelocity: 0 });
        g.startLayout({ minVelocity: 0 });
        flushFrames(3);
        expect(frames.size).toBe(1);
    });

    it('stops the layout', () => {
        g.startLayout({ minVelocity: 0 });
        flushFrames(2);
        g.stopLayout();
        expect(frames.size).toBe(0);
    });

    it('keeps animated mode when restarting the layout', () => {
        g.setAnimated(true);
        g.startLayout({ maxIterations: 3 });
        flushFrames(20);
        // Animated layouts never auto-stop
        expect(frames.size).toBe(1);
        g.stopLayout();
        expect(frames.size).toBe(0);
    });

    it('moves nodes while running', () => {
        const before = g.positions.slice(0, 40);
        g.startLayout();
        flushFrames(3);
        expect(Array.from(g.positions.subarray(0, 40))).not.toEqual(Array.from(before));
        g.stopLayout();
    });
});
