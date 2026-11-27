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

describe('GraphGPU: palette', () => {
    it('keeps explicit node colors when switching palettes', () => {
        const red = g.put('custom', 'Red', {}, { color: '#ff0000' });
        g.setPalette('vibrant');
        expect(Array.from(g.colors.subarray(red * 4, red * 4 + 3))).toEqual([1, 0, 0]);
    });
});

describe('GraphGPU: layout events', () => {
    function record() {
        const log: string[] = [];
        const ticks: { iteration: number; energy?: number }[] = [];
        g.on('layout:start', () => log.push('start'));
        g.on<{ iteration: number; energy?: number }>('layout:tick', (e) => { log.push('tick'); ticks.push(e.target); });
        g.on('layout:stop', () => log.push('stop'));
        return { log, ticks };
    }

    it('emits start, ticks and stop', () => {
        const { log, ticks } = record();
        g.startLayout({ minVelocity: 0 });
        flushFrames(3);
        g.stopLayout();
        expect(log).toEqual(['start', 'tick', 'tick', 'tick', 'stop']);
        expect(ticks.map(t => t.iteration)).toEqual([3, 6, 9]);
        expect(ticks.every(t => typeof t.energy === 'number')).toBe(true);
    });

    it('emits stop when the layout settles by itself', () => {
        const { log } = record();
        g.startLayout({ maxIterations: 6, minVelocity: 0 });
        flushFrames(10);
        expect(log.at(-1)).toBe('stop');
        expect(log.filter(e => e === 'stop')).toHaveLength(1);
    });

    it('emits start for animated mode, and stop + start when restarted', () => {
        const { log } = record();
        g.setAnimated(true);
        expect(log).toEqual(['start']);
        g.startLayout();
        expect(log).toEqual(['start', 'stop', 'start']);
        g.stopLayout();
    });
});

describe('GraphGPU: static layouts', () => {
    it('applies a layout and stops the running one', () => {
        g.startLayout({ minVelocity: 0 });
        g.applyLayout('grid', { spacing: 0.5 });
        expect(frames.size).toBe(0);
        // 20 nodes: 5 columns
        expect(g.positions[0]).toBeCloseTo(-1);
        expect(g.positions[2]).toBeCloseTo(-0.5);
        g.applyLayout('circular');
        const r = Math.hypot(g.positions[0], g.positions[1]);
        expect(Math.hypot(g.positions[2], g.positions[3])).toBeCloseTo(r);
    });
});
