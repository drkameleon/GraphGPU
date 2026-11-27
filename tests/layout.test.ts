// ============================================================
// graphGPU - ForceLayout tests
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Graph } from '../src/core/Graph';
import { ForceLayout } from '../src/layout/ForceLayout';
import { randomGraph, nodeDistance } from './helpers';

// -----------------------------------------------------------
// Manual requestAnimationFrame, so loops can be stepped
// -----------------------------------------------------------

let frames: FrameRequestCallback[] = [];

function flushFrames(count: number): void {
    for (let i = 0; i < count && frames.length > 0; i++) {
        const pending = frames;
        frames = [];
        for (const cb of pending) cb(0);
    }
}

beforeEach(() => {
    frames = [];
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
        frames.push(cb);
        return frames.length;
    });
    vi.stubGlobal('cancelAnimationFrame', () => {});
});

afterEach(() => {
    vi.unstubAllGlobals();
});

// -----------------------------------------------------------
// Physics
// -----------------------------------------------------------

describe('ForceLayout: physics', () => {
    it('does nothing on an empty graph', () => {
        const layout = new ForceLayout(new Graph(), { type: 'force' });
        expect(layout.step(10)).toBe(0);
    });

    it('pushes unconnected nodes apart', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: -0.01, y: 0 });
        const b = g.addNode({ tag: 'n', x: 0.01, y: 0 });
        const before = nodeDistance(g, a, b);

        new ForceLayout(g, { type: 'force', centralGravity: 0 }).step(20);
        expect(nodeDistance(g, a, b)).toBeGreaterThan(before);
    });

    it('pulls connected nodes together', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: -0.9, y: 0 });
        const b = g.addNode({ tag: 'n', x: 0.9, y: 0 });
        g.addEdge({ tag: 'e', source: a, target: b });
        const before = nodeDistance(g, a, b);

        new ForceLayout(g, { type: 'force', gravitationalConstant: 0, centralGravity: 0 }).step(50);
        expect(nodeDistance(g, a, b)).toBeLessThan(before);
    });

    it('relaxes springs toward their rest length', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: -0.9, y: 0 });
        const b = g.addNode({ tag: 'n', x: 0.9, y: 0 });
        g.addEdge({ tag: 'e', source: a, target: b });

        new ForceLayout(g, {
            type: 'force',
            gravitationalConstant: 0,
            centralGravity: 0,
            springLength: 0.4,
            minVelocity: 0,
        }).step(3000);
        expect(nodeDistance(g, a, b)).toBeCloseTo(0.4, 1);
    });

    it('pulls isolated nodes toward the origin', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: 0.8, y: 0.6 });
        const before = Math.hypot(g.positions[0], g.positions[1]);

        new ForceLayout(g, { type: 'force' }).step(50);
        expect(Math.hypot(g.positions[a * 2], g.positions[a * 2 + 1])).toBeLessThan(before);
    });

    it('clamps velocity to maxVelocity', () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: -0.001, y: 0 });
        g.addNode({ tag: 'n', x: 0.001, y: 0 });

        const maxVel = new ForceLayout(g, { type: 'force', maxVelocity: 0.01 }).step(1);
        // Per-axis clamp → magnitude ≤ √2 · maxVelocity
        expect(maxVel).toBeLessThanOrEqual(0.01 * Math.SQRT2 + 1e-9);
    });

    it('stops early once stable (non-animated)', () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: 0, y: 0 });
        const layout = new ForceLayout(g, { type: 'force' });
        expect(layout.step(100)).toBeLessThan(0.0005);
    });

    it('keeps pinned nodes in place', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: 0.5, y: 0.5 });
        const b = g.addNode({ tag: 'n', x: -0.5, y: -0.5 });
        g.addEdge({ tag: 'e', source: a, target: b });

        const layout = new ForceLayout(g, { type: 'force' });
        layout.pin(a);
        layout.step(30);
        expect(g.positions[a * 2]).toBeCloseTo(0.5);
        expect(g.positions[a * 2 + 1]).toBeCloseTo(0.5);

        layout.unpin(a);
        layout.step(30);
        expect(g.positions[a * 2]).not.toBeCloseTo(0.5);
    });

    it('skips removed nodes and their edges', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: 0.5, y: 0 });
        const b = g.addNode({ tag: 'n', x: -0.5, y: 0 });
        const c = g.addNode({ tag: 'n', x: 0, y: 0.5 });
        g.addEdge({ tag: 'e', source: a, target: b });
        g.addEdge({ tag: 'e', source: b, target: c });
        g.removeNode(b);

        new ForceLayout(g, { type: 'force' }).step(20);
        expect(g.positions[b * 2]).toBe(0);
        expect(g.positions[b * 2 + 1]).toBe(0);
        for (const v of g.positions.subarray(0, 6)) expect(Number.isFinite(v)).toBe(true);
    });

    it('stays finite on a random graph (Barnes-Hut and brute-force)', () => {
        for (const theta of [0, 0.5]) {
            const g = randomGraph(300, 600, 11);
            new ForceLayout(g, { type: 'force', barnesHutTheta: theta }).step(100);
            for (const v of g.positions.subarray(0, 600)) {
                expect(Number.isFinite(v)).toBe(true);
            }
        }
    });

    it('separates nodes placed at the same position', () => {
        for (const theta of [0, 0.5]) {
            const g = new Graph();
            const a = g.addNode({ tag: 'n', x: 0.2, y: 0.2 });
            const b = g.addNode({ tag: 'n', x: 0.2, y: 0.2 });
            g.addNode({ tag: 'n', x: -0.4, y: 0.1 });

            expect(() => new ForceLayout(g, { type: 'force', barnesHutTheta: theta }).step(20)).not.toThrow();
            expect(nodeDistance(g, a, b)).toBeGreaterThan(0.01);
            for (const v of g.positions.subarray(0, 6)) expect(Number.isFinite(v)).toBe(true);
        }
    });

    it('marks the graph dirty after a step', () => {
        const g = randomGraph(10, 10);
        g.dirtyNodes = false;
        g.dirtyEdges = false;
        new ForceLayout(g, { type: 'force' }).step(1);
        expect(g.dirtyNodes).toBe(true);
        expect(g.dirtyEdges).toBe(true);
    });

    it('handles nodes added after construction', () => {
        const g = randomGraph(5, 5);
        const layout = new ForceLayout(g, { type: 'force' });
        for (let i = 0; i < 50; i++) g.addNode({ tag: 'late', x: i / 100, y: 0.3 });

        expect(() => layout.step(5)).not.toThrow();
        for (const v of g.positions.subarray(0, g.nodeSlots * 2)) {
            expect(Number.isFinite(v)).toBe(true);
        }
    });

    it('moves nodes added after construction', () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: 0, y: 0 });
        const layout = new ForceLayout(g, { type: 'force' });
        layout.step(1);

        const late = g.addNode({ tag: 'n', x: 0.05, y: 0 });
        layout.step(10);
        expect(g.positions[late * 2]).toBeGreaterThan(0.05);
    });

    it('supports the legacy option names without breaking', () => {
        const g = randomGraph(20, 20);
        const layout = new ForceLayout(g, { type: 'force', repulsion: 1, attraction: 1, gravity: 1 });
        expect(() => layout.step(5)).not.toThrow();
    });
});

// -----------------------------------------------------------
// Run loop
// -----------------------------------------------------------

describe('ForceLayout: run loop', () => {
    it('starts, ticks and stops', () => {
        const g = randomGraph(50, 80);
        const layout = new ForceLayout(g, { type: 'force' });
        const onTick = vi.fn();
        const onStop = vi.fn();
        layout.onTick = onTick;
        layout.onStop = onStop;

        layout.start();
        expect(layout.isRunning()).toBe(true);

        flushFrames(3);
        expect(onTick).toHaveBeenCalledTimes(3);

        layout.stop();
        expect(layout.isRunning()).toBe(false);
        expect(onStop).toHaveBeenCalled();

        flushFrames(3);
        expect(onTick).toHaveBeenCalledTimes(3);
    });

    it('auto-stops at maxIterations', () => {
        const g = randomGraph(50, 80);
        const layout = new ForceLayout(g, { type: 'force', maxIterations: 9, minVelocity: 0 });
        layout.start();
        flushFrames(100);
        expect(layout.isRunning()).toBe(false);
    });

    it('never auto-stops in animated mode', () => {
        const g = randomGraph(50, 80);
        const layout = new ForceLayout(g, { type: 'force', maxIterations: 9 });
        layout.setAnimated(true);
        expect(layout.isAnimated()).toBe(true);
        expect(layout.isRunning()).toBe(true);

        flushFrames(20);
        expect(layout.isRunning()).toBe(true);
        layout.stop();
    });

    it('keeps running when nodes are added in animated mode', () => {
        const g = randomGraph(20, 30);
        const layout = new ForceLayout(g, { type: 'force' });
        layout.setAnimated(true);
        flushFrames(2);

        for (let i = 0; i < 100; i++) g.addNode({ tag: 'late', x: i / 200, y: -0.1 });
        expect(() => flushFrames(5)).not.toThrow();
        expect(layout.isRunning()).toBe(true);
        for (const v of g.positions.subarray(0, g.nodeSlots * 2)) {
            expect(Number.isFinite(v)).toBe(true);
        }
        layout.stop();
    });
});
