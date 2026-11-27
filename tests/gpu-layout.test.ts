// ============================================================
// graphGPU - GPUForceLayout lifecycle tests (fake GPU device)
// ============================================================

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Graph } from '../src/core/Graph';
import { GPUForceLayout, buildAdjacency } from '../src/layout/GPUForceLayout';
import { randomGraph } from './helpers';
import {
    gpu, fakeDevice, releaseMaps,
    installFakeWebGPU, installFakeFrames, flushFrames, pendingFrames,
} from './fake-gpu';

beforeEach(() => {
    installFakeWebGPU();
    installFakeFrames();
});

afterEach(() => {
    vi.unstubAllGlobals();
});

// -----------------------------------------------------------
// Tests
// -----------------------------------------------------------

describe('GPUForceLayout', () => {
    it('reads positions back after stepping', async () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: 0.5, y: -0.5 });
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1 });

        await layout.step(2);
        expect(g.positions[0]).toBeCloseTo(2.5);
        expect(g.positions[1]).toBeCloseTo(1.5);
        expect(gpu.errors).toEqual([]);
        layout.destroy();
    });

    it('picks up nodes added while running, without out-of-range copies', async () => {
        const g = randomGraph(10, 10);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1 });
        await layout.start();

        for (let i = 0; i < 200; i++) g.addNode({ tag: 'late', x: 0, y: 0 });
        g.addEdge({ tag: 'e', source: 0, target: 150 });
        await flushFrames(2);

        expect(gpu.errors).toEqual([]);
        // The new nodes were uploaded and stepped too
        expect(g.positions[150 * 2]).toBeGreaterThan(0);
        layout.destroy();
    });

    it('drops a readback that finishes after stop()', async () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: 0, y: 0 });
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1 });

        gpu.holdMaps = true;
        const started = layout.start();
        layout.stop();

        // The user moves the node meanwhile
        g.setNodePosition(0, 0.25, 0.25);
        releaseMaps();
        await started;

        expect(g.positions[0]).toBeCloseTo(0.25);
        layout.destroy();
    });

    it('survives destroy() during a pending readback', async () => {
        const g = randomGraph(20, 20);
        const before = g.positions.slice(0, 40);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1 });

        gpu.holdMaps = true;
        const started = layout.start();
        layout.destroy();

        // Must resolve, not reject (an unhandled rejection fails the run)
        await expect(started).resolves.toBeUndefined();
        expect(Array.from(g.positions.subarray(0, 40))).toEqual(Array.from(before));
        expect(pendingFrames()).toBe(0);
    });

    it('runs a single loop after stop() + start()', async () => {
        const g = randomGraph(20, 20);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1, maxIterations: 1000 });
        await layout.start();
        layout.stop();
        await layout.start();
        await flushFrames(3);
        expect(pendingFrames()).toBe(1);
        expect(layout.isRunning()).toBe(true);
        layout.destroy();
        expect(pendingFrames()).toBe(0);
    });

    it('stops by itself at maxIterations', async () => {
        const g = randomGraph(20, 20);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 2, maxIterations: 6 });
        await layout.start();
        await flushFrames(10);
        expect(layout.isRunning()).toBe(false);
        layout.destroy();
    });

    it('does nothing once destroyed', async () => {
        const g = randomGraph(5, 5);
        const layout = new GPUForceLayout(fakeDevice(), g);
        layout.destroy();
        await layout.start();
        await layout.step(3);
        expect(layout.isRunning()).toBe(false);
        expect(pendingFrames()).toBe(0);
    });
});

describe('buildAdjacency', () => {
    const lists = (g: Graph) => {
        const { offsets, neighbors } = buildAdjacency(g, g.nodeSlots, g.edgeSlots);
        return Array.from({ length: g.nodeSlots }, (_, i) =>
            Array.from(neighbors.subarray(offsets[i], offsets[i + 1])).sort((a, b) => a - b));
    };

    it('lists every edge under both endpoints', () => {
        const g = new Graph();
        for (let i = 0; i < 4; i++) g.addNode({ tag: 'n' });
        g.addEdge({ tag: 'e', source: 0, target: 1 });
        g.addEdge({ tag: 'e', source: 0, target: 2 });
        g.addEdge({ tag: 'e', source: 3, target: 0 });
        expect(lists(g)).toEqual([[1, 2, 3], [0], [0], [0]]);
    });

    it('keeps parallel edges (each one is a spring)', () => {
        const g = new Graph();
        g.addNode({ tag: 'n' });
        g.addNode({ tag: 'n' });
        g.addEdge({ tag: 'e', source: 0, target: 1 });
        g.addEdge({ tag: 'e', source: 1, target: 0 });
        expect(lists(g)).toEqual([[1, 1], [0, 0]]);
    });

    it('skips removed edges, self-loops and out-of-range endpoints', () => {
        const g = new Graph();
        for (let i = 0; i < 3; i++) g.addNode({ tag: 'n' });
        const e = g.addEdge({ tag: 'e', source: 0, target: 1 });
        g.addEdge({ tag: 'e', source: 2, target: 2 });
        g.addEdge({ tag: 'e', source: 1, target: 2 });
        g.removeEdge(e);
        expect(lists(g)).toEqual([[], [2], [1]]);

        const { offsets } = buildAdjacency(g, 2, g.edgeSlots);
        expect(Array.from(offsets)).toEqual([0, 0, 0]);
    });

    it('handles a large random graph consistently', () => {
        const g = randomGraph(500, 2000, 3);
        const { offsets, neighbors } = buildAdjacency(g, g.nodeSlots, g.edgeSlots);
        let selfLoops = 0;
        for (const id of g.activeEdgeIds()) {
            if (g.edgeIndices[id * 2] === g.edgeIndices[id * 2 + 1]) selfLoops++;
        }
        expect(neighbors.length).toBe((2000 - selfLoops) * 2);
        expect(offsets[500]).toBe(neighbors.length);
        for (let i = 0; i < 500; i++) expect(offsets[i + 1]).toBeGreaterThanOrEqual(offsets[i]);
    });
});

describe('GPUForceLayout: settling', () => {
    it('stops by itself once nodes stop moving', async () => {
        gpu.stepDelta = 0; // the fake compute leaves positions alone
        const g = randomGraph(20, 20);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1, maxIterations: 100000 });
        await layout.start();
        await flushFrames(10);
        expect(layout.isRunning()).toBe(false);
        layout.destroy();
    });

    it('keeps running while nodes move', async () => {
        const g = randomGraph(20, 20);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1, maxIterations: 100000 });
        await layout.start();
        await flushFrames(10);
        expect(layout.isRunning()).toBe(true);
        layout.destroy();
    });
});
