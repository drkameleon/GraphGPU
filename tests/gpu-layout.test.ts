// ============================================================
// graphGPU - GPUForceLayout lifecycle tests (fake GPU device)
// ============================================================

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Graph } from '../src/core/Graph';
import { GPUForceLayout } from '../src/layout/GPUForceLayout';
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
