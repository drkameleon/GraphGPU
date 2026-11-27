// ============================================================
// graphGPU - GPUForceLayout lifecycle tests (fake GPU device)
// ============================================================
// No real GPU in CI: a minimal fake device validates buffer
// bounds like WebGPU would, and its "compute" pretends to move
// every node by +1 per step, so readbacks are observable.
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Graph } from '../src/core/Graph';
import { GPUForceLayout } from '../src/layout/GPUForceLayout';
import { randomGraph } from './helpers';

// -----------------------------------------------------------
// Fake WebGPU
// -----------------------------------------------------------

let errors: string[] = [];
let holdMaps = false;
let pendingMaps: (() => void)[] = [];

class FakeBuffer {
    data: ArrayBuffer;
    destroyed = false;
    private reject: ((e: Error) => void) | null = null;

    constructor(public size: number) {
        this.data = new ArrayBuffer(size);
    }

    mapAsync(_mode: number, offset = 0, size = this.size - offset): Promise<void> {
        if (this.destroyed) return Promise.reject(new Error('destroyed'));
        if (offset + size > this.size) errors.push('mapAsync out of range');
        return new Promise((resolve, reject) => {
            this.reject = reject;
            const done = () => { this.reject = null; resolve(); };
            if (holdMaps) pendingMaps.push(done);
            else queueMicrotask(done);
        });
    }

    getMappedRange(offset = 0, size = this.size - offset): ArrayBuffer {
        return this.data.slice(offset, offset + size);
    }

    unmap(): void {}

    destroy(): void {
        this.destroyed = true;
        this.reject?.(new Error('AbortError: buffer destroyed'));
        this.reject = null;
    }
}

function fakeDevice(): GPUDevice {
    const buffers: FakeBuffer[] = [];
    let positions: FakeBuffer | null = null;

    const device = {
        createShaderModule: () => ({}),
        createBindGroupLayout: () => ({}),
        createPipelineLayout: () => ({}),
        createComputePipeline: () => ({}),
        createBindGroup: (desc: GPUBindGroupDescriptor) => {
            // binding 1 = positions
            const entry = [...desc.entries].find(e => e.binding === 1);
            positions = (entry?.resource as { buffer: FakeBuffer } | undefined)?.buffer ?? positions;
            return {};
        },
        createBuffer: (desc: GPUBufferDescriptor) => {
            const b = new FakeBuffer(desc.size);
            buffers.push(b);
            return b;
        },
        createCommandEncoder: () => {
            let dispatched = false;
            const copies: (() => void)[] = [];
            return {
                beginComputePass: () => ({
                    setPipeline() {},
                    setBindGroup() {},
                    dispatchWorkgroups() { dispatched = true; },
                    end() {},
                }),
                copyBufferToBuffer(src: FakeBuffer, so: number, dst: FakeBuffer, d: number, size: number) {
                    if (so + size > src.size || d + size > dst.size) {
                        errors.push(`copy of ${size} bytes out of range`);
                        return;
                    }
                    copies.push(() => new Uint8Array(dst.data, d, size).set(new Uint8Array(src.data, so, size)));
                },
                finish: () => ({ dispatched, copies }),
            };
        },
        queue: {
            writeBuffer(buf: FakeBuffer, offset: number, data: ArrayBufferLike | ArrayBufferView, dataOffset = 0, size?: number) {
                const src = ArrayBuffer.isView(data)
                    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                    : new Uint8Array(data);
                const n = size ?? src.byteLength - dataOffset;
                if (offset + n > buf.size) {
                    errors.push(`write of ${n} bytes out of range`);
                    return;
                }
                new Uint8Array(buf.data, offset, n).set(src.subarray(dataOffset, dataOffset + n));
            },
            submit(cmds: { dispatched: boolean; copies: (() => void)[] }[]) {
                for (const cmd of cmds) {
                    if (cmd.dispatched && positions) {
                        const f = new Float32Array(positions.data);
                        for (let i = 0; i < f.length; i++) f[i] += 1;
                    }
                    for (const copy of cmd.copies) copy();
                }
            },
        },
    };
    return device as unknown as GPUDevice;
}

// -----------------------------------------------------------
// Frames
// -----------------------------------------------------------

let frames = new Map<number, FrameRequestCallback>();
let nextFrameId = 1;

async function flushFrames(count: number): Promise<void> {
    for (let i = 0; i < count && frames.size > 0; i++) {
        const pending = [...frames.values()];
        frames = new Map();
        for (const cb of pending) cb(0);
        await settle();
    }
}

const settle = () => new Promise<void>(r => setTimeout(r, 0));

function releaseMaps(): void {
    const maps = pendingMaps;
    pendingMaps = [];
    for (const done of maps) done();
}

beforeEach(() => {
    errors = [];
    holdMaps = false;
    pendingMaps = [];
    frames = new Map();
    vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8, UNIFORM: 16, VERTEX: 32 });
    vi.stubGlobal('GPUShaderStage', { COMPUTE: 1, VERTEX: 2, FRAGMENT: 4 });
    vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
        const id = nextFrameId++;
        frames.set(id, cb);
        return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
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
        expect(errors).toEqual([]);
        layout.destroy();
    });

    it('picks up nodes added while running, without out-of-range copies', async () => {
        const g = randomGraph(10, 10);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1 });
        await layout.start();

        for (let i = 0; i < 200; i++) g.addNode({ tag: 'late', x: 0, y: 0 });
        g.addEdge({ tag: 'e', source: 0, target: 150 });
        await flushFrames(2);

        expect(errors).toEqual([]);
        // The new nodes were uploaded and stepped too
        expect(g.positions[150 * 2]).toBeGreaterThan(0);
        layout.destroy();
    });

    it('drops a readback that finishes after stop()', async () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: 0, y: 0 });
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1 });

        holdMaps = true;
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

        holdMaps = true;
        const started = layout.start();
        layout.destroy();

        // Must resolve, not reject (an unhandled rejection fails the run)
        await expect(started).resolves.toBeUndefined();
        expect(Array.from(g.positions.subarray(0, 40))).toEqual(Array.from(before));
        expect(frames.size).toBe(0);
    });

    it('runs a single loop after stop() + start()', async () => {
        const g = randomGraph(20, 20);
        const layout = new GPUForceLayout(fakeDevice(), g, { stepsPerFrame: 1, maxIterations: 1000 });
        await layout.start();
        layout.stop();
        await layout.start();
        await flushFrames(3);
        expect(frames.size).toBe(1);
        expect(layout.isRunning()).toBe(true);
        layout.destroy();
        expect(frames.size).toBe(0);
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
        expect(frames.size).toBe(0);
    });
});
