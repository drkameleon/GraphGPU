// ============================================================
// graphGPU - Fake WebGPU + animation frames (for tests)
// ============================================================
// No real GPU in CI: a minimal fake device validates buffer
// bounds like WebGPU would and records what gets submitted.
// Its "compute" pretends to move every node by +1 per submit
// with dispatches, so layout readbacks are observable.
// ============================================================

import { vi } from 'vitest';

export const gpu = {
    /** Validation errors (out-of-range writes/copies/maps) */
    errors: [] as string[],
    /** Keep mapAsync() pending until releaseMaps() */
    holdMaps: false,
    pendingMaps: [] as (() => void)[],
    /** Submitted command buffers that contained a render pass */
    renderPasses: 0,
    /** Draw calls: [vertexCount, instanceCount] */
    draws: [] as [number, number][],
};

export function releaseMaps(): void {
    const maps = gpu.pendingMaps;
    gpu.pendingMaps = [];
    for (const done of maps) done();
}

// -----------------------------------------------------------
// Buffers
// -----------------------------------------------------------

export class FakeBuffer {
    data: ArrayBuffer;
    destroyed = false;
    private reject: ((e: Error) => void) | null = null;

    constructor(public size: number) {
        this.data = new ArrayBuffer(size);
    }

    mapAsync(_mode: number, offset = 0, size = this.size - offset): Promise<void> {
        if (this.destroyed) return Promise.reject(new Error('destroyed'));
        if (offset + size > this.size) gpu.errors.push('mapAsync out of range');
        return new Promise((resolve, reject) => {
            this.reject = reject;
            const done = () => { this.reject = null; resolve(); };
            if (gpu.holdMaps) gpu.pendingMaps.push(done);
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

// -----------------------------------------------------------
// Device
// -----------------------------------------------------------

interface FakeCommands {
    dispatched: boolean;
    rendered: boolean;
    copies: (() => void)[];
}

export function fakeDevice(): GPUDevice {
    let positions: FakeBuffer | null = null;

    const view = () => ({});
    const device = {
        createShaderModule: () => ({}),
        createBindGroupLayout: () => ({}),
        createPipelineLayout: () => ({}),
        createComputePipeline: () => ({}),
        createRenderPipeline: () => ({}),
        createTexture: () => ({ createView: view, destroy() {} }),
        createBindGroup: (desc: GPUBindGroupDescriptor) => {
            // Compute layout: binding 1 = positions
            const entry = [...desc.entries].find(e => e.binding === 1);
            const buffer = (entry?.resource as { buffer?: FakeBuffer } | undefined)?.buffer;
            if (buffer) positions = buffer;
            return {};
        },
        createBuffer: (desc: GPUBufferDescriptor) => new FakeBuffer(desc.size),
        createCommandEncoder: () => {
            const cmd: FakeCommands = { dispatched: false, rendered: false, copies: [] };
            return {
                beginComputePass: () => ({
                    setPipeline() {},
                    setBindGroup() {},
                    dispatchWorkgroups() { cmd.dispatched = true; },
                    end() {},
                }),
                beginRenderPass: () => {
                    cmd.rendered = true;
                    return {
                        setPipeline() {},
                        setBindGroup() {},
                        setVertexBuffer() {},
                        draw(vertices: number, instances = 1) { gpu.draws.push([vertices, instances]); },
                        end() {},
                    };
                },
                copyBufferToBuffer(src: FakeBuffer, so: number, dst: FakeBuffer, d: number, size: number) {
                    if (so + size > src.size || d + size > dst.size) {
                        gpu.errors.push(`copy of ${size} bytes out of range`);
                        return;
                    }
                    cmd.copies.push(() => new Uint8Array(dst.data, d, size).set(new Uint8Array(src.data, so, size)));
                },
                finish: () => cmd,
            };
        },
        queue: {
            writeBuffer(buf: FakeBuffer, offset: number, data: ArrayBufferLike | ArrayBufferView, dataOffset = 0, size?: number) {
                const src = ArrayBuffer.isView(data)
                    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                    : new Uint8Array(data);
                // Like WebGPU: offset/size count elements for typed arrays, bytes otherwise
                const elem = ArrayBuffer.isView(data) ? (data as Float32Array).BYTES_PER_ELEMENT ?? 1 : 1;
                const start = dataOffset * elem;
                const n = size !== undefined ? size * elem : src.byteLength - start;
                if (offset + n > buf.size) {
                    gpu.errors.push(`write of ${n} bytes out of range`);
                    return;
                }
                new Uint8Array(buf.data, offset, n).set(src.subarray(start, start + n));
            },
            submit(cmds: FakeCommands[]) {
                for (const cmd of cmds) {
                    if (cmd.dispatched && positions) {
                        const f = new Float32Array(positions.data);
                        for (let i = 0; i < f.length; i++) f[i] += 1;
                    }
                    if (cmd.rendered) gpu.renderPasses++;
                    for (const copy of cmd.copies) copy();
                }
            },
        },
        destroy() {},
    };
    return device as unknown as GPUDevice;
}

/**
 * Install WebGPU globals, a fake `navigator.gpu` and a fake
 * 'webgpu' canvas context, so Renderer.init() succeeds.
 */
export function installFakeWebGPU(canvas?: HTMLCanvasElement): void {
    gpu.errors = [];
    gpu.holdMaps = false;
    gpu.pendingMaps = [];
    gpu.renderPasses = 0;
    gpu.draws = [];

    vi.stubGlobal('GPUBufferUsage', {
        MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
        INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128,
    });
    vi.stubGlobal('GPUShaderStage', { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 });
    vi.stubGlobal('GPUMapMode', { READ: 1, WRITE: 2 });
    vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 16 });

    if (canvas) {
        const device = fakeDevice();
        vi.stubGlobal('navigator', {
            ...globalThis.navigator,
            gpu: {
                requestAdapter: async () => ({ requestDevice: async () => device }),
                getPreferredCanvasFormat: () => 'bgra8unorm',
            },
        });
        const context = {
            configure() {},
            getCurrentTexture: () => ({ createView: () => ({}) }),
        };
        const original = canvas.getContext.bind(canvas);
        vi.spyOn(canvas, 'getContext').mockImplementation(((type: string) =>
            type === 'webgpu' ? context : original(type as '2d')) as typeof canvas.getContext);
    }
}

// -----------------------------------------------------------
// Animation frames
// -----------------------------------------------------------

let frames = new Map<number, FrameRequestCallback>();
let nextFrameId = 1;
let now = 0;

/** Replace requestAnimationFrame with a manually stepped one */
export function installFakeFrames(): void {
    frames = new Map();
    now = 0;
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
        const id = nextFrameId++;
        frames.set(id, cb);
        return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
}

export function pendingFrames(): number {
    return frames.size;
}

/** Run `count` frames (each ~16ms apart), letting promises settle in between */
export async function flushFrames(count: number): Promise<void> {
    for (let i = 0; i < count && frames.size > 0; i++) {
        const pending = [...frames.values()];
        frames = new Map();
        now += 16;
        for (const cb of pending) cb(now);
        await new Promise<void>(r => setTimeout(r, 0));
    }
}
