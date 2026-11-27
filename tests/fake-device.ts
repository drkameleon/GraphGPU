// ============================================================
// graphGPU - Fake WebGPU device (for tests & benchmarks)
// ============================================================
// No real GPU in CI: a minimal fake device validates buffer
// bounds like WebGPU would and records what gets submitted.
// Its "compute" pretends to move every node by +1 per submit
// with dispatches, so layout readbacks are observable.
//
// Framework-free (no vitest imports), so benchmarks can use it.
// ============================================================

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
    /** queue.writeBuffer calls: target buffer & byte count */
    writes: [] as { buffer: FakeBuffer; bytes: number }[],
    /** Entry points of the compute pipelines created */
    entryPoints: [] as string[],
    /** Devices handed out by the fake navigator.gpu */
    devices: [] as GPUDevice[],
};

/** Simulate losing the most recent device (driver reset, ...) */
export function loseDevice(reason: string = 'unknown'): void {
    const device = gpu.devices.at(-1) as unknown as { lose?: (r: string) => void };
    device?.lose?.(reason);
}

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
    /** Integration steps dispatched (each moves every node by +1) */
    dispatched: number;
    rendered: boolean;
    copies: (() => void)[];
}

export function fakeDevice(): GPUDevice {
    let positions: FakeBuffer | null = null;
    let lose: (reason: string) => void = () => {};
    const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
        lose = (reason) => resolve({ reason, message: `fake device lost (${reason})` } as GPUDeviceLostInfo);
    });

    const view = () => ({});
    const device = {
        createShaderModule: () => ({}),
        createBindGroupLayout: () => ({}),
        createPipelineLayout: () => ({}),
        createComputePipeline: (desc: GPUComputePipelineDescriptor) => {
            gpu.entryPoints.push(desc.compute.entryPoint ?? '');
            return { entryPoint: desc.compute.entryPoint };
        },
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
            const cmd: FakeCommands = { dispatched: 0, rendered: false, copies: [] };
            let pipeline: { entryPoint?: string } = {};
            return {
                beginComputePass: () => ({
                    setPipeline(p: { entryPoint?: string }) { pipeline = p; },
                    setBindGroup() {},
                    dispatchWorkgroups() {
                        // Shaders without a separate integrate pass count every dispatch
                        if (pipeline.entryPoint === undefined || pipeline.entryPoint === 'cs_integrate') cmd.dispatched++;
                    },
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
                gpu.writes.push({ buffer: buf, bytes: n });
            },
            submit(cmds: FakeCommands[]) {
                for (const cmd of cmds) {
                    if (cmd.dispatched > 0 && positions) {
                        const f = new Float32Array(positions.data);
                        for (let i = 0; i < f.length; i++) f[i] += cmd.dispatched;
                    }
                    if (cmd.rendered) gpu.renderPasses++;
                    for (const copy of cmd.copies) copy();
                }
            },
        },
        lost,
        lose,
        destroy() { lose('destroyed'); },
    };
    return device as unknown as GPUDevice;
}

/** WebGPU enum globals the library reads */
export const WEBGPU_GLOBALS: Record<string, Record<string, number>> = {
    GPUBufferUsage: {
        MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
        INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128,
    },
    GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
    GPUMapMode: { READ: 1, WRITE: 2 },
    GPUTextureUsage: { RENDER_ATTACHMENT: 16 },
};

/** A `navigator.gpu` that hands out a new fake device per request */
export function fakeNavigatorGPU() {
    return {
        requestAdapter: async () => ({
            requestDevice: async () => {
                const device = fakeDevice();
                gpu.devices.push(device);
                return device;
            },
        }),
        getPreferredCanvasFormat: () => 'bgra8unorm',
    };
}

/** A 'webgpu' canvas context */
export function fakeWebGPUContext() {
    return {
        configure() {},
        getCurrentTexture: () => ({ createView: () => ({}) }),
    };
}

/** Reset recorded state */
export function resetGPU(): void {
    gpu.errors = [];
    gpu.holdMaps = false;
    gpu.pendingMaps = [];
    gpu.renderPasses = 0;
    gpu.draws = [];
    gpu.writes = [];
    gpu.devices = [];
    gpu.entryPoints = [];
}
