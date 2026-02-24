// ============================================================
// graphGPU - GPU Force-Directed Layout (Compute Shader)
// ============================================================
// Runs the force simulation on the GPU via WebGPU compute.
// Falls back to CPU layout if compute is unavailable.
// ============================================================

import { FORCE_COMPUTE_SHADER } from '../shaders/index';
import type { Graph } from '../core/Graph';

export interface GPUForceConfig {
    repulsion: number;
    attraction: number;
    gravity: number;
    damping: number;
    deltaTime: number;
    stepsPerFrame: number;
    maxIterations: number;
}

const DEFAULTS: GPUForceConfig = {
    repulsion: 1.0,
    attraction: 0.01,
    gravity: 0.05,
    damping: 0.92,
    deltaTime: 0.016,
    stepsPerFrame: 3,
    maxIterations: 500,
};

const WORKGROUP_SIZE = 64;

/**
 * GPU-accelerated force-directed layout.
 *
 * Architecture:
 * - Positions buffer is shared with the Renderer (avoids readback on every frame)
 * - Velocities buffer is GPU-only
 * - Edge indices uploaded once, re-uploaded on graph change
 * - Param uniform updated per dispatch
 *
 * Note: The attraction kernel has potential data races on velocity writes.
 * This is acceptable for force-directed layout - the races introduce small
 * errors that are equivalent to noise/jitter and don't affect convergence.
 */
export class GPUForceLayout {
    private config: GPUForceConfig;
    private device: GPUDevice;
    private graph: Graph;

    // Compute pipelines
    private repulsionPipeline!: GPUComputePipeline;
    private attractionPipeline!: GPUComputePipeline;
    private integrationPipeline!: GPUComputePipeline;

    // GPU Buffers
    private positionsBuffer!: GPUBuffer;     // read-write storage (positions)
    private velocitiesBuffer!: GPUBuffer;    // read-write storage
    private edgesBuffer!: GPUBuffer;         // read-only storage
    private sizesBuffer!: GPUBuffer;         // read-only storage (for skip check)
    private paramsBuffer!: GPUBuffer;        // uniform
    private readbackBuffer!: GPUBuffer;      // for reading positions back to CPU

    // Bind groups
    private repulsionBindGroup!: GPUBindGroup;
    private attractionBindGroup!: GPUBindGroup;
    private integrationBindGroup!: GPUBindGroup;

    // Layout
    private bindGroupLayout!: GPUBindGroupLayout;

    // State
    private running = false;
    private iteration = 0;
    private animationId = 0;
    private nodeCapacity = 0;
    private edgeCapacity = 0;

    constructor(device: GPUDevice, graph: Graph, opts?: Partial<GPUForceConfig>) {
        this.device = device;
        this.graph = graph;
        this.config = { ...DEFAULTS, ...opts };
        this.init();
    }

    private init(): void {
        const module = this.device.createShaderModule({ code: FORCE_COMPUTE_SHADER });

        // Shared bind group layout for all 3 kernels
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.repulsionPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_repulsion' },
        });

        this.attractionPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_attraction' },
        });

        this.integrationPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_integrate' },
        });

        // Params uniform (8 floats = 32 bytes, but first 2 are u32)
        this.paramsBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.allocateBuffers();
    }

    private allocateBuffers(): void {
        const nNodes = Math.max(this.graph.numNodes, 64);
        const nEdges = Math.max(this.graph.numEdges, 64);

        // Destroy old buffers
        this.positionsBuffer?.destroy();
        this.velocitiesBuffer?.destroy();
        this.edgesBuffer?.destroy();
        this.sizesBuffer?.destroy();
        this.readbackBuffer?.destroy();

        // Positions: vec2f per node
        this.positionsBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Velocities: vec2f per node (starts at zero)
        this.velocitiesBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Edges: vec2u per edge (src, tgt)
        this.edgesBuffer = this.device.createBuffer({
            size: nEdges * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Sizes: f32 per node
        this.sizesBuffer = this.device.createBuffer({
            size: nNodes * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Readback buffer for GPU → CPU position transfer
        this.readbackBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.nodeCapacity = nNodes;
        this.edgeCapacity = nEdges;

        this.createBindGroups();
    }

    private createBindGroups(): void {
        const entries = [
            { binding: 0, resource: { buffer: this.paramsBuffer } },
            { binding: 1, resource: { buffer: this.positionsBuffer } },
            { binding: 2, resource: { buffer: this.velocitiesBuffer } },
            { binding: 3, resource: { buffer: this.edgesBuffer } },
            { binding: 4, resource: { buffer: this.sizesBuffer } },
        ];

        // All 3 kernels share the same layout and buffers
        const descriptor = { layout: this.bindGroupLayout, entries };
        this.repulsionBindGroup = this.device.createBindGroup(descriptor);
        this.attractionBindGroup = this.device.createBindGroup(descriptor);
        this.integrationBindGroup = this.device.createBindGroup(descriptor);
    }

    /** Upload current graph state to GPU */
    private uploadData(): void {
        const n = this.graph.numNodes;
        const e = this.graph.numEdges;

        if (n > this.nodeCapacity || e > this.edgeCapacity) {
            this.allocateBuffers();
        }

        // Upload positions
        this.device.queue.writeBuffer(
            this.positionsBuffer, 0,
            this.graph.positions.buffer, 0, n * 2 * 4,
        );

        // Upload sizes
        this.device.queue.writeBuffer(
            this.sizesBuffer, 0,
            this.graph.sizes.buffer, 0, n * 4,
        );

        // Upload edges
        this.device.queue.writeBuffer(
            this.edgesBuffer, 0,
            this.graph.edgeIndices.buffer, 0, e * 2 * 4,
        );

        // Zero velocities
        const zeros = new Float32Array(n * 2);
        this.device.queue.writeBuffer(this.velocitiesBuffer, 0, zeros);
    }

    /** Write params uniform */
    private uploadParams(): void {
        const buf = new ArrayBuffer(32);
        const u32 = new Uint32Array(buf);
        const f32 = new Float32Array(buf);

        u32[0] = this.graph.numNodes;
        u32[1] = this.graph.numEdges;
        f32[2] = this.config.repulsion;
        f32[3] = this.config.attraction;
        f32[4] = this.config.gravity;
        f32[5] = this.config.damping;
        f32[6] = this.config.deltaTime;
        f32[7] = 0; // padding

        this.device.queue.writeBuffer(this.paramsBuffer, 0, buf);
    }

    /** Dispatch one simulation step (3 compute passes) */
    private dispatchStep(): void {
        const n = this.graph.numNodes;
        const e = this.graph.numEdges;
        const nodeGroups = Math.ceil(n / WORKGROUP_SIZE);
        const edgeGroups = Math.ceil(e / WORKGROUP_SIZE);

        const encoder = this.device.createCommandEncoder();

        // Pass 1: Repulsion (dispatches over nodes)
        const repPass = encoder.beginComputePass();
        repPass.setPipeline(this.repulsionPipeline);
        repPass.setBindGroup(0, this.repulsionBindGroup);
        repPass.dispatchWorkgroups(nodeGroups);
        repPass.end();

        // Pass 2: Attraction (dispatches over edges)
        const attPass = encoder.beginComputePass();
        attPass.setPipeline(this.attractionPipeline);
        attPass.setBindGroup(0, this.attractionBindGroup);
        attPass.dispatchWorkgroups(edgeGroups);
        attPass.end();

        // Pass 3: Integration (dispatches over nodes)
        const intPass = encoder.beginComputePass();
        intPass.setPipeline(this.integrationPipeline);
        intPass.setBindGroup(0, this.integrationBindGroup);
        intPass.dispatchWorkgroups(nodeGroups);
        intPass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    /** Read positions back from GPU to CPU (async) */
    private async readbackPositions(): Promise<void> {
        const n = this.graph.numNodes;
        const byteSize = n * 2 * 4;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.positionsBuffer, 0, this.readbackBuffer, 0, byteSize);
        this.device.queue.submit([encoder.finish()]);

        await this.readbackBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
        this.readbackBuffer.unmap();

        // Write back to graph positions
        this.graph.positions.set(data.subarray(0, n * 2));
        this.graph.dirtyNodes = true;
        this.graph.dirtyEdges = true;
    }

    /** Start the layout loop */
    async start(): Promise<void> {
        if (this.running) return;
        this.running = true;
        this.iteration = 0;

        this.uploadData();
        this.uploadParams();

        const loop = async () => {
            if (!this.running) return;

            for (let i = 0; i < this.config.stepsPerFrame; i++) {
                this.dispatchStep();
                this.iteration++;
            }

            await this.readbackPositions();

            if (this.iteration < this.config.maxIterations && this.running) {
                this.animationId = requestAnimationFrame(() => loop());
            } else {
                this.running = false;
            }
        };

        loop();
    }

    /** Stop the layout */
    stop(): void {
        this.running = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
    }

    /** Run N steps synchronously (blocking - useful for initial settle) */
    async step(n: number): Promise<void> {
        this.uploadData();
        this.uploadParams();

        for (let i = 0; i < n; i++) {
            this.dispatchStep();
        }

        await this.readbackPositions();
    }

    /** Destroy GPU resources */
    destroy(): void {
        this.stop();
        this.positionsBuffer?.destroy();
        this.velocitiesBuffer?.destroy();
        this.edgesBuffer?.destroy();
        this.sizesBuffer?.destroy();
        this.paramsBuffer?.destroy();
        this.readbackBuffer?.destroy();
    }
}
