// ============================================================
// graphGPU - GPU Force-Directed Layout (Compute Shader)
// ============================================================
// Runs the vis.js-style force simulation entirely on the GPU
// via WebGPU compute shaders:
//   Pass 1: Reset forces
//   Pass 2: Gravitational repulsion (N-body, O(n²))
//   Pass 3: Hooke spring attraction along edges
//   Pass 4: Central gravity toward origin
//   Pass 5: Velocity integration (F - damping*v) / mass
// ============================================================

import { FORCE_COMPUTE_SHADER } from '../shaders/index';
import type { Graph } from '../core/Graph';

export interface GPUForceConfig {
    gravitationalConstant: number;
    springLength: number;
    springConstant: number;
    centralGravity: number;
    damping: number;
    timestep: number;
    maxVelocity: number;
    stepsPerFrame: number;
    maxIterations: number;
}

const DEFAULTS: GPUForceConfig = {
    gravitationalConstant: -0.25,
    springLength: 0.2,
    springConstant: 0.06,
    centralGravity: 0.012,
    damping: 0.18,
    timestep: 0.35,
    maxVelocity: 0.06,
    stepsPerFrame: 3,
    maxIterations: 1000,
};

const WORKGROUP_SIZE = 64;

/**
 * GPU-accelerated force-directed layout using vis.js-style physics.
 *
 * Architecture:
 * - 5 compute passes per step (reset, repulsion, attraction, gravity, integrate)
 * - Forces buffer accumulates per-node forces each step
 * - Positions read back to CPU after each frame for rendering
 * - Edge attraction has benign data races (acceptable for force layout)
 */
export class GPUForceLayout {
    private config: GPUForceConfig;
    private device: GPUDevice;
    private graph: Graph;

    // Compute pipelines (5 passes)
    private resetPipeline!: GPUComputePipeline;
    private repulsionPipeline!: GPUComputePipeline;
    private attractionPipeline!: GPUComputePipeline;
    private gravityPipeline!: GPUComputePipeline;
    private integrationPipeline!: GPUComputePipeline;

    // GPU Buffers
    private positionsBuffer!: GPUBuffer;
    private velocitiesBuffer!: GPUBuffer;
    private forcesBuffer!: GPUBuffer;
    private edgesBuffer!: GPUBuffer;
    private sizesBuffer!: GPUBuffer;
    private paramsBuffer!: GPUBuffer;
    private readbackBuffer!: GPUBuffer;

    // Bind group
    private bindGroup!: GPUBindGroup;
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

        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.resetPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_reset_forces' },
        });
        this.repulsionPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_repulsion' },
        });
        this.attractionPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_attraction' },
        });
        this.gravityPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_gravity' },
        });
        this.integrationPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_integrate' },
        });

        // Params: 12 x f32 = 48 bytes (first 2 are u32)
        this.paramsBuffer = this.device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.allocateBuffers();
    }

    private allocateBuffers(): void {
        const nNodes = Math.max(this.graph.numNodes, 64);
        const nEdges = Math.max(this.graph.numEdges, 64);

        this.positionsBuffer?.destroy();
        this.velocitiesBuffer?.destroy();
        this.forcesBuffer?.destroy();
        this.edgesBuffer?.destroy();
        this.sizesBuffer?.destroy();
        this.readbackBuffer?.destroy();

        this.positionsBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.velocitiesBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.forcesBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.edgesBuffer = this.device.createBuffer({
            size: nEdges * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.sizesBuffer = this.device.createBuffer({
            size: nNodes * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.readbackBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.nodeCapacity = nNodes;
        this.edgeCapacity = nEdges;
        this.createBindGroup();
    }

    private createBindGroup(): void {
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: this.positionsBuffer } },
                { binding: 2, resource: { buffer: this.velocitiesBuffer } },
                { binding: 3, resource: { buffer: this.edgesBuffer } },
                { binding: 4, resource: { buffer: this.sizesBuffer } },
                { binding: 5, resource: { buffer: this.forcesBuffer } },
            ],
        });
    }

    private uploadData(): void {
        const n = this.graph.numNodes;
        const e = this.graph.numEdges;
        if (n > this.nodeCapacity || e > this.edgeCapacity) {
            this.allocateBuffers();
        }
        this.device.queue.writeBuffer(this.positionsBuffer, 0, this.graph.positions.buffer, 0, n * 2 * 4);
        this.device.queue.writeBuffer(this.sizesBuffer, 0, this.graph.sizes.buffer, 0, n * 4);
        this.device.queue.writeBuffer(this.edgesBuffer, 0, this.graph.edgeIndices.buffer, 0, e * 2 * 4);
        const zeros = new Float32Array(n * 2);
        this.device.queue.writeBuffer(this.velocitiesBuffer, 0, zeros);
        this.device.queue.writeBuffer(this.forcesBuffer, 0, zeros);
    }

    private uploadParams(): void {
        const buf = new ArrayBuffer(48);
        const u32 = new Uint32Array(buf);
        const f32 = new Float32Array(buf);
        u32[0] = this.graph.numNodes;
        u32[1] = this.graph.numEdges;
        f32[2] = this.config.gravitationalConstant;
        f32[3] = this.config.springLength;
        f32[4] = this.config.springConstant;
        f32[5] = this.config.centralGravity;
        f32[6] = this.config.damping;
        f32[7] = this.config.timestep;
        f32[8] = this.config.maxVelocity;
        f32[9] = 0; f32[10] = 0; f32[11] = 0;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buf);
    }

    private dispatchStep(): void {
        const n = this.graph.numNodes;
        const e = this.graph.numEdges;
        const nodeGroups = Math.ceil(n / WORKGROUP_SIZE);
        const edgeGroups = Math.ceil(e / WORKGROUP_SIZE);

        const encoder = this.device.createCommandEncoder();

        const p1 = encoder.beginComputePass();
        p1.setPipeline(this.resetPipeline);
        p1.setBindGroup(0, this.bindGroup);
        p1.dispatchWorkgroups(nodeGroups);
        p1.end();

        const p2 = encoder.beginComputePass();
        p2.setPipeline(this.repulsionPipeline);
        p2.setBindGroup(0, this.bindGroup);
        p2.dispatchWorkgroups(nodeGroups);
        p2.end();

        const p3 = encoder.beginComputePass();
        p3.setPipeline(this.attractionPipeline);
        p3.setBindGroup(0, this.bindGroup);
        p3.dispatchWorkgroups(edgeGroups);
        p3.end();

        const p4 = encoder.beginComputePass();
        p4.setPipeline(this.gravityPipeline);
        p4.setBindGroup(0, this.bindGroup);
        p4.dispatchWorkgroups(nodeGroups);
        p4.end();

        const p5 = encoder.beginComputePass();
        p5.setPipeline(this.integrationPipeline);
        p5.setBindGroup(0, this.bindGroup);
        p5.dispatchWorkgroups(nodeGroups);
        p5.end();

        this.device.queue.submit([encoder.finish()]);
    }

    private async readbackPositions(): Promise<void> {
        const n = this.graph.numNodes;
        const byteSize = n * 2 * 4;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.positionsBuffer, 0, this.readbackBuffer, 0, byteSize);
        this.device.queue.submit([encoder.finish()]);
        await this.readbackBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
        this.readbackBuffer.unmap();
        this.graph.positions.set(data.subarray(0, n * 2));
        this.graph.dirtyNodes = true;
        this.graph.dirtyEdges = true;
    }

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

    stop(): void {
        this.running = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
    }

    async step(n: number): Promise<void> {
        this.uploadData();
        this.uploadParams();
        for (let i = 0; i < n; i++) {
            this.dispatchStep();
        }
        await this.readbackPositions();
    }

    destroy(): void {
        this.stop();
        this.positionsBuffer?.destroy();
        this.velocitiesBuffer?.destroy();
        this.forcesBuffer?.destroy();
        this.edgesBuffer?.destroy();
        this.sizesBuffer?.destroy();
        this.paramsBuffer?.destroy();
        this.readbackBuffer?.destroy();
    }
}
