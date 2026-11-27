// ============================================================
// graphGPU - GPU Force-Directed Layout (Compute Shader)
// ============================================================
// With a PositionTarget (the renderer), the layout writes node
// positions directly into the buffer the renderer draws from:
// nothing is copied back and re-uploaded per frame, and frames
// never wait on the GPU. The CPU copy of positions (for picking,
// labels, fitView) is refreshed by a background readback.
//
// Runs the vis.js-style force simulation entirely on the GPU
// via WebGPU compute shaders, two passes per step:
//   cs_forces     repulsion (tiled), springs (per-node adjacency,
//                 race-free) and central gravity
//   cs_integrate  velocity + position update
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
 * Owner of a shared node-positions buffer (e.g. the renderer).
 * The buffer must allow STORAGE | COPY_SRC | COPY_DST usage.
 */
export interface PositionTarget {
    /** Start sharing: make room for `slots` nodes and return the buffer */
    acquirePositions(slots: number): GPUBuffer;
    /** Stop sharing: CPU positions are up to date again */
    releasePositions(): void;
    /** New positions were computed on the GPU */
    positionsChanged(): void;
}

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

    // Compute pipelines (2 passes per step)
    private forcesPipeline!: GPUComputePipeline;
    private integrationPipeline!: GPUComputePipeline;

    // GPU Buffers
    private positionsBuffer!: GPUBuffer;    // shared (target) or own
    private ownsPositions = true;
    private velocitiesBuffer!: GPUBuffer;
    private forcesBuffer!: GPUBuffer;
    private neighborsBuffer!: GPUBuffer;   // CSR adjacency: neighbor ids
    private offsetsBuffer!: GPUBuffer;     // CSR adjacency: per-node start
    private sizesBuffer!: GPUBuffer;
    private paramsBuffer!: GPUBuffer;
    private readbackBuffer!: GPUBuffer;

    // Bind group
    private bindGroup!: GPUBindGroup;
    private bindGroupLayout!: GPUBindGroupLayout;

    // State
    private running = false;
    private destroyed = false;
    private iteration = 0;
    private animationId = 0;
    private nodeCapacity = 0;
    private edgeCapacity = 0;

    /** Bumped on every start/stop: async readbacks from an older run are dropped */
    private generation = 0;
    /** A readback is in flight (the buffer can only be mapped once) */
    private mapping = false;

    // What's currently on the GPU
    private uploadedNodes = 0;
    private uploadedVersion = -1;
    private uploadedNeighbors = 0;

    private target: PositionTarget | null;
    private sharing = false;

    // Callbacks
    onStart?: () => void;
    onTick?: (iteration: number) => void;
    onStop?: () => void;

    constructor(
        device: GPUDevice,
        graph: Graph,
        opts?: Partial<GPUForceConfig>,
        target?: PositionTarget,
    ) {
        this.device = device;
        this.graph = graph;
        this.config = { ...DEFAULTS, ...opts };
        this.target = target ?? null;
        this.ownsPositions = !this.target;
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
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.forcesPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'cs_forces' },
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
        // Grow with headroom, so adding nodes one by one doesn't
        // reallocate every time
        const nNodes = Math.max(64, this.graph.nodeSlots * 2);
        const nEdges = Math.max(64, this.graph.edgeSlots * 2);

        if (this.ownsPositions) this.positionsBuffer?.destroy();
        this.velocitiesBuffer?.destroy();
        this.forcesBuffer?.destroy();
        this.neighborsBuffer?.destroy();
        this.offsetsBuffer?.destroy();
        this.sizesBuffer?.destroy();
        this.readbackBuffer?.destroy();

        if (this.ownsPositions) {
            this.positionsBuffer = this.device.createBuffer({
                size: nNodes * 2 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
        } else if (this.sharing) {
            this.positionsBuffer = this.target!.acquirePositions(nNodes);
        }
        this.velocitiesBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.forcesBuffer = this.device.createBuffer({
            size: nNodes * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.neighborsBuffer = this.device.createBuffer({
            size: nEdges * 2 * 4, // both directions
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.offsetsBuffer = this.device.createBuffer({
            size: (nNodes + 1) * 4,
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
        if (this.positionsBuffer) this.createBindGroup();
    }

    /** Start writing into the target's buffer (if any) */
    private beginSharing(): void {
        if (!this.target || this.sharing) return;
        this.sharing = true;
        this.positionsBuffer = this.target.acquirePositions(this.nodeCapacity);
        this.createBindGroup();
    }

    private endSharing(): void {
        if (!this.sharing) return;
        this.sharing = false;
        this.target!.releasePositions();
    }

    /** The target may have reallocated its buffer (e.g. it grew) */
    private refreshSharedBuffer(): void {
        if (!this.sharing) return;
        const buffer = this.target!.acquirePositions(this.uploadedNodes);
        if (buffer !== this.positionsBuffer) {
            this.positionsBuffer = buffer;
            this.createBindGroup();
        }
    }

    private createBindGroup(): void {
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: this.positionsBuffer } },
                { binding: 2, resource: { buffer: this.velocitiesBuffer } },
                { binding: 3, resource: { buffer: this.neighborsBuffer } },
                { binding: 4, resource: { buffer: this.sizesBuffer } },
                { binding: 5, resource: { buffer: this.forcesBuffer } },
                { binding: 6, resource: { buffer: this.offsetsBuffer } },
            ],
        });
    }

    /**
     * Upload the graph. With `allPositions`, positions come from the
     * CPU; otherwise (re-sync while running) only slots that are new
     * since the last upload do, so the GPU's newer positions are kept.
     */
    private uploadData(allPositions: boolean = true): void {
        const n = this.graph.nodeSlots;
        const e = this.graph.edgeSlots;
        const grew = n > this.nodeCapacity || e > this.edgeCapacity;
        if (grew) {
            this.allocateBuffers();
        }

        const known = Math.min(this.uploadedNodes, n);

        // Positions: on a re-sync, keep what the GPU computed and only
        // write new slots. A shared buffer keeps its contents when the
        // target grows; our own one doesn't.
        const posFrom = allPositions || (grew && !this.sharing) ? 0 : known;
        if (n > posFrom) {
            this.device.queue.writeBuffer(
                this.positionsBuffer, posFrom * 8,
                this.graph.positions.buffer, posFrom * 8, (n - posFrom) * 8,
            );
        }
        this.device.queue.writeBuffer(this.sizesBuffer, 0, this.graph.sizes.buffer, 0, n * 4);
        this.uploadAdjacency(n, e);

        // Velocities: fresh buffers start at rest, otherwise only
        // the new slots do
        const vFrom = allPositions || grew ? 0 : known;
        if (n > vFrom) {
            const zeros = new Float32Array((n - vFrom) * 2);
            this.device.queue.writeBuffer(this.velocitiesBuffer, vFrom * 8, zeros);
            this.device.queue.writeBuffer(this.forcesBuffer, vFrom * 8, zeros);
        }

        this.uploadedNodes = n;
        this.uploadedVersion = this.graph.structureVersion;
    }

    /**
     * Build & upload the per-node neighbor lists (CSR). Each active
     * edge appears in both endpoints' lists, so every node sums its
     * own spring forces: no two threads write the same node.
     */
    private uploadAdjacency(n: number, e: number): void {
        const { offsets, neighbors } = buildAdjacency(this.graph, n, e);
        this.device.queue.writeBuffer(this.offsetsBuffer, 0, offsets);
        if (neighbors.length > 0) {
            this.device.queue.writeBuffer(this.neighborsBuffer, 0, neighbors);
        }
        this.uploadedNeighbors = neighbors.length;
    }

    /** Re-upload if nodes/edges were added or removed since the last upload */
    private syncWithGraph(): void {
        if (this.uploadedVersion === this.graph.structureVersion) return;
        this.uploadData(false);
        this.uploadParams();
    }

    private uploadParams(): void {
        const buf = new ArrayBuffer(48);
        const u32 = new Uint32Array(buf);
        const f32 = new Float32Array(buf);
        u32[0] = this.uploadedNodes;
        u32[1] = this.uploadedNeighbors;
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

    /** Run `steps` simulation steps in a single submit */
    private dispatchSteps(steps: number): void {
        const groups = Math.ceil(this.uploadedNodes / WORKGROUP_SIZE);
        if (groups === 0 || steps <= 0) return;

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setBindGroup(0, this.bindGroup);
        for (let i = 0; i < steps; i++) {
            pass.setPipeline(this.forcesPipeline);
            pass.dispatchWorkgroups(groups);
            pass.setPipeline(this.integrationPipeline);
            pass.dispatchWorkgroups(groups);
        }
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    /**
     * Copy positions back into the graph. Safe to race with stop(),
     * start() and destroy(): a readback from an older run (or after
     * destroy) is dropped instead of overwriting newer positions.
     */
    private async readbackPositions(generation: number): Promise<void> {
        if (this.mapping || this.destroyed) return;

        const n = this.uploadedNodes;
        const byteSize = n * 2 * 4;
        if (byteSize === 0) return;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.positionsBuffer, 0, this.readbackBuffer, 0, byteSize);
        this.device.queue.submit([encoder.finish()]);

        this.mapping = true;
        let data: Float32Array;
        try {
            await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, byteSize);
            if (this.destroyed) return;
            data = new Float32Array(this.readbackBuffer.getMappedRange(0, byteSize).slice(0));
            this.readbackBuffer.unmap();
        } catch {
            // Buffer destroyed (or device lost) while mapping
            return;
        } finally {
            this.mapping = false;
        }

        if (generation !== this.generation) return;

        // The graph may have changed size while we were waiting
        const count = Math.min(n, this.graph.nodeSlots) * 2;
        this.graph.positions.set(data.subarray(0, count));

        if (this.sharing) {
            // The GPU has the truth and already draws it; this is only
            // the CPU cache. Re-apply edits the GPU hasn't seen yet.
            this.graph.reapplyPositionEdits();
            this.graph.nodesVersion++; // CPU copy changed (e.g. for picking)
            this.target!.positionsChanged();
        } else {
            this.graph.dirtyNodes = true;
            this.graph.dirtyEdges = true;
        }
    }

    async start(): Promise<void> {
        if (this.running || this.destroyed) return;
        this.running = true;
        this.iteration = 0;
        const generation = ++this.generation;
        this.beginSharing();
        this.uploadData();
        this.uploadParams();
        this.onStart?.();

        const loop = () => {
            if (!this.running || generation !== this.generation) return;
            // The renderer may have reallocated the shared buffer since
            this.refreshSharedBuffer();
            this.syncWithGraph();
            this.dispatchSteps(this.config.stepsPerFrame);
            this.iteration += this.config.stepsPerFrame;
            this.onTick?.(this.iteration);
            if (this.sharing) this.target!.positionsChanged();

            // Never wait on the GPU: refresh the CPU copy in the
            // background (skipped while the previous one is in flight)
            void this.readbackPositions(generation);

            if (this.iteration < this.config.maxIterations) {
                this.animationId = requestAnimationFrame(loop);
            } else {
                this.finish();
            }
        };
        loop();
    }

    stop(): void {
        if (!this.running) return;
        this.finish();
    }

    /** Stop stepping; in shared mode, sync the CPU copy before handing positions back */
    private finish(): void {
        const wasRunning = this.running;
        this.running = false;
        const generation = ++this.generation;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
        if (wasRunning) this.onStop?.();
        if (!this.sharing) return;
        void this.finalSync(generation);
    }

    private async finalSync(generation: number): Promise<void> {
        // Wait out an in-flight readback (its result is stale now)
        while (this.mapping && !this.destroyed) {
            await new Promise<void>(r => setTimeout(r, 0));
        }
        if (generation !== this.generation) return; // restarted meanwhile
        this.refreshSharedBuffer();
        await this.readbackPositions(generation);
        if (generation === this.generation) this.endSharing();
    }

    async step(n: number): Promise<void> {
        if (this.destroyed) return;
        this.uploadData();
        this.uploadParams();
        this.dispatchSteps(n);
        await this.readbackPositions(this.generation);
    }

    isRunning(): boolean {
        return this.running;
    }

    destroy(): void {
        if (this.running) this.onStop?.();
        this.running = false;
        this.generation++;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
        this.destroyed = true;
        this.endSharing();
        if (this.ownsPositions) this.positionsBuffer?.destroy();
        this.velocitiesBuffer?.destroy();
        this.forcesBuffer?.destroy();
        this.neighborsBuffer?.destroy();
        this.offsetsBuffer?.destroy();
        this.sizesBuffer?.destroy();
        this.paramsBuffer?.destroy();
        this.readbackBuffer?.destroy();
    }
}

// -----------------------------------------------------------
// Adjacency
// -----------------------------------------------------------

/**
 * Per-node neighbor lists in CSR form over the first `n` node slots:
 * node i's neighbors are neighbors[offsets[i] .. offsets[i + 1]].
 * Removed edges, self-loops and edges outside `n` are left out.
 */
export function buildAdjacency(
    graph: Graph,
    n: number,
    e: number,
): { offsets: Uint32Array<ArrayBuffer>; neighbors: Uint32Array<ArrayBuffer> } {
    const idx = graph.edgeIndices;
    const offsets = new Uint32Array(n + 1);

    const usable = (i: number) => {
        if (!graph.isEdgeActive(i)) return false;
        const a = idx[i * 2], b = idx[i * 2 + 1];
        return a !== b && a < n && b < n;
    };

    for (let i = 0; i < e; i++) {
        if (!usable(i)) continue;
        offsets[idx[i * 2] + 1]++;
        offsets[idx[i * 2 + 1] + 1]++;
    }
    for (let i = 0; i < n; i++) offsets[i + 1] += offsets[i];

    const neighbors = new Uint32Array(offsets[n]);
    const fill = offsets.slice(0, n) as Uint32Array<ArrayBuffer>;
    for (let i = 0; i < e; i++) {
        if (!usable(i)) continue;
        const a = idx[i * 2], b = idx[i * 2 + 1];
        neighbors[fill[a]++] = b;
        neighbors[fill[b]++] = a;
    }
    return { offsets, neighbors };
}
