// ============================================================
// graphGPU - WebGPU Renderer
// ============================================================
// Handles all GPU resource management, pipeline creation,
// and the render loop. Uploads only dirty buffers.
// ============================================================

import type { GraphGPUOptions, RenderStats } from '../types';
import type { Graph } from './Graph';
import { Camera } from './Camera';
import type { PositionTarget } from '../layout/GPUForceLayout';
import { NODE_SHADER, EDGE_SHADER } from '../shaders';
import { parseColor } from '../utils/color';

// -----------------------------------------------------------
// Uniform buffer layouts (std140 aligned)
// -----------------------------------------------------------

// NodeUniforms: camera (3×vec4) + viewport (vec2) + time (f32) + nodeScale (f32)
const NODE_UNIFORM_SIZE = 4 * (12 + 2 + 1 + 1); // 64 bytes

// EdgeUniforms: camera (3×vec4) + viewport (vec2) + edgeOpacity (f32) + nodeScale (f32)
const EDGE_UNIFORM_SIZE = 4 * (12 + 2 + 1 + 1); // 64 bytes

export class Renderer implements PositionTarget {
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;
    private msaaTexture: GPUTexture | null = null;

    // Pipelines
    private nodePipeline!: GPURenderPipeline;
    private edgePipeline!: GPURenderPipeline;

    // GPU Buffers - Nodes
    private gpuPositions!: GPUBuffer;
    private gpuColors!: GPUBuffer;
    private gpuSizes!: GPUBuffer;
    private gpuSelection!: GPUBuffer;
    private nodeUniformBuffer!: GPUBuffer;
    private nodeBindGroup!: GPUBindGroup;

    // GPU Buffers - Edges
    private gpuEdges!: GPUBuffer;           // per edge: src u32, tgt u32, rgba8 color
    private edgeBindGroupLayout!: GPUBindGroupLayout;
    private edgeUniformBuffer!: GPUBuffer;
    private edgeBindGroup!: GPUBindGroup;

    // CPU edge staging: [src, tgt, packedColor] per edge slot
    private edgeData: Uint32Array = new Uint32Array(0);

    // Edges currently on the GPU (re-uploaded only when edges change)
    private uploadedStructure: number = -1;
    private uploadedEdgeColors: number = -1;
    private drawnEdges: number = 0;        // instances drawn (edge slots)
    private visibleEdges: number = 0;      // active ones

    // Config
    private canvas: HTMLCanvasElement;
    private camera: Camera;
    private pixelRatio: number;
    private nodeScale: number;
    private edgeOpacity: number;
    private backgroundColor: [number, number, number, number];
    private antialias: boolean;

    // Stats
    private frameCount: number = 0;
    private lastFpsTime: number = 0;
    private currentFps: number = 0;

    // Buffer capacity tracking
    private gpuNodeCapacity: number = 0;
    private gpuEdgeCapacity: number = 0;

    // Selection state (CPU-side, uploaded per frame)
    private selectionData: Float32Array = new Float32Array(0);
    selectionDirty: boolean = false;

    // Label overlay (Canvas2D on top of WebGPU)
    private labelCanvas: HTMLCanvasElement | null = null;
    private labelCtx: CanvasRenderingContext2D | null = null;

    // State
    private initialized: boolean = false;
    private animationId: number = 0;
    private labelsVisible: boolean = true;

    // Positions owned by the GPU layout (shared buffer, no CPU uploads)
    private externalPositions: boolean = false;

    // Render on demand: frames are only drawn when something changed
    private needsRender: boolean = true;
    private renderedFrames: number = 0;
    private onFontsLoaded = () => this.invalidate();

    constructor(
        private graph: Graph,
        private options: GraphGPUOptions,
    ) {
        // Resolve canvas
        if (typeof options.canvas === 'string') {
            this.canvas = document.querySelector(options.canvas) as HTMLCanvasElement;
        } else {
            this.canvas = options.canvas;
        }

        this.pixelRatio = options.pixelRatio ?? window.devicePixelRatio;
        this.nodeScale = options.nodeSize ?? 6;
        this.edgeOpacity = options.edgeOpacity ?? 0.12;
        this.antialias = options.antialias ?? true;

        const bg = options.background
            ? parseColor(options.background)
            : [0.02, 0.02, 0.04, 1.0] as [number, number, number, number];
        this.backgroundColor = bg as [number, number, number, number];

        this.camera = new Camera(
            this.canvas.width,
            this.canvas.height,
            options.interaction?.zoomRange ?? [0.01, 100],
        );
    }

    /** Initialize WebGPU - must be called before rendering */
    async init(): Promise<boolean> {
        if (!navigator.gpu) {
            this.options.onWebGPUUnavailable?.();
            return false;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            this.options.onWebGPUUnavailable?.();
            return false;
        }

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque',
        });

        this.resizeCanvas();
        this.createPipelines();
        this.allocateBuffers();

        this.initialized = true;
        return true;
    }

    /** Get the camera for external manipulation */
    getCamera(): Camera {
        return this.camera;
    }

    /** Resize canvas to match container */
    resizeCanvas(): void {
        this.invalidate();
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * this.pixelRatio;
        this.canvas.height = rect.height * this.pixelRatio;
        this.camera.resize(this.canvas.width, this.canvas.height);

        if (this.antialias && this.device) {
            this.msaaTexture?.destroy();
            this.msaaTexture = this.device.createTexture({
                size: { width: this.canvas.width, height: this.canvas.height },
                sampleCount: 4,
                format: this.format,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }
    }

    // =========================================================
    // Pipeline creation
    // =========================================================

    private createPipelines(): void {
        const blendState: GPUBlendState = {
            color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
            },
            alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
            },
        };

        const sampleCount = this.antialias ? 4 : 1;

        // --- Node pipeline (instanced quads) ---
        const nodeModule = this.device.createShaderModule({ code: NODE_SHADER });

        const nodeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' },
            }],
        });

        this.nodePipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [nodeBindGroupLayout],
            }),
            vertex: {
                module: nodeModule,
                entryPoint: 'vs_node',
                buffers: [
                    {
                        // positions: vec2f per instance
                        arrayStride: 8,
                        stepMode: 'instance',
                        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
                    },
                    {
                        // colors: vec4f per instance
                        arrayStride: 16,
                        stepMode: 'instance',
                        attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }],
                    },
                    {
                        // sizes: f32 per instance
                        arrayStride: 4,
                        stepMode: 'instance',
                        attributes: [{ shaderLocation: 2, offset: 0, format: 'float32' }],
                    },
                    {
                        // selection: f32 per instance (0 or 1)
                        arrayStride: 4,
                        stepMode: 'instance',
                        attributes: [{ shaderLocation: 3, offset: 0, format: 'float32' }],
                    },
                ],
            },
            fragment: {
                module: nodeModule,
                entryPoint: 'fs_node',
                targets: [{ format: this.format, blend: blendState }],
            },
            primitive: { topology: 'triangle-list' },
            multisample: { count: sampleCount },
        });

        // --- Edge pipeline (lines) ---
        const edgeModule = this.device.createShaderModule({ code: EDGE_SHADER });

        this.edgeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' },
                },
                {
                    // Node positions, read directly by the edge vertex shader
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'read-only-storage' },
                },
            ],
        });

        this.edgePipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.edgeBindGroupLayout],
            }),
            vertex: {
                module: edgeModule,
                entryPoint: 'vs_edge',
                buffers: [{
                    // Per-instance: endpoints (2 × u32) + color (rgba8, a = 0 → hidden)
                    arrayStride: 12,
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'uint32x2' },   // src, tgt
                        { shaderLocation: 1, offset: 8, format: 'unorm8x4' },   // color
                    ],
                }],
            },
            fragment: {
                module: edgeModule,
                entryPoint: 'fs_edge',
                targets: [{ format: this.format, blend: blendState }],
            },
            primitive: { topology: 'triangle-list' },
            multisample: { count: sampleCount },
        });

        // --- Uniform buffers ---
        this.nodeUniformBuffer = this.device.createBuffer({
            size: NODE_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.edgeUniformBuffer = this.device.createBuffer({
            size: EDGE_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // --- Bind groups ---
        this.nodeBindGroup = this.device.createBindGroup({
            layout: nodeBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.nodeUniformBuffer } }],
        });
    }

    /** (Re)create the edge bind group: it references the positions buffer */
    private createEdgeBindGroup(): void {
        this.edgeBindGroup = this.device.createBindGroup({
            layout: this.edgeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.edgeUniformBuffer } },
                { binding: 1, resource: { buffer: this.gpuPositions } },
            ],
        });
    }

    // =========================================================
    // Buffer management
    // =========================================================

    private allocateBuffers(): void {
        this.reallocNodeBuffers(this.graph.nodeSlots || 256);
        this.reallocEdgeBuffers(this.graph.edgeSlots || 512);
    }

    private reallocNodeBuffers(capacity: number): void {
        const cap = Math.max(capacity, 256);
        const oldPositions = this.gpuPositions;
        const oldCapacity = this.gpuNodeCapacity;
        this.gpuColors?.destroy();
        this.gpuSizes?.destroy();
        this.gpuSelection?.destroy();

        this.gpuPositions = this.device.createBuffer({
            size: cap * 2 * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.gpuColors = this.device.createBuffer({
            size: cap * 4 * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.gpuSizes = this.device.createBuffer({
            size: cap * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.gpuSelection = this.device.createBuffer({
            size: cap * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.gpuNodeCapacity = cap;
        this.createEdgeBindGroup();

        if (oldPositions) {
            // While the GPU layout owns positions, the GPU copy is the
            // only up-to-date one: carry it over
            if (this.externalPositions && oldCapacity > 0) {
                const encoder = this.device.createCommandEncoder();
                encoder.copyBufferToBuffer(oldPositions, 0, this.gpuPositions, 0, Math.min(oldCapacity, cap) * 8);
                this.device.queue.submit([encoder.finish()]);
            }
            oldPositions.destroy();
        }
    }

    private reallocEdgeBuffers(capacity: number): void {
        const cap = Math.max(capacity, 512);
        this.gpuEdges?.destroy();
        this.gpuEdges = this.device.createBuffer({
            size: cap * 3 * 4, // src, tgt, packed color
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.gpuEdgeCapacity = cap;
        this.edgeData = new Uint32Array(cap * 3);
    }

    /**
     * Upload node data. Moving nodes (layouts, dragging) only marks
     * positions dirty, so usually only 8 bytes per node go up; colors
     * and sizes only when they changed, selection only when it did.
     */
    private uploadNodeData(): void {
        const n = this.graph.nodeSlots;
        let all = this.graph.dirtyNodeAttributes;
        if (n > this.gpuNodeCapacity) {
            this.reallocNodeBuffers(n * 2);
            all = true; // fresh buffers
        }

        // Grow selection buffer if needed
        if (this.selectionData.length < n) {
            const old = this.selectionData;
            this.selectionData = new Float32Array(n);
            this.selectionData.set(old);
        }

        if (this.externalPositions) {
            // The GPU layout writes positions; only send explicit edits
            // (dragged / added / reset nodes)
            this.flushPositionEdits();
        } else if (this.graph.dirtyNodes || all) {
            this.device.queue.writeBuffer(
                this.gpuPositions, 0,
                this.graph.positions.buffer, 0, n * 2 * 4,
            );
        }

        if (all) {
            this.device.queue.writeBuffer(
                this.gpuColors, 0,
                this.graph.colors.buffer, 0, n * 4 * 4,
            );
            this.device.queue.writeBuffer(
                this.gpuSizes, 0,
                this.graph.sizes.buffer, 0, n * 4,
            );
        }

        if (this.selectionDirty || all) {
            this.device.queue.writeBuffer(
                this.gpuSelection, 0,
                this.selectionData.buffer, 0, n * 4,
            );
        }

        this.graph.dirtyNodes = false;
        this.graph.dirtyNodeAttributes = false;
        this.selectionDirty = false;
    }

    /**
     * Upload the edge list. Edges only store endpoint ids + color:
     * the vertex shader reads node positions from the GPU, so this
     * only runs when edges are added, removed or recolored — not
     * when nodes move.
     */
    private uploadEdgeData(): void {
        const edgeCount = this.graph.edgeSlots;
        if (edgeCount > this.gpuEdgeCapacity) {
            this.reallocEdgeBuffers(edgeCount * 2);
        }

        const indices = this.graph.edgeIndices;
        const edgeColors = this.graph.edgeColors;
        const data = this.edgeData;
        let visible = 0;

        for (let i = 0; i < edgeCount; i++) {
            const src = indices[i * 2];
            const tgt = indices[i * 2 + 1];
            const o = i * 3;
            if (!this.graph.isEdgeActive(i) ||
                !this.graph.isNodeActive(src) || !this.graph.isNodeActive(tgt)) {
                // Removed slot: alpha 0 makes the shader skip it
                data[o] = 0;
                data[o + 1] = 0;
                data[o + 2] = 0;
                continue;
            }
            data[o] = src;
            data[o + 1] = tgt;
            data[o + 2] = packColor(edgeColors[i * 3], edgeColors[i * 3 + 1], edgeColors[i * 3 + 2]);
            visible++;
        }

        if (edgeCount > 0) {
            this.device.queue.writeBuffer(this.gpuEdges, 0, data.buffer, 0, edgeCount * 3 * 4);
        }

        this.drawnEdges = edgeCount;
        this.visibleEdges = visible;
        this.uploadedStructure = this.graph.structureVersion;
        this.uploadedEdgeColors = this.graph.edgeColorVersion;
    }

    // =========================================================
    // Render frame
    // =========================================================

    /** Render a single frame */
    renderFrame(timestamp: number): RenderStats {
        if (!this.initialized) {
            return this.emptyStats();
        }

        this.needsRender = false;
        this.renderedFrames++;

        const t = timestamp / 1000;

        // Update camera matrix
        this.camera.update();

        // Upload dirty data. Edges follow node positions on the GPU,
        // so moving nodes only uploads node data.
        if (this.graph.dirtyNodes || this.graph.dirtyNodeAttributes || this.selectionDirty) {
            this.uploadNodeData();
        }
        if (this.uploadedStructure !== this.graph.structureVersion ||
            this.uploadedEdgeColors !== this.graph.edgeColorVersion) {
            this.uploadEdgeData();
        }
        this.graph.dirtyEdges = false;

        // Write uniforms
        const uniforms = new Float32Array(16);

        // Camera matrix (3 × vec4, 12 floats)
        uniforms.set(this.camera.matrix, 0);

        // Viewport
        uniforms[12] = this.canvas.width;
        uniforms[13] = this.canvas.height;

        // Time + nodeScale
        uniforms[14] = t;
        uniforms[15] = this.nodeScale;

        this.device.queue.writeBuffer(this.nodeUniformBuffer, 0, uniforms);

        // Edge uniforms: same camera/viewport, but edgeOpacity + nodeScale in last 2 slots
        uniforms[14] = this.edgeOpacity;
        uniforms[15] = this.nodeScale;
        this.device.queue.writeBuffer(this.edgeUniformBuffer, 0, uniforms);

        // Begin render pass
        const encoder = this.device.createCommandEncoder();

        const colorAttachment: GPURenderPassColorAttachment = this.antialias
            ? {
                    view: this.msaaTexture!.createView(),
                    resolveTarget: this.context.getCurrentTexture().createView(),
                    clearValue: {
                        r: this.backgroundColor[0],
                        g: this.backgroundColor[1],
                        b: this.backgroundColor[2],
                        a: this.backgroundColor[3],
                    },
                    loadOp: 'clear' as const,
                    storeOp: 'discard' as const,
                }
            : {
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: {
                        r: this.backgroundColor[0],
                        g: this.backgroundColor[1],
                        b: this.backgroundColor[2],
                        a: this.backgroundColor[3],
                    },
                    loadOp: 'clear' as const,
                    storeOp: 'store' as const,
                };

        const pass = encoder.beginRenderPass({
            colorAttachments: [colorAttachment],
        });

        // Draw edges first (behind nodes) - 6 verts per quad, instanced
        const visibleEdges = this.visibleEdges;
        if (visibleEdges > 0) {
            pass.setPipeline(this.edgePipeline);
            pass.setBindGroup(0, this.edgeBindGroup);
            pass.setVertexBuffer(0, this.gpuEdges);
            pass.draw(6, this.drawnEdges);
        }

        // Draw nodes
        const nodeCount = this.graph.nodeSlots;
        if (nodeCount > 0) {
            pass.setPipeline(this.nodePipeline);
            pass.setBindGroup(0, this.nodeBindGroup);
            pass.setVertexBuffer(0, this.gpuPositions);
            pass.setVertexBuffer(1, this.gpuColors);
            pass.setVertexBuffer(2, this.gpuSizes);
            pass.setVertexBuffer(3, this.gpuSelection);
            pass.draw(6, nodeCount); // 6 vertices per quad, instanced
        }

        pass.end();
        this.device.queue.submit([encoder.finish()]);

        // Render edges + labels on Canvas2D overlay
        if (this.labelsVisible) {
            this.renderOverlay();
        } else if (this.labelCtx && this.labelCanvas) {
            // Clear overlay when labels are hidden
            this.labelCtx.clearRect(0, 0, this.labelCanvas.width, this.labelCanvas.height);
        }

        // FPS
        this.frameCount++;
        if (timestamp - this.lastFpsTime >= 500) {
            this.currentFps = Math.round(
                this.frameCount / ((timestamp - this.lastFpsTime) / 1000),
            );
            this.frameCount = 0;
            this.lastFpsTime = timestamp;
        }

        return {
            fps: this.currentFps,
            frameTime: 0,
            nodeCount: this.graph.numNodes,
            edgeCount: this.graph.numEdges,
            visibleNodes: this.graph.numNodes,
            visibleEdges,
        };
    }

    /** Start the render loop */
    start(): void {
        if (this.animationId) return; // already running
        const loop = (ts: number) => {
            // Skip the frame entirely (no GPU work, no label pass)
            // unless something changed since the last one
            if (this.isDirty()) this.renderFrame(ts);
            this.animationId = requestAnimationFrame(loop);
        };
        this.invalidate();
        this.animationId = requestAnimationFrame(loop);
    }

    // =========================================================
    // Shared positions (PositionTarget, for the GPU layout)
    // =========================================================

    acquirePositions(slots: number): GPUBuffer {
        if (!this.externalPositions) {
            this.externalPositions = true;
            this.graph.trackPositionEdits(true);
        }
        if (slots > this.gpuNodeCapacity) {
            this.reallocNodeBuffers(slots);
            this.graph.dirtyNodes = true;
            this.graph.dirtyNodeAttributes = true; // new buffers: colors/sizes too
        }
        return this.gpuPositions;
    }

    releasePositions(): void {
        if (!this.externalPositions) return;
        this.flushPositionEdits();
        this.externalPositions = false;
        this.graph.trackPositionEdits(false);
        // CPU positions are in sync again
        this.graph.dirtyNodes = true;
        this.invalidate();
    }

    positionsChanged(): void {
        this.invalidate();
    }

    private flushPositionEdits(): void {
        for (const [id, [x, y]] of this.graph.takePositionEdits()) {
            if (id >= this.gpuNodeCapacity) continue; // covered by the next full sync
            this.device.queue.writeBuffer(this.gpuPositions, id * 8, new Float32Array([x, y]));
        }
    }

    /** Force the next frame to be drawn */
    invalidate(): void {
        this.needsRender = true;
    }

    /** Whether anything changed since the last drawn frame */
    isDirty(): boolean {
        return this.needsRender ||
            this.camera.dirty ||
            this.graph.dirtyNodes ||
            this.graph.dirtyNodeAttributes ||
            this.graph.dirtyEdges ||
            this.selectionDirty;
    }

    /** Number of frames actually drawn so far */
    getRenderedFrames(): number {
        return this.renderedFrames;
    }

    /** Stop the render loop */
    stop(): void {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
    }

    /** Get the GPU device (for compute layout) */
    getDevice(): GPUDevice | null {
        return this.device ?? null;
    }

    /** Change background color at runtime */
    setBackground(color: [number, number, number, number]): void {
        this.invalidate();
        this.backgroundColor = color;
    }

    /** Change node scale at runtime */
    setNodeScale(scale: number): void {
        this.invalidate();
        this.nodeScale = scale;
    }

    /** Change edge opacity at runtime */
    setEdgeOpacity(opacity: number): void {
        this.invalidate();
        this.edgeOpacity = opacity;
    }

    /** Show or hide all labels (node + edge) */
    setLabelsVisible(visible: boolean): void {
        this.invalidate();
        this.labelsVisible = visible;
    }

    /** Set selection state for a node (1.0 = selected, 0.0 = not) */
    setSelection(nodeId: number, selected: boolean): void {
        if (this.selectionData.length <= nodeId) {
            const old = this.selectionData;
            this.selectionData = new Float32Array(Math.max(nodeId + 1, this.graph.nodeSlots));
            this.selectionData.set(old);
        }
        this.selectionData[nodeId] = selected ? 1.0 : 0.0;
        this.selectionDirty = true;
    }

    /** Clear all selections */
    clearSelection(): void {
        this.selectionData.fill(0);
        this.selectionDirty = true;
    }

    /** Initialize overlay canvases */
    initLabelOverlay(): void {
        // Labels drawn before a web font loads must be redrawn after
        document.fonts?.addEventListener?.('loadingdone', this.onFontsLoaded);
        // Label canvas: sits IN FRONT of the WebGPU canvas
        this.labelCanvas = document.createElement('canvas');
        this.labelCanvas.style.cssText =
            'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2;';

        // Append to canvas parent so the overlay is scoped to the graph container,
        // not the whole page. Ensure the parent is a positioning context.
        const parent = this.canvas.parentElement ?? document.body;
        if (parent !== document.body) {
            const pos = getComputedStyle(parent).position;
            if (pos === 'static' || pos === '') {
                parent.style.position = 'relative';
            }
        }
        parent.appendChild(this.labelCanvas);
        this.labelCtx = this.labelCanvas.getContext('2d');
    }

    // =========================================================
    // Label overlay
    // =========================================================
    // Per frame, allocation-free where it matters:
    //   1. project active nodes to screen (typed arrays)
    //   2. collect edge label candidates (culled & measured)
    //   3. bucket node circles near those labels into a grid, so
    //      each label only tests nearby nodes for overlap
    //   4. draw edge labels, then node labels on top
    // =========================================================

    // Reused per-frame buffers
    // (Float64, so label placement matches plain JS number math exactly)
    private labelSX: Float64Array = new Float64Array(0);
    private labelSY: Float64Array = new Float64Array(0);
    private labelR: Float64Array = new Float64Array(0);
    private labelCorners: Float64Array = new Float64Array(0); // 8 per candidate
    private labelInfo: Float64Array = new Float64Array(0);    // sx, sy, angle, offset per candidate
    private labelEdges: Int32Array = new Int32Array(0);
    private labelFonts: string[] = [];

    // Text measurement caches (cleared when they grow too big)
    private textWidths = new Map<string, number>();
    private truncations = new Map<string, string | null>();

    private measure(ctx: CanvasRenderingContext2D, font: string, text: string): number {
        const key = font + '\u0000' + text;
        let w = this.textWidths.get(key);
        if (w === undefined) {
            if (this.textWidths.size > 20000) this.textWidths.clear();
            w = ctx.measureText(text).width;
            this.textWidths.set(key, w);
        }
        return w;
    }

    /** Label as displayed within maxW (ellipsized), or null if it can't fit */
    private fitLabel(ctx: CanvasRenderingContext2D, font: string, label: string, maxW: number): string | null {
        const key = font + '\u0000' + maxW + '\u0000' + label;
        const cached = this.truncations.get(key);
        if (cached !== undefined) return cached;
        if (this.truncations.size > 20000) this.truncations.clear();

        let disp = label;
        if (this.measure(ctx, font, disp) > maxW) {
            let lo = 1, hi = label.length;
            while (lo < hi) {
                const mid = (lo + hi + 1) >> 1;
                if (this.measure(ctx, font, label.slice(0, mid) + '…') <= maxW) lo = mid; else hi = mid - 1;
            }
            disp = label.slice(0, lo) + '…';
        }
        const result = this.measure(ctx, font, disp) > maxW ? null : disp;
        this.truncations.set(key, result);
        return result;
    }

    /** Render labels on Canvas2D overlay */
    private renderOverlay(): void {
        const lctx = this.labelCtx;
        const labelCanvas = this.labelCanvas;
        if (!lctx || !labelCanvas) return;

        const pr = this.pixelRatio;
        const cw = this.canvas.clientWidth;
        const ch = this.canvas.clientHeight;
        const tw = Math.round(cw * pr);
        const th = Math.round(ch * pr);

        if (labelCanvas.width !== tw || labelCanvas.height !== th) {
            labelCanvas.width = tw;
            labelCanvas.height = th;
        }
        lctx.clearRect(0, 0, tw, th);

        const graph = this.graph;
        const slots = graph.nodeSlots;
        const positions = graph.positions;
        const sizes = graph.sizes;
        const m = this.camera.matrix;
        const camZoomGlobal = Math.abs(m[0]);
        const nodeRadius = (size: number) => size * this.nodeScale * 0.01 * camZoomGlobal * cw * 0.5;

        // Labels only appear on nodes at least 16px in radius (and
        // edge labels only alongside them): skip everything otherwise
        let maxSize = 0;
        for (let id = 0; id < slots; id++) {
            if (sizes[id] > maxSize && graph.isNodeActive(id)) maxSize = sizes[id];
        }
        if (nodeRadius(maxSize) < 16) return;

        lctx.save();
        lctx.scale(pr, pr);
        lctx.textAlign = 'center';
        lctx.textBaseline = 'middle';

        // ---- 1. Project nodes (world → CSS pixels) ----
        if (this.labelSX.length < slots) {
            this.labelSX = new Float64Array(slots * 2);
            this.labelSY = new Float64Array(slots * 2);
            this.labelR = new Float64Array(slots * 2);
        }
        const SX = this.labelSX, SY = this.labelSY, R = this.labelR;
        for (let id = 0; id < slots; id++) {
            const wx = positions[id * 2];
            const wy = positions[id * 2 + 1];
            SX[id] = (wx * m[0] + wy * m[4] + m[8] + 1) * 0.5 * cw;
            SY[id] = (-(wx * m[1] + wy * m[5] + m[9]) + 1) * 0.5 * ch;
            R[id] = nodeRadius(sizes[id]);
        }

        // ---- 2. Edge label candidates ----
        const projectedNodeR = this.nodeScale * 0.01 * camZoomGlobal * cw * 0.5;
        const edgeWidthPx = Math.max(2.5, projectedNodeR * 1.0);
        const edgeIndices = graph.edgeIndices;
        const edgeSlots = graph.edgeSlots;
        if (this.labelEdges.length < edgeSlots) {
            this.labelEdges = new Int32Array(edgeSlots * 2);
            this.labelInfo = new Float64Array(edgeSlots * 2 * 4);
            this.labelCorners = new Float64Array(edgeSlots * 2 * 8);
        }
        const cand = this.labelEdges, info = this.labelInfo, corners = this.labelCorners;
        const fonts = this.labelFonts;
        let count = 0;
        let qMinX = Infinity, qMinY = Infinity, qMaxX = -Infinity, qMaxY = -Infinity;

        for (let eid = 0; eid < edgeSlots; eid++) {
            const tag = graph.getEdgeTag(eid);
            if (!tag) continue; // removed, or untagged
            const src = edgeIndices[eid * 2];
            const tgt = edgeIndices[eid * 2 + 1];
            if (!graph.isNodeActive(src) || !graph.isNodeActive(tgt)) continue;

            const sx1 = SX[src], sy1 = SY[src], sx2 = SX[tgt], sy2 = SY[tgt];
            const mx = (positions[src * 2] + positions[tgt * 2]) * 0.5;
            const my = (positions[src * 2 + 1] + positions[tgt * 2 + 1]) * 0.5;
            const sx = (mx * m[0] + my * m[4] + m[8] + 1) * 0.5 * cw;
            const sy = (-(mx * m[1] + my * m[5] + m[9]) + 1) * 0.5 * ch;
            if (sx < -80 || sx > cw + 80 || sy < -40 || sy > ch + 40) continue;
            if (Math.hypot(sx2 - sx1, sy2 - sy1) < 40) continue;

            let angle = Math.atan2(sy2 - sy1, sx2 - sx1);
            if (angle > Math.PI / 2) angle -= Math.PI;
            if (angle < -Math.PI / 2) angle += Math.PI;

            // Edge font = 65% of what the node label would be WITHOUT the 22px cap
            const avgNodeR = (R[src] + R[tgt]) * 0.5;
            const edgeFontSize = Math.max(7, avgNodeR * 0.28 * 0.65);
            const font = `500 ${edgeFontSize}px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif`;
            if (lctx.font !== font) lctx.font = font;

            const labelOffset = edgeWidthPx * 0.5 + edgeFontSize * 0.55 + 3;
            const textW = this.measure(lctx, font, tag) * 0.5 + 4;
            const textH = edgeFontSize * 0.6 + 4;
            const cosA = Math.cos(angle);
            const sinA = Math.sin(angle);
            const offY = -labelOffset;

            // Four corners of the rotated label box
            const c = count * 8;
            corners[c] = sx + cosA * (-textW) - sinA * (offY - textH);
            corners[c + 1] = sy + sinA * (-textW) + cosA * (offY - textH);
            corners[c + 2] = sx + cosA * textW - sinA * (offY - textH);
            corners[c + 3] = sy + sinA * textW + cosA * (offY - textH);
            corners[c + 4] = sx + cosA * textW - sinA * (offY + textH);
            corners[c + 5] = sy + sinA * textW + cosA * (offY + textH);
            corners[c + 6] = sx + cosA * (-textW) - sinA * (offY + textH);
            corners[c + 7] = sy + sinA * (-textW) + cosA * (offY + textH);
            for (let k = 0; k < 8; k += 2) {
                const x = corners[c + k], y = corners[c + k + 1];
                if (x < qMinX) qMinX = x;
                if (x > qMaxX) qMaxX = x;
                if (y < qMinY) qMinY = y;
                if (y > qMaxY) qMaxY = y;
            }

            const o = count * 4;
            info[o] = sx;
            info[o + 1] = sy;
            info[o + 2] = angle;
            info[o + 3] = labelOffset;
            cand[count] = eid;
            fonts[count] = font;
            count++;
        }

        // ---- 3. Drop labels overlapping any node circle ----
        if (count > 0) {
            const grid = this.buildNodeGrid(slots, qMinX, qMinY, qMaxX, qMaxY);

            lctx.fillStyle = 'rgba(160,155,175,0.85)';
            for (let i = 0; i < count; i++) {
                const c = i * 8;
                let overlaps = false;
                for (let k = 0; k < 8 && !overlaps; k += 2) {
                    overlaps = grid.hits(corners[c + k], corners[c + k + 1]);
                }
                if (overlaps) continue;

                // ---- 4a. Draw edge label ----
                const o = i * 4;
                const font = fonts[i];
                if (lctx.font !== font) lctx.font = font;
                lctx.save();
                lctx.translate(info[o], info[o + 1]);
                lctx.rotate(info[o + 2]);
                lctx.fillText(graph.getEdgeTag(cand[i])!, 0, -info[o + 3]);
                lctx.restore();
            }
        }

        // ---- 4b. Node labels (on top of edge labels) ----
        const colors = graph.colors;
        for (let id = 0; id < slots; id++) {
            const nodeScreenR = R[id];
            if (nodeScreenR < 16 || !graph.isNodeActive(id)) continue;
            const sx = SX[id], sy = SY[id];
            if (sx < -60 || sx > cw + 60 || sy < -60 || sy > ch + 60) continue;

            const props = graph.getNodeProperties(id)!;
            const label = (props.name ?? props.title ?? graph.getNodeTag(id)) as string;
            if (!label) continue;

            const fontSize = Math.max(7, Math.min(nodeScreenR * 0.28, 22));
            const font = `600 ${fontSize}px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif`;
            if (lctx.font !== font) lctx.font = font;

            const disp = this.fitLabel(lctx, font, label, nodeScreenR * 1.6);
            if (disp === null) continue;

            const lum = colors[id * 4] * 0.2126 + colors[id * 4 + 1] * 0.7152 + colors[id * 4 + 2] * 0.0722;
            lctx.fillStyle = lum > 0.189 ? 'rgba(0,0,0,0.88)' : 'rgba(255,255,255,0.93)';
            lctx.fillText(disp, sx, sy);
        }

        lctx.restore();
    }

    // Reused grid storage (cell → node list, CSR layout)
    private gridIds: Int32Array = new Int32Array(0);
    private gridStart: Int32Array = new Int32Array(0);
    private gridItems: Int32Array = new Int32Array(0);
    private gridSpan: Int32Array = new Int32Array(4);

    /**
     * Uniform grid of the node circles that intersect the given area
     * (in CSS pixels), for "is this point inside any node?" queries.
     * Cells are sized by both node density and node size, so every
     * node only lands in a few cells.
     */
    private buildNodeGrid(slots: number, minX: number, minY: number, maxX: number, maxY: number) {
        const SX = this.labelSX, SY = this.labelSY, R = this.labelR;
        const graph = this.graph;

        // Nodes whose circle reaches the query area
        if (this.gridIds.length < slots) this.gridIds = new Int32Array(slots * 2);
        const ids = this.gridIds;
        let n = 0;
        let sumR = 0;
        for (let id = 0; id < slots; id++) {
            const r = R[id];
            if (r <= 0) continue;
            if (SX[id] + r < minX || SX[id] - r > maxX || SY[id] + r < minY || SY[id] - r > maxY) continue;
            if (!graph.isNodeActive(id)) continue;
            ids[n++] = id;
            sumR += r;
        }

        const w = Math.max(1, maxX - minX);
        const h = Math.max(1, maxY - minY);
        const byDensity = Math.sqrt((w * h) / Math.max(1, n));
        const byDiameter = n > 0 ? (2 * sumR) / n : 0;
        const cell = Math.min(1024, Math.max(16, byDensity, byDiameter));
        const cols = Math.ceil(w / cell) + 1;
        const rows = Math.ceil(h / cell) + 1;
        const cellCount = cols * rows;

        // Two passes (count, then fill) into flat arrays
        if (this.gridStart.length < cellCount + 1) this.gridStart = new Int32Array((cellCount + 1) * 2);
        const start = this.gridStart;
        start.fill(0, 0, cellCount + 1);

        // Cell range covered by a node: [x0, x1, y0, y1]
        const sp = this.gridSpan;
        const span = (id: number) => {
            const r = R[id];
            sp[0] = Math.max(0, Math.floor((SX[id] - r - minX) / cell));
            sp[1] = Math.min(cols - 1, Math.floor((SX[id] + r - minX) / cell));
            sp[2] = Math.max(0, Math.floor((SY[id] - r - minY) / cell));
            sp[3] = Math.min(rows - 1, Math.floor((SY[id] + r - minY) / cell));
        };

        let total = 0;
        for (let i = 0; i < n; i++) {
            span(ids[i]);
            for (let cy = sp[2]; cy <= sp[3]; cy++) {
                for (let cx = sp[0]; cx <= sp[1]; cx++) start[cy * cols + cx + 1]++;
            }
            total += (sp[1] - sp[0] + 1) * (sp[3] - sp[2] + 1);
        }
        for (let c = 0; c < cellCount; c++) start[c + 1] += start[c];

        if (this.gridItems.length < total) this.gridItems = new Int32Array(total * 2);
        const items = this.gridItems;
        const fill = start.slice(0, cellCount);
        for (let i = 0; i < n; i++) {
            const id = ids[i];
            span(id);
            for (let cy = sp[2]; cy <= sp[3]; cy++) {
                for (let cx = sp[0]; cx <= sp[1]; cx++) items[fill[cy * cols + cx]++] = id;
            }
        }

        return {
            hits: (x: number, y: number): boolean => {
                const cx = Math.floor((x - minX) / cell);
                const cy = Math.floor((y - minY) / cell);
                if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return false;
                const c = cy * cols + cx;
                for (let k = start[c], end = start[c + 1]; k < end; k++) {
                    const id = items[k];
                    const dx = x - SX[id];
                    const dy = y - SY[id];
                    if (dx * dx + dy * dy < R[id] * R[id]) return true;
                }
                return false;
            },
        };
    }

    /** Clean up all GPU resources */
    destroy(): void {
        this.stop();
        if (typeof document !== 'undefined') {
            document.fonts?.removeEventListener?.('loadingdone', this.onFontsLoaded);
        }
        this.gpuPositions?.destroy();
        this.gpuColors?.destroy();
        this.gpuSizes?.destroy();
        this.gpuSelection?.destroy();
        this.gpuEdges?.destroy();
        this.nodeUniformBuffer?.destroy();
        this.edgeUniformBuffer?.destroy();
        this.msaaTexture?.destroy();
        this.labelCanvas?.remove();
        this.device?.destroy();
    }

    private emptyStats(): RenderStats {
        return {
            fps: 0, frameTime: 0,
            nodeCount: 0, edgeCount: 0,
            visibleNodes: 0, visibleEdges: 0,
        };
    }
}

// -----------------------------------------------------------
// Helpers
// -----------------------------------------------------------

/** Pack an RGB color (0-1) into rgba8unorm with alpha = 255 */
function packColor(r: number, g: number, b: number): number {
    const to8 = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
    return (to8(r) | (to8(g) << 8) | (to8(b) << 16) | (255 << 24)) >>> 0;
}
