// ============================================================
// graphGPU - WebGPU Renderer
// ============================================================
// Handles all GPU resource management, pipeline creation,
// and the render loop. Uploads only dirty buffers.
// ============================================================

import type { GraphGPUOptions, RenderStats } from '../types';
import type { Graph } from './Graph';
import { Camera } from './Camera';
import { NODE_SHADER, EDGE_SHADER } from '../shaders';
import { parseColor } from '../utils/color';

// -----------------------------------------------------------
// Uniform buffer layouts (std140 aligned)
// -----------------------------------------------------------

// NodeUniforms: camera (3×vec4) + viewport (vec2) + time (f32) + nodeScale (f32)
const NODE_UNIFORM_SIZE = 4 * (12 + 2 + 1 + 1); // 64 bytes

// EdgeUniforms: camera (3×vec4) + viewport (vec2) + edgeOpacity (f32) + nodeScale (f32)
const EDGE_UNIFORM_SIZE = 4 * (12 + 2 + 1 + 1); // 64 bytes

export class Renderer {
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
    private gpuEdgeVerts!: GPUBuffer;
    private edgeUniformBuffer!: GPUBuffer;
    private edgeBindGroup!: GPUBindGroup;

    // CPU edge vertex staging
    private edgeVertexData: Float32Array = new Float32Array(0);

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

        const edgeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' },
            }],
        });

        this.edgePipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [edgeBindGroupLayout],
            }),
            vertex: {
                module: edgeModule,
                entryPoint: 'vs_edge',
                buffers: [{
                    // Per-instance: srcPos(vec2f) + tgtPos(vec2f) + alpha(f32) + color(vec3f)
                    arrayStride: 32,  // 8 floats × 4 bytes
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },   // srcPos
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },   // tgtPos
                        { shaderLocation: 2, offset: 16, format: 'float32' },    // alpha
                        { shaderLocation: 3, offset: 20, format: 'float32x3' },  // color RGB
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

        this.edgeBindGroup = this.device.createBindGroup({
            layout: edgeBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.edgeUniformBuffer } }],
        });
    }

    // =========================================================
    // Buffer management
    // =========================================================

    private allocateBuffers(): void {
        this.reallocNodeBuffers(this.graph.numNodes || 256);
        this.reallocEdgeBuffers(this.graph.numEdges || 512);
    }

    private reallocNodeBuffers(capacity: number): void {
        const cap = Math.max(capacity, 256);
        this.gpuPositions?.destroy();
        this.gpuColors?.destroy();
        this.gpuSizes?.destroy();
        this.gpuSelection?.destroy();

        this.gpuPositions = this.device.createBuffer({
            size: cap * 2 * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
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
    }

    private reallocEdgeBuffers(capacity: number): void {
        const cap = Math.max(capacity, 512);
        this.gpuEdgeVerts?.destroy();
        this.gpuEdgeVerts = this.device.createBuffer({
            size: cap * 8 * 4, // 8 floats per edge instance (srcXY, tgtXY, alpha, RGB)
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.gpuEdgeCapacity = cap;
        this.edgeVertexData = new Float32Array(cap * 8);
    }

    private uploadNodeData(): void {
        const n = this.graph.numNodes;
        if (n > this.gpuNodeCapacity) {
            this.reallocNodeBuffers(n * 2);
        }

        // Grow selection buffer if needed
        if (this.selectionData.length < n) {
            const old = this.selectionData;
            this.selectionData = new Float32Array(n);
            this.selectionData.set(old);
        }

        this.device.queue.writeBuffer(
            this.gpuPositions, 0,
            this.graph.positions.buffer, 0, n * 2 * 4,
        );
        this.device.queue.writeBuffer(
            this.gpuColors, 0,
            this.graph.colors.buffer, 0, n * 4 * 4,
        );
        this.device.queue.writeBuffer(
            this.gpuSizes, 0,
            this.graph.sizes.buffer, 0, n * 4,
        );
        this.device.queue.writeBuffer(
            this.gpuSelection, 0,
            this.selectionData.buffer, 0, n * 4,
        );

        this.graph.dirtyNodes = false;
        this.selectionDirty = false;
    }

    // Visible edge count (updated during upload)
    private lastVisibleEdges: number = 0;

    private uploadEdgeData(): void {
        const edgeCount = this.graph.numEdges;
        if (edgeCount > this.gpuEdgeCapacity) {
            this.reallocEdgeBuffers(edgeCount * 2);
        }

        const positions = this.graph.positions;
        const indices = this.graph.edgeIndices;
        const edgeColors = this.graph.edgeColors;
        let writeIdx = 0;
        let visibleEdges = 0;

        for (let i = 0; i < edgeCount; i++) {
            const src = indices[i * 2];
            const tgt = indices[i * 2 + 1];
            if (!this.graph.isNodeActive(src) || !this.graph.isNodeActive(tgt)) continue;
            if (!this.graph.isEdgeActive(i)) continue;

            // Per-instance: srcX, srcY, tgtX, tgtY, alpha, R, G, B
            this.edgeVertexData[writeIdx++] = positions[src * 2];
            this.edgeVertexData[writeIdx++] = positions[src * 2 + 1];
            this.edgeVertexData[writeIdx++] = positions[tgt * 2];
            this.edgeVertexData[writeIdx++] = positions[tgt * 2 + 1];
            this.edgeVertexData[writeIdx++] = 1.0;
            this.edgeVertexData[writeIdx++] = edgeColors[i * 3];
            this.edgeVertexData[writeIdx++] = edgeColors[i * 3 + 1];
            this.edgeVertexData[writeIdx++] = edgeColors[i * 3 + 2];
            visibleEdges++;
        }

        this.lastVisibleEdges = visibleEdges;

        if (visibleEdges > 0) {
            this.device.queue.writeBuffer(
                this.gpuEdgeVerts, 0,
                this.edgeVertexData.buffer, 0, visibleEdges * 8 * 4,
            );
        }

        this.graph.dirtyEdges = false;
    }

    // =========================================================
    // Render frame
    // =========================================================

    /** Render a single frame */
    renderFrame(timestamp: number): RenderStats {
        if (!this.initialized) {
            return this.emptyStats();
        }

        const t = timestamp / 1000;

        // Update camera matrix
        this.camera.update();

        // Upload dirty data
        if (this.graph.dirtyNodes || this.selectionDirty) this.uploadNodeData();
        if (this.graph.dirtyEdges) this.uploadEdgeData();

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
        const visibleEdges = this.lastVisibleEdges;
        if (visibleEdges > 0) {
            pass.setPipeline(this.edgePipeline);
            pass.setBindGroup(0, this.edgeBindGroup);
            pass.setVertexBuffer(0, this.gpuEdgeVerts);
            pass.draw(6, visibleEdges);
        }

        // Draw nodes
        const nodeCount = this.graph.numNodes;
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
        this.renderOverlay();

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
            nodeCount,
            edgeCount: this.graph.numEdges,
            visibleNodes: nodeCount,
            visibleEdges,
        };
    }

    /** Start the render loop */
    start(): void {
        const loop = (ts: number) => {
            this.renderFrame(ts);
            this.animationId = requestAnimationFrame(loop);
        };
        this.animationId = requestAnimationFrame(loop);
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
        this.backgroundColor = color;
    }

    /** Change node scale at runtime */
    setNodeScale(scale: number): void {
        this.nodeScale = scale;
    }

    /** Change edge opacity at runtime */
    setEdgeOpacity(opacity: number): void {
        this.edgeOpacity = opacity;
    }

    /** Set selection state for a node (1.0 = selected, 0.0 = not) */
    setSelection(nodeId: number, selected: boolean): void {
        if (this.selectionData.length <= nodeId) {
            const old = this.selectionData;
            this.selectionData = new Float32Array(Math.max(nodeId + 1, this.graph.numNodes));
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

    /**
     * Convert world position to CSS-pixel screen position.
     * This duplicates the Camera math but returns CSS pixels directly.
     */
    private worldToCSS(wx: number, wy: number): [number, number] {
        const m = this.camera.matrix;
        // matrix is col-major std140: col0=[m0,m1,_,_] col1=[m4,m5,_,_] col2=[m8,m9,_,_]
        const ndcX = wx * m[0] + wy * m[4] + m[8];
        const ndcY = wx * m[1] + wy * m[5] + m[9];
        // NDC [-1,1] → CSS pixels [0, clientWidth]
        const cw = this.canvas.clientWidth;
        const ch = this.canvas.clientHeight;
        const sx = (ndcX + 1) * 0.5 * cw;
        const sy = (-ndcY + 1) * 0.5 * ch;
        return [sx, sy];
    }

    /** Render labels on Canvas2D overlay */
    private renderOverlay(): void {
        const pr = this.pixelRatio;
        const cw = this.canvas.clientWidth;
        const ch = this.canvas.clientHeight;
        const tw = Math.round(cw * pr);
        const th = Math.round(ch * pr);

        // ---- Labels (in front of WebGPU canvas) ----
        if (this.labelCtx && this.labelCanvas) {
            if (this.labelCanvas.width !== tw || this.labelCanvas.height !== th) {
                this.labelCanvas.width = tw;
                this.labelCanvas.height = th;
            }
            const lctx = this.labelCtx;
            lctx.clearRect(0, 0, tw, th);
            lctx.save();
            lctx.scale(pr, pr);
            lctx.textAlign = 'center';
            lctx.textBaseline = 'middle';

            const positions = this.graph.positions;
            const sizes = this.graph.sizes;

            const camZoomGlobal = Math.abs(this.camera.matrix[0]);

            // ---- Pre-compute node screen positions & radii for overlap culling ----
            const nodeScreenData: { sx: number; sy: number; r: number }[] = [];
            for (const id of this.graph.activeNodeIds()) {
                const wx = positions[id * 2];
                const wy = positions[id * 2 + 1];
                const [sx, sy] = this.worldToCSS(wx, wy);
                const nodeWorldSize = sizes[id] * this.nodeScale * 0.01;
                const nodeScreenR = nodeWorldSize * camZoomGlobal * cw * 0.5;
                nodeScreenData[id] = { sx, sy, r: nodeScreenR };
            }

            // ---- Edge labels (drawn FIRST so node labels paint over them) ----
            const baseNodeWorldSize = this.nodeScale * 0.01;
            const projectedNodeR = baseNodeWorldSize * camZoomGlobal * cw * 0.5;
            const edgeWidthPx = Math.max(2.5, projectedNodeR * 1.0);

            // Only show edge labels when node labels are visible.
            // Node labels appear when nodeScreenR >= 16. Check any active node.
            let anyNodeLabelVisible = false;
            for (const id of this.graph.activeNodeIds()) {
                if (nodeScreenData[id] && nodeScreenData[id].r >= 16) {
                    anyNodeLabelVisible = true;
                    break;
                }
            }

            if (anyNodeLabelVisible) {
                lctx.textAlign = 'center';
                lctx.textBaseline = 'middle';

                const edgeIndices = this.graph.edgeIndices;
                const pos = this.graph.positions;

                for (const eid of this.graph.activeEdgeIds()) {
                    const src = edgeIndices[eid * 2];
                    const tgt = edgeIndices[eid * 2 + 1];
                    if (!this.graph.isNodeActive(src) || !this.graph.isNodeActive(tgt)) continue;

                    const edge = this.graph.getEdge(eid);
                    if (!edge || !edge.tag) continue;

                    // Use the average screen radius of src and tgt nodes.
                    // This is the CORRECT nodeScreenR (includes sizes[id] * nodeScale).
                    const srcR = nodeScreenData[src]?.r ?? 0;
                    const tgtR = nodeScreenData[tgt]?.r ?? 0;
                    const avgNodeR = (srcR + tgtR) * 0.5;

                    // Edge font = 65% of what the node label would be WITHOUT the 22px cap.
                    // No cap on edge labels — they scale with zoom just like everything else.
                    const edgeFontSize = Math.max(7, avgNodeR * 0.28 * 0.65);

                    lctx.font = `500 ${edgeFontSize}px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif`;
                    lctx.fillStyle = 'rgba(160,155,175,0.85)';

                    const labelOffset = edgeWidthPx * 0.5 + edgeFontSize * 0.55 + 3;

                    // Midpoint in world coords
                    const mx = (pos[src * 2] + pos[tgt * 2]) * 0.5;
                    const my = (pos[src * 2 + 1] + pos[tgt * 2 + 1]) * 0.5;
                    const [sx, sy] = this.worldToCSS(mx, my);

                    if (sx < -80 || sx > cw + 80 || sy < -40 || sy > ch + 40) continue;

                    // Compute edge angle for rotated text
                    const [sx1, sy1] = this.worldToCSS(pos[src * 2], pos[src * 2 + 1]);
                    const [sx2, sy2] = this.worldToCSS(pos[tgt * 2], pos[tgt * 2 + 1]);
                    let angle = Math.atan2(sy2 - sy1, sx2 - sx1);

                    if (angle > Math.PI / 2) angle -= Math.PI;
                    if (angle < -Math.PI / 2) angle += Math.PI;

                    const edgeLen = Math.hypot(sx2 - sx1, sy2 - sy1);
                    if (edgeLen < 40) continue;

                    // Compute the actual label position (offset perpendicular to edge)
                    const perpX = -Math.sin(angle) * labelOffset;
                    const perpY = Math.cos(angle) * labelOffset;
                    const labelX = sx + perpX;
                    const labelY = sy - Math.abs(perpY);

                    // Skip if label center overlaps any node circle
                    let overlapsNode = false;
                    for (const nid of this.graph.activeNodeIds()) {
                        const nd = nodeScreenData[nid];
                        if (!nd) continue;
                        const dx = labelX - nd.sx;
                        const dy = labelY - nd.sy;
                        if (dx * dx + dy * dy < nd.r * nd.r) {
                            overlapsNode = true;
                            break;
                        }
                    }
                    if (overlapsNode) continue;

                    lctx.save();
                    lctx.translate(sx, sy);
                    lctx.rotate(angle);
                    lctx.fillText(edge.tag, 0, -labelOffset);
                    lctx.restore();
                }
            }

            // ---- Node labels (drawn AFTER edge labels so they paint on top) ----
            for (const id of this.graph.activeNodeIds()) {
                const nd = nodeScreenData[id];
                if (!nd) continue;
                const { sx, sy, r: nodeScreenR } = nd;

                if (sx < -60 || sx > cw + 60 || sy < -60 || sy > ch + 60) continue;
                if (nodeScreenR < 16) continue;

                const node = this.graph.getNode(id);
                if (!node) continue;
                const label = (node.properties.name ?? node.properties.title ?? node.tag) as string;
                if (!label) continue;

                const fontSize = Math.max(7, Math.min(nodeScreenR * 0.28, 22));
                lctx.font = `600 ${fontSize}px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif`;

                const maxW = nodeScreenR * 1.6;
                let disp = label;
                if (lctx.measureText(disp).width > maxW) {
                    let lo = 1, hi = label.length;
                    while (lo < hi) {
                        const mid = (lo + hi + 1) >> 1;
                        if (lctx.measureText(label.slice(0, mid) + '…').width <= maxW) lo = mid; else hi = mid - 1;
                    }
                    disp = label.slice(0, lo) + '…';
                }
                if (lctx.measureText(disp).width > maxW) continue;

                const r = this.graph.colors[id * 4];
                const g = this.graph.colors[id * 4 + 1];
                const b = this.graph.colors[id * 4 + 2];
                const lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
                lctx.fillStyle = lum > 0.189 ? 'rgba(0,0,0,0.88)' : 'rgba(255,255,255,0.93)';

                lctx.fillText(disp, sx, sy);
            }

            lctx.restore();
        }
    }

    /** Clean up all GPU resources */
    destroy(): void {
        this.stop();
        this.gpuPositions?.destroy();
        this.gpuColors?.destroy();
        this.gpuSizes?.destroy();
        this.gpuSelection?.destroy();
        this.gpuEdgeVerts?.destroy();
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