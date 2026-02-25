// ============================================================
// graphGPU - Main Entry Point
// ============================================================

import type {
    GraphGPUOptions, NodeId, EdgeId,
    Node, Edge,
    GraphGPUEventType, GraphGPUEventHandler,
    CameraState, Palette,
} from './types';
import { Graph } from './core/Graph';
import { Renderer } from './core/Renderer';
import { Camera } from './core/Camera';
import { InteractionController } from './interaction/Controls';
import { ForceLayout } from './layout/ForceLayout';
import { GPUForceLayout } from './layout/GPUForceLayout';
import { PALETTES } from './utils/palette';

export class GraphGPU {
    private graph: Graph;
    private renderer: Renderer;
    private camera: Camera;
    private controls: InteractionController;
    private layout: ForceLayout | null = null;
    private gpuLayout: GPUForceLayout | null = null;

    private canvas: HTMLCanvasElement;
    private options: GraphGPUOptions;

    constructor(options: GraphGPUOptions) {
        this.options = options;

        // Resolve canvas
        if (typeof options.canvas === 'string') {
            this.canvas = document.querySelector(options.canvas) as HTMLCanvasElement;
        } else {
            this.canvas = options.canvas;
        }

        if (!this.canvas) {
            throw new Error('graphGPU: canvas element not found');
        }

        // Create graph data structure
        this.graph = new Graph({
            maxNodes: options.maxNodes,
            maxEdges: options.maxEdges,
            nodeSize: options.nodeSize,
            palette: typeof options.palette === 'string'
                ? options.palette
                : options.palette?.name,
        });

        // Create renderer
        this.renderer = new Renderer(this.graph, options);

        // Camera (temporary - renderer creates the real one)
        this.camera = this.renderer.getCamera();

        // Controls
        this.controls = new InteractionController(
            this.canvas,
            this.graph,
            this.camera,
            options.interaction,
            options.nodeSize,
        );
    }

    // =========================================================
    // Initialization
    // =========================================================

    /** Initialize WebGPU and start rendering. Must be awaited. */
    async init(): Promise<boolean> {
        const ok = await this.renderer.init();
        if (!ok) return false;

        this.camera = this.renderer.getCamera();

        // Initialize label overlay
        this.renderer.initLabelOverlay();

        // Wire selection → renderer
        this.controls.on('node:select', (e) => {
            const node = e.target as Node | null;
            if (node) this.renderer.setSelection(node.id, true);
        });
        this.controls.on('node:deselect', (e) => {
            const node = e.target as Node | null;
            if (node) this.renderer.setSelection(node.id, false);
        });
        this.controls.on('canvas:click', () => {
            this.renderer.clearSelection();
        });

        // Start render loop
        this.renderer.start();

        return true;
    }

    // =========================================================
    // Graph operations (graphGPU DSL-inspired)
    // =========================================================

    /**
     * Add a node to the graph.
     * Inspired by Grafito's `put` command.
     *
     * Supports two calling conventions:
     *   g.put('person', 'Alice', { role: 'engineer' })  // tag, label, properties
     *   g.put('person', { name: 'Alice', role: 'engineer' })  // tag, properties
     *
     * @example
     *   const alice = g.put('person', 'Alice', { role: 'engineer' });
     *   const bob   = g.put('person', { name: 'Bob', role: 'designer' });
     */
    put(tag: string, labelOrProps?: string | Record<string, unknown>, propsOrOpts?: Record<string, unknown> | {
        x?: number; y?: number; size?: number; color?: string;
    }, opts?: {
        x?: number; y?: number; size?: number; color?: string;
    }): NodeId {
        let properties: Record<string, unknown> | undefined;
        let renderOpts: { x?: number; y?: number; size?: number; color?: string } | undefined;

        if (typeof labelOrProps === 'string') {
            // put(tag, label, properties?, opts?)
            properties = (propsOrOpts as Record<string, unknown>) ?? {};
            properties = { name: labelOrProps, ...properties };
            renderOpts = opts;
        } else {
            // put(tag, properties?, opts?)
            properties = labelOrProps;
            renderOpts = propsOrOpts as { x?: number; y?: number; size?: number; color?: string } | undefined;
        }

        return this.graph.addNode({
            tag,
            properties,
            x: renderOpts?.x,
            y: renderOpts?.y,
            size: renderOpts?.size,
            color: renderOpts?.color,
        });
    }

    /**
     * Batch-add nodes.
     *
     * @example
     *   const ids = g.putMany('person', [
     *     { name: 'John' },
     *     { name: 'Mary' },
     *   ]);
     */
    putMany(tag: string, items: Record<string, unknown>[]): NodeId[] {
        return this.graph.addNodes(
            items.map(props => ({ tag, properties: props })),
        );
    }

    /**
     * Remove a node and all its edges.
     * Inspired by graphGPU's `unput` command.
     */
    unput(id: NodeId): void {
        this.graph.removeNode(id);
    }

    /**
     * Create an edge between two nodes.
     * Inspired by graphGPU's `link` / `~>` operator.
     *
     * @example
     *   g.link(john, 'marriedTo', mary);
     */
    link(
        source: NodeId | NodeId[],
        tag: string,
        target: NodeId | NodeId[],
        properties?: Record<string, unknown>,
    ): EdgeId[] {
        const sources = Array.isArray(source) ? source : [source];
        const targets = Array.isArray(target) ? target : [target];
        const ids: EdgeId[] = [];

        for (const src of sources) {
            for (const tgt of targets) {
                ids.push(this.graph.addEdge({
                    tag,
                    source: src,
                    target: tgt,
                    properties,
                }));
            }
        }

        return ids;
    }

    /**
     * Remove edge(s) between nodes.
     * Inspired by graphGPU's `unlink` command.
     */
    unlink(edgeId: EdgeId): void {
        this.graph.removeEdge(edgeId);
    }

    /**
     * Fetch nodes by tag, optionally filtering properties.
     * Inspired by graphGPU's `fetch` command.
     *
     * @example
     *   const people = g.fetch('person', { surname: 'Doe' });
     */
    fetch(tag: string, filter?: Record<string, unknown>): Node[] {
        return this.graph.fetchByTag(tag, filter);
    }

    /**
     * Get a single node by ID.
     * Inspired by graphGPU's `what` command.
     */
    what(id: NodeId): Node | null {
        return this.graph.getNode(id);
    }

    /**
     * Get related nodes (neighbors) for a given node.
     * Inspired by graphGPU's `getRelatedNodes`.
     */
    related(id: NodeId): Node[] {
        return this.graph.getRelatedNodes(id);
    }

    /**
     * Get all edges for a node.
     */
    edges(nodeId: NodeId): Edge[] {
        return this.graph.getNodeEdges(nodeId);
    }

    /**
     * Get all unique tags in the graph.
     */
    tags(): string[] {
        return this.graph.getAllTags();
    }

    /**
     * Clear the entire graph.
     */
    clear(): void {
        this.graph.clear();
    }

    /**
     * Get node/edge counts.
     */
    get nodeCount(): number { return this.graph.numNodes; }
    get edgeCount(): number { return this.graph.numEdges; }

    // =========================================================
    // Layout
    // =========================================================

    /**
     * Start force-directed layout.
     */
    startLayout(opts?: {
        gravitationalConstant?: number;
        springLength?: number;
        springConstant?: number;
        centralGravity?: number;
        damping?: number;
        timestep?: number;
        maxVelocity?: number;
        minVelocity?: number;
        maxIterations?: number;
        // Legacy compat
        repulsion?: number;
        attraction?: number;
        gravity?: number;
    }): void {
        this.layout = new ForceLayout(this.graph, {
            type: 'force',
            ...opts,
        });
        this.layout.start();
    }

    /**
     * Stop the current layout.
     */
    stopLayout(): void {
        this.layout?.stop();
    }

    /**
     * Run layout for N steps synchronously.
     */
    stepLayout(steps: number = 1): number {
        if (!this.layout) {
            this.layout = new ForceLayout(this.graph, {
                type: 'force',
                ...this.options.layout,
            });
        }
        return this.layout.step(steps);
    }

    /**
     * Start GPU-accelerated force layout (requires WebGPU).
     * Falls back to CPU layout if GPU compute unavailable.
     */
    async startGPULayout(opts?: {
        repulsion?: number;
        attraction?: number;
        gravity?: number;
        damping?: number;
        stepsPerFrame?: number;
        maxIterations?: number;
    }): Promise<void> {
        // Stop any running layout
        this.stopLayout();
        this.stopGPULayout();

        const device = this.renderer.getDevice();
        if (!device) {
            // Fallback to CPU
            console.warn('graphGPU: GPU compute not available, falling back to CPU layout');
            this.startLayout(opts);
            return;
        }

        this.gpuLayout = new GPUForceLayout(device, this.graph, opts);
        await this.gpuLayout.start();
    }

    /**
     * Stop the GPU layout.
     */
    stopGPULayout(): void {
        this.gpuLayout?.stop();
        this.gpuLayout?.destroy();
        this.gpuLayout = null;
    }

    // =========================================================
    // Camera
    // =========================================================

    /**
     * Fit the view to show all nodes.
     */
    fitView(padding: number = 0.1): void {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const id of this.graph.activeNodeIds()) {
            const x = this.graph.positions[id * 2];
            const y = this.graph.positions[id * 2 + 1];
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        if (minX === Infinity) return; // no nodes
        this.camera.fitBounds(minX, minY, maxX, maxY, padding);
    }

    /**
     * Handle container/window resize. Updates canvas dimensions,
     * MSAA texture, and camera aspect ratio.
     * Call this from your resize handler instead of getGraph().
     */
    resize(): void {
        this.renderer.resizeCanvas();
    }

    /**
     * Set camera position and zoom.
     */
    setView(x: number, y: number, zoom?: number): void {
        this.camera.setPosition(x, y);
        if (zoom !== undefined) this.camera.setZoom(zoom);
    }

    /**
     * Reset camera to default.
     */
    resetView(): void {
        this.camera.reset();
    }

    /**
     * Re-randomize all node positions and restart layout.
     * Use this for a full "reset to initial state" effect.
     */
    resetPositions(): void {
        this.graph.resetPositions();
    }

    /**
     * Get current camera state.
     */
    getView(): CameraState {
        return this.camera.getState();
    }

    // =========================================================
    // Palette
    // =========================================================

    /**
     * Switch color palette.
     * Inspired by graphGPU's `.palette` attribute.
     */
    setPalette(palette: string | Palette): void {
        this.graph.tagColors.setPalette(palette);
        // Re-color all existing nodes
        for (const id of this.graph.activeNodeIds()) {
            const tag = this.graph.getNodeTag(id);
            if (tag) {
                const color = this.graph.tagColors.getColor(tag);
                this.graph.setNodeColor(id, color.bg);
            }
        }
        // Re-color all existing edges
        for (const id of this.graph.activeEdgeIds()) {
            const edge = this.graph.getEdge(id);
            if (edge?.tag) {
                const color = this.graph.tagColors.getColor(edge.tag);
                this.graph.edgeColors[id * 3] = color.bg[0];
                this.graph.edgeColors[id * 3 + 1] = color.bg[1];
                this.graph.edgeColors[id * 3 + 2] = color.bg[2];
            }
        }
        this.graph.dirtyEdges = true;
    }

    /**
     * Get available built-in palettes.
     */
    static get palettes(): Record<string, Palette> {
        return PALETTES;
    }

    /**
     * Get current tag→color assignments (useful for building legends).
     */
    getTagColors(): Map<string, { bg: [number, number, number, number] }> {
        const result = new Map<string, { bg: [number, number, number, number] }>();
        for (const [tag, assignment] of this.graph.tagColors.getAll()) {
            result.set(tag, { bg: assignment.bg });
        }
        return result;
    }

    // =========================================================
    // Appearance
    // =========================================================

    /**
     * Change background color at runtime.
     */
    setBackground(color: [number, number, number, number]): void {
        this.renderer.setBackground(color);
    }

    // =========================================================
    // Selection
    // =========================================================

    /**
     * Enable or disable node selection.
     */
    setSelectionEnabled(enabled: boolean): void {
        this.controls.setSelectionEnabled(enabled);
        if (!enabled) this.renderer.clearSelection();
    }

    /**
     * Enable or disable multi-select (shift+click or always-on).
     */
    setMultiSelectEnabled(enabled: boolean): void {
        this.controls.setMultiSelectEnabled(enabled);
    }

    /**
     * Enable animated mode: live physics simulation that never auto-stops.
     * Dragging a node pins it while the rest of the graph reacts with
     * spring-like elasticity (vis.js-style "wobbly" physics).
     */
    setAnimated(enabled: boolean): void {
        if (!this.layout) {
            // Create layout if not yet started
            this.layout = new ForceLayout(this.graph, {
                type: 'force',
                ...this.options.layout,
            });
        }
        this.layout.setAnimated(enabled);
        this.controls.setAnimatedMode(enabled, this.layout);
    }

    /**
     * Pin a node: its position is fixed, physics won't move it.
     */
    pinNode(nodeId: number): void {
        this.layout?.pin(nodeId);
    }

    /**
     * Unpin a node: rejoin the physics simulation.
     */
    unpinNode(nodeId: number): void {
        this.layout?.unpin(nodeId);
    }

    // =========================================================
    // Events
    // =========================================================

    /**
     * Listen for graph/interaction events.
     *
     * @example
     *   g.on('node:click', (e) => console.log(e.target));
     */
    on<T = unknown>(type: GraphGPUEventType, handler: GraphGPUEventHandler<T>): void {
        this.controls.on(type, handler);
    }

    /**
     * Remove an event listener.
     */
    off<T = unknown>(type: GraphGPUEventType, handler: GraphGPUEventHandler<T>): void {
        this.controls.off(type, handler);
    }

    // =========================================================
    // Direct buffer access (for power users)
    // =========================================================

    /**
     * Get raw position buffer. Modify and call markDirty().
     */
    get positions(): Float32Array {
        return this.graph.positions;
    }

    /**
     * Get raw color buffer.
     */
    get colors(): Float32Array {
        return this.graph.colors;
    }

    /**
     * Mark node data as dirty (triggers GPU re-upload).
     */
    markDirty(): void {
        this.graph.dirtyNodes = true;
        this.graph.dirtyEdges = true;
    }

    /**
     * Access the underlying Graph object.
     */
    getGraph(): Graph {
        return this.graph;
    }

    // =========================================================
    // Lifecycle
    // =========================================================

    /**
     * Pause rendering.
     */
    pause(): void {
        this.renderer.stop();
    }

    /**
     * Resume rendering.
     */
    resume(): void {
        this.renderer.start();
    }

    /**
     * Destroy everything - GPU resources, event listeners, etc.
     */
    destroy(): void {
        this.layout?.stop();
        this.gpuLayout?.stop();
        this.gpuLayout?.destroy();
        this.gpuLayout = null;
        this.controls.destroy();
        this.renderer.destroy();
    }
}

// =========================================================
// Re-exports
// =========================================================

export { Graph } from './core/Graph';
export { Camera } from './core/Camera';
export { Renderer } from './core/Renderer';
export { ForceLayout } from './layout/ForceLayout';
export { InteractionController } from './interaction/Controls';
export { TagColorMap, PALETTES } from './utils/palette';
export { parseColor, idealForeground, darken, lighten } from './utils/color';
export * from './types';
