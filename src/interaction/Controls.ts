// ============================================================
// graphGPU - Interaction Controller
// ============================================================
// Handles pointer input (mouse, touch, pen) for pan, zoom,
// node/edge picking, dragging, and selection. Node picking goes
// through a spatial grid (rebuilt only when nodes change); edges
// are only picked when someone listens to edge events.
//
// Uses Pointer Events with pointer capture, so a drag keeps
// tracking (and ends properly) even when released outside
// the canvas.
// ============================================================

import type { NodeId, EdgeId, InteractionOptions, GraphGPUEventType, GraphGPUEventHandler } from '../types';
import type { Graph } from '../core/Graph';
import { Camera } from '../core/Camera';

/** A layout that can hold dragged nodes in place */
export interface PinnableLayout {
    pin(nodeId: number): void;
    unpin(nodeId: number): void;
}
import { NodeGrid } from './SpatialIndex';

interface DragState {
    active: boolean;
    pointerId: number;
    nodeId: NodeId | null;
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
    isPan: boolean;
    isBox: boolean;   // box selection (interaction.lasso + shift-drag)
    moved: boolean;
}

/** Movement (in canvas pixels) before a press counts as a drag */
const DRAG_THRESHOLD = 3;

/** Zoom speed per wheel pixel: one ~100px mouse notch ≈ 1.16× */
const WHEEL_ZOOM_SPEED = 0.0015;

/** Trackpad pinch (wheel + ctrlKey) sends much smaller deltas */
const PINCH_WHEEL_BOOST = 8;

/** Per-event zoom factor limits, against huge jumps */
const MAX_WHEEL_FACTOR = 2;

export class InteractionController {
    private canvas: HTMLCanvasElement;
    private graph: Graph;
    private camera: Camera;
    private opts: Required<InteractionOptions>;
    private nodeScale: number;

    // State
    private drag: DragState = {
        active: false,
        pointerId: -1,
        nodeId: null,
        startX: 0, startY: 0,
        lastX: 0, lastY: 0,
        isPan: false,
        isBox: false,
        moved: false,
    };

    /** Box selection rectangle (a DOM overlay while dragging) */
    private boxElement: HTMLDivElement | null = null;

    private hoveredNode: NodeId | null = null;
    private hoveredEdge: EdgeId | null = null;

    // Picking grid (rebuilt when nodes or node scale change)
    private grid = new NodeGrid();
    private gridVersion = -1;
    private gridScale = -1;
    private selectedNodes: Set<NodeId> = new Set();
    private selectedEdges: Set<EdgeId> = new Set();

    /** Active pointers (canvas coords), for pinch-zoom */
    private pointers = new Map<number, { x: number; y: number }>();
    private lastPinchDist: number = 0;

    // Event listeners
    private listeners = new Map<GraphGPUEventType, Set<GraphGPUEventHandler>>();

    // Bound handlers (for cleanup)
    private boundPointerDown: (e: PointerEvent) => void;
    private boundPointerMove: (e: PointerEvent) => void;
    private boundPointerUp: (e: PointerEvent) => void;
    private boundPointerCancel: (e: PointerEvent) => void;
    private boundPointerLeave: (e: PointerEvent) => void;
    private boundWheel: (e: WheelEvent) => void;
    private boundDblClick: (e: MouseEvent) => void;

    constructor(
        canvas: HTMLCanvasElement,
        graph: Graph,
        camera: Camera,
        opts?: InteractionOptions,
        nodeScale?: number,
    ) {
        this.canvas = canvas;
        this.graph = graph;
        this.camera = camera;
        this.nodeScale = nodeScale ?? 6;

        this.opts = {
            pan: opts?.pan ?? true,
            zoom: opts?.zoom ?? true,
            zoomRange: opts?.zoomRange ?? [0.01, 100],
            dragNodes: opts?.dragNodes ?? true,
            hover: opts?.hover ?? true,
            selection: opts?.selection ?? true,
            multiSelect: opts?.multiSelect ?? true,
            lasso: opts?.lasso ?? false,
        };

        // Bind handlers
        this.boundPointerDown = this.onPointerDown.bind(this);
        this.boundPointerMove = this.onPointerMove.bind(this);
        this.boundPointerUp = this.onPointerUp.bind(this);
        this.boundPointerCancel = this.onPointerCancel.bind(this);
        this.boundPointerLeave = this.onPointerLeave.bind(this);
        this.boundWheel = this.onWheel.bind(this);
        this.boundDblClick = this.onDblClick.bind(this);

        this.attach();
    }

    /** Attach event listeners to canvas */
    attach(): void {
        this.canvas.addEventListener('pointerdown', this.boundPointerDown);
        this.canvas.addEventListener('pointermove', this.boundPointerMove);
        this.canvas.addEventListener('pointerup', this.boundPointerUp);
        this.canvas.addEventListener('pointercancel', this.boundPointerCancel);
        this.canvas.addEventListener('pointerleave', this.boundPointerLeave);
        this.canvas.addEventListener('wheel', this.boundWheel, { passive: false });
        this.canvas.addEventListener('dblclick', this.boundDblClick);

        this.canvas.style.cursor = 'grab';
        // Prevent browser's default touch behaviors (scroll, zoom)
        this.canvas.style.touchAction = 'none';
    }

    /** Detach event listeners */
    detach(): void {
        this.canvas.removeEventListener('pointerdown', this.boundPointerDown);
        this.canvas.removeEventListener('pointermove', this.boundPointerMove);
        this.canvas.removeEventListener('pointerup', this.boundPointerUp);
        this.canvas.removeEventListener('pointercancel', this.boundPointerCancel);
        this.canvas.removeEventListener('pointerleave', this.boundPointerLeave);
        this.canvas.removeEventListener('wheel', this.boundWheel);
        this.canvas.removeEventListener('dblclick', this.boundDblClick);
    }

    /** Register an event handler */
    on<T = unknown>(type: GraphGPUEventType, handler: GraphGPUEventHandler<T>): void {
        if (!this.listeners.has(type)) {
            this.listeners.set(type, new Set());
        }
        this.listeners.get(type)!.add(handler as GraphGPUEventHandler);
    }

    /** Remove an event handler */
    off<T = unknown>(type: GraphGPUEventType, handler: GraphGPUEventHandler<T>): void {
        this.listeners.get(type)?.delete(handler as GraphGPUEventHandler);
    }

    /** Get currently selected nodes */
    getSelectedNodes(): Set<NodeId> {
        return new Set(this.selectedNodes);
    }

    /** Get currently selected edges */
    getSelectedEdges(): Set<EdgeId> {
        return new Set(this.selectedEdges);
    }

    /**
     * Select nodes, replacing the selection (nodes and edges) or adding
     * to it. Fires node:select / node:deselect as clicks do.
     */
    selectNodes(ids: Iterable<NodeId>, additive: boolean = false): void {
        const wanted = new Set<NodeId>();
        for (const id of ids) if (this.graph.isNodeActive(id)) wanted.add(id);
        if (!additive) {
            this.deselectEdges();
            for (const id of [...this.selectedNodes]) {
                if (wanted.has(id)) continue;
                this.selectedNodes.delete(id);
                this.emit('node:deselect', this.graph.getNode(id));
            }
        }
        for (const id of wanted) {
            if (this.selectedNodes.has(id)) continue;
            this.selectedNodes.add(id);
            this.emit('node:select', this.graph.getNode(id));
        }
    }

    /** Select edges, replacing the selection or adding to it */
    selectEdges(ids: Iterable<EdgeId>, additive: boolean = false): void {
        const wanted = new Set<EdgeId>();
        for (const id of ids) if (this.graph.isEdgeActive(id)) wanted.add(id);
        if (!additive) {
            this.deselectNodes([...this.selectedNodes]);
            for (const id of [...this.selectedEdges]) {
                if (wanted.has(id)) continue;
                this.selectedEdges.delete(id);
                this.emit('edge:deselect', this.graph.getEdge(id) ?? { id });
            }
        }
        for (const id of wanted) {
            if (this.selectedEdges.has(id)) continue;
            this.selectedEdges.add(id);
            this.emit('edge:select', this.graph.getEdge(id));
        }
    }

    /** Remove nodes from the selection */
    deselectNodes(ids: Iterable<NodeId>): void {
        for (const id of ids) {
            if (this.selectedNodes.delete(id)) this.emit('node:deselect', this.graph.getNode(id));
        }
    }

    /** Remove edges from the selection */
    deselectEdgeIds(ids: Iterable<EdgeId>): void {
        for (const id of ids) {
            if (this.selectedEdges.delete(id)) this.emit('edge:deselect', this.graph.getEdge(id) ?? { id });
        }
    }

    /** Deselect everything */
    clearSelection(): void {
        this.deselectNodes([...this.selectedNodes]);
        this.deselectEdges();
    }

    /**
     * Drop nodes/edges that are about to be removed from the graph:
     * deselect them (with events), stop hovering or dragging them.
     * Call before removing, while they can still be looked up.
     */
    forget(nodes: Iterable<NodeId>, edges: Iterable<EdgeId>): void {
        for (const id of nodes) {
            if (this.drag.active && this.drag.nodeId === id) this.cancelDrag();
            if (this.hoveredNode === id) this.updateHover(null);
            if (this.selectedNodes.delete(id)) this.emit('node:deselect', this.graph.getNode(id));
        }
        for (const id of edges) {
            if (this.hoveredEdge === id) this.updateEdgeHover(null);
            if (this.selectedEdges.delete(id)) this.emit('edge:deselect', this.graph.getEdge(id) ?? { id });
        }
    }

    /** forget() everything (the graph is being cleared) */
    forgetAll(): void {
        const nodes = [...this.selectedNodes];
        if (this.hoveredNode !== null) nodes.push(this.hoveredNode);
        if (this.drag.nodeId !== null) nodes.push(this.drag.nodeId);
        const edges = [...this.selectedEdges];
        if (this.hoveredEdge !== null) edges.push(this.hoveredEdge);
        this.forget(nodes, edges);
    }

    /** Toggle selection on/off */
    setSelectionEnabled(enabled: boolean): void {
        this.opts.selection = enabled;
        if (!enabled) {
            this.selectedNodes.clear();
            this.selectedEdges.clear();
        }
    }

    // Multi-select always active (without needing shift)
    private alwaysMultiSelect: boolean = false;

    // Animated mode: layout ref for pin/unpin during drag
    private animatedMode: boolean = false;
    private animatedLayout: PinnableLayout | null = null;

    /** Toggle multi-select always-on mode */
    setMultiSelectEnabled(enabled: boolean): void {
        this.alwaysMultiSelect = enabled;
    }

    /** Set animated mode + provide layout reference for pin/unpin */
    setAnimatedMode(enabled: boolean, layout: PinnableLayout | null): void {
        this.animatedMode = enabled;
        this.animatedLayout = layout;
    }

    private edgeWidth = 0; // CSS px, 0 = scale with nodes

    /** Fixed edge width in CSS pixels (must match the renderer's, for edge picking) */
    setEdgeWidth(width: number): void {
        this.edgeWidth = Math.max(0, width);
    }

    /** Update node scale (must match the renderer's, for hit testing) */
    setNodeScale(scale: number): void {
        this.nodeScale = scale;
    }

    /** Get currently hovered node */
    getHoveredNode(): NodeId | null {
        return this.hoveredNode;
    }

    // =========================================================
    // Hit testing
    // =========================================================

    /** Find the node at screen position (closest one if several) */
    hitTest(screenX: number, screenY: number): NodeId | null {
        // World-space hit radius matching shader: size * nodeScale * 0.01, +20% forgiveness
        const radiusScale = this.nodeScale * 0.01 * 1.2;
        if (this.gridVersion !== this.graph.nodesVersion || this.gridScale !== radiusScale) {
            this.grid.build(this.graph, radiusScale);
            this.gridVersion = this.graph.nodesVersion;
            this.gridScale = radiusScale;
        }
        const [wx, wy] = this.camera.screenToWorld(screenX, screenY);
        return this.grid.query(wx, wy);
    }

    /**
     * Find the edge at screen position (closest one within its drawn
     * width plus a few pixels), or null.
     */
    hitTestEdge(screenX: number, screenY: number): EdgeId | null {
        const [wx, wy] = this.camera.screenToWorld(screenX, screenY);
        const [, wy1] = this.camera.screenToWorld(screenX, screenY + 1);
        const worldPerPx = Math.abs(wy1 - wy);

        // Same width as the edge shader (half-width ≥ 1.25px, or the fixed
        // edgeWidth in CSS px), + 3px slack
        const rect = this.canvas.getBoundingClientRect();
        const pxRatio = rect.width > 0 ? this.canvas.width / rect.width : 1;
        const halfWidth = this.edgeWidth > 0
            ? this.edgeWidth * 0.5 * pxRatio * worldPerPx
            : Math.max(1.25 * worldPerPx, this.nodeScale * 0.01);
        const tolerance = halfWidth + 3 * worldPerPx;

        const graph = this.graph;
        const pos = graph.positions;
        const idx = graph.edgeIndices;
        let best: EdgeId | null = null;
        let bestDist = tolerance;

        for (let e = 0; e < graph.edgeSlots; e++) {
            if (graph.getEdgeTag(e) === undefined) continue;
            const a = idx[e * 2], b = idx[e * 2 + 1];
            if (a === b || !graph.isNodeActive(a) || !graph.isNodeActive(b)) continue;

            const ax = pos[a * 2], ay = pos[a * 2 + 1];
            const bx = pos[b * 2], by = pos[b * 2 + 1];
            // Quick reject on the segment's bounding box
            if (wx < Math.min(ax, bx) - tolerance || wx > Math.max(ax, bx) + tolerance ||
                wy < Math.min(ay, by) - tolerance || wy > Math.max(ay, by) + tolerance) continue;

            const dx = bx - ax, dy = by - ay;
            const lenSq = dx * dx + dy * dy;
            const t = lenSq > 0 ? Math.max(0, Math.min(1, ((wx - ax) * dx + (wy - ay) * dy) / lenSq)) : 0;
            const dist = Math.hypot(wx - (ax + t * dx), wy - (ay + t * dy));
            if (dist < bestDist) {
                bestDist = dist;
                best = e;
            }
        }
        return best;
    }

    private listens(type: GraphGPUEventType): boolean {
        return (this.listeners.get(type)?.size ?? 0) > 0;
    }

    /** Edge picking on click: for selection, or edge click listeners */
    private wantsEdgeClicks(): boolean {
        return this.opts.selection || this.listens('edge:click');
    }

    /** Edge picking on hover only runs when someone listens to it */
    private wantsEdgeHover(): boolean {
        return this.listens('edge:hover') || this.listens('edge:hoverout');
    }

    // =========================================================
    // Pointer handlers
    // =========================================================

    /**
     * Event position in canvas pixels. The scale comes from the canvas
     * itself (backing size / CSS size), so it always matches what the
     * renderer draws, whatever pixel ratio was configured.
     */
    private getCanvasCoords(e: MouseEvent): [number, number] {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = rect.width > 0 ? this.canvas.width / rect.width : 1;
        const scaleY = rect.height > 0 ? this.canvas.height / rect.height : 1;
        return [
            (e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top) * scaleY,
        ];
    }

    private onPointerDown(e: PointerEvent): void {
        // Only the primary (left) mouse button; touch & pen report 0 too
        if (e.button !== 0) return;
        e.preventDefault();

        const [sx, sy] = this.getCanvasCoords(e);
        this.pointers.set(e.pointerId, { x: sx, y: sy });

        // Keep receiving this pointer's events outside the canvas
        try {
            this.canvas.setPointerCapture(e.pointerId);
        } catch {
            // Synthetic events have no active pointer to capture
        }

        if (this.pointers.size === 2) {
            // Second finger: switch to pinch-zoom, cancel any drag
            this.lastPinchDist = this.pinchDistance();
            this.cancelDrag();
            return;
        }
        if (this.pointers.size > 2) return;

        const hitNode = this.opts.dragNodes ? this.hitTest(sx, sy) : null;

        this.drag.active = true;
        this.drag.pointerId = e.pointerId;
        this.drag.startX = sx;
        this.drag.startY = sy;
        this.drag.lastX = sx;
        this.drag.lastY = sy;
        this.drag.moved = false;

        if (hitNode !== null) {
            this.drag.nodeId = hitNode;
            this.drag.isPan = false;
            // In animated mode, pin the dragged node so physics doesn't fight the cursor
            if (this.animatedMode && this.animatedLayout) {
                this.animatedLayout.pin(hitNode);
            }
            // Don't change cursor yet - keep pointer until movement starts
        } else if (this.opts.lasso && this.opts.selection && e.shiftKey) {
            this.drag.nodeId = null;
            this.drag.isPan = false;
            this.drag.isBox = true;
        } else {
            this.drag.nodeId = null;
            this.drag.isPan = true;
        }
    }

    private onPointerMove(e: PointerEvent): void {
        const [sx, sy] = this.getCanvasCoords(e);

        if (this.pointers.has(e.pointerId)) {
            this.pointers.set(e.pointerId, { x: sx, y: sy });
        }

        // Pinch-zoom
        if (this.pointers.size === 2) {
            if (this.opts.zoom) this.onPinch();
            return;
        }

        if (this.drag.active && e.pointerId === this.drag.pointerId) {
            this.onDrag(sx, sy);
        } else if (!this.drag.active && this.opts.hover && e.pointerType !== 'touch') {
            const node = this.hitTest(sx, sy);
            this.updateHover(node);
            // Edges under nodes don't count
            this.updateEdgeHover(node === null && this.wantsEdgeHover() ? this.hitTestEdge(sx, sy) : null);
        }
    }

    private onPointerUp(e: PointerEvent): void {
        this.releasePointer(e);

        if (!this.drag.active || e.pointerId !== this.drag.pointerId) return;

        if (this.drag.isBox && this.drag.moved) {
            this.selectInBox(e);
            this.resetDrag();
            return;
        }

        const isClick = !this.drag.moved;

        if (this.drag.nodeId !== null) {
            const id = this.drag.nodeId;
            this.unpinDragged();
            if (this.drag.moved) {
                this.emit('node:dragend', this.graph.getNode(id), e);
            }

            // Selection happens on release (click without drag)
            if (isClick && this.opts.selection) {
                const isMulti = this.alwaysMultiSelect || e.shiftKey;

                if (isMulti) {
                    if (this.selectedNodes.has(id)) {
                        this.selectedNodes.delete(id);
                        this.emit('node:deselect', this.graph.getNode(id), e);
                    } else {
                        this.selectedNodes.add(id);
                        this.emit('node:select', this.graph.getNode(id), e);
                    }
                } else {
                    // Deselect all others (edges too)
                    this.deselectEdges(e);
                    for (const prev of this.selectedNodes) {
                        if (prev !== id) this.emit('node:deselect', this.graph.getNode(prev), e);
                    }
                    this.selectedNodes.clear();
                    this.selectedNodes.add(id);
                    this.emit('node:select', this.graph.getNode(id), e);
                }
            }

            if (isClick) this.emit('node:click', this.graph.getNode(id), e);
        } else if (isClick) {
            // Clicked away from nodes - deselect all
            if (this.opts.selection) {
                for (const id of this.selectedNodes) {
                    this.emit('node:deselect', this.graph.getNode(id), e);
                }
                this.selectedNodes.clear();
            }
            const edge = this.wantsEdgeClicks() ? this.hitTestEdge(this.drag.startX, this.drag.startY) : null;
            if (edge !== null && this.opts.selection) this.selectEdge(edge, this.alwaysMultiSelect || e.shiftKey, e);
            else if (this.opts.selection) this.deselectEdges(e);
            if (edge !== null) this.emit('edge:click', this.graph.getEdge(edge), e);
            else this.emit('canvas:click', null, e);
        }

        this.resetDrag();
    }

    private onPointerCancel(e: PointerEvent): void {
        this.releasePointer(e);
        if (this.drag.active && e.pointerId === this.drag.pointerId) {
            this.cancelDrag();
        }
    }

    private onPointerLeave(e: PointerEvent): void {
        // Captured pointers keep reporting; only clear plain hover
        if (!this.drag.active && e.pointerType !== 'touch') {
            this.updateHover(null);
            this.updateEdgeHover(null);
        }
    }

    // =========================================================
    // Gestures
    // =========================================================

    private onDrag(sx: number, sy: number): void {
        const dx = sx - this.drag.lastX;
        const dy = sy - this.drag.lastY;

        if (!this.drag.moved &&
            (Math.abs(sx - this.drag.startX) > DRAG_THRESHOLD ||
             Math.abs(sy - this.drag.startY) > DRAG_THRESHOLD)) {
            this.drag.moved = true;
            if (this.drag.nodeId !== null) {
                this.emit('node:dragstart', this.graph.getNode(this.drag.nodeId));
            }
        }

        if (this.drag.isBox) {
            if (this.drag.moved) this.updateBox(sx, sy);
        } else if (this.drag.isPan && this.opts.pan) {
            if (this.drag.moved) this.canvas.style.cursor = 'grabbing';
            this.camera.pan(dx, dy);
            this.emit('canvas:pan', this.camera.getState());
        } else if (this.drag.nodeId !== null && this.opts.dragNodes && this.drag.moved) {
            this.canvas.style.cursor = 'grabbing';
            // Move node in world space
            const [wx1, wy1] = this.camera.screenToWorld(this.drag.lastX, this.drag.lastY);
            const [wx2, wy2] = this.camera.screenToWorld(sx, sy);
            const id = this.drag.nodeId;
            const newX = this.graph.positions[id * 2] + (wx2 - wx1);
            const newY = this.graph.positions[id * 2 + 1] + (wy2 - wy1);
            this.graph.setNodePosition(id, newX, newY);

            this.emit('node:drag', this.graph.getNode(id));
        }

        this.drag.lastX = sx;
        this.drag.lastY = sy;
    }

    private onPinch(): void {
        const dist = this.pinchDistance();
        if (this.lastPinchDist > 0) {
            const [a, b] = [...this.pointers.values()];
            // Zoom by the change in finger distance, toward their midpoint
            this.camera.zoomBy(dist / this.lastPinchDist, (a.x + b.x) / 2, (a.y + b.y) / 2);
            this.emit('canvas:zoom', this.camera.getState());
        }
        this.lastPinchDist = dist;
    }

    private pinchDistance(): number {
        const [a, b] = [...this.pointers.values()];
        return Math.hypot(b.x - a.x, b.y - a.y);
    }

    private updateHover(hitNode: NodeId | null): void {
        if (hitNode === this.hoveredNode) return;

        if (this.hoveredNode !== null) {
            this.emit('node:hoverout', this.graph.getNode(this.hoveredNode));
        }
        this.hoveredNode = hitNode;
        if (hitNode !== null) {
            this.emit('node:hover', this.graph.getNode(hitNode));
            this.canvas.style.cursor = 'pointer';
        } else {
            this.canvas.style.cursor = 'grab';
        }
    }

    /** Select an edge: alone, or toggled into the selection (multi) */
    private selectEdge(edge: EdgeId, multi: boolean, e: Event): void {
        if (multi) {
            if (this.selectedEdges.delete(edge)) {
                this.emit('edge:deselect', this.graph.getEdge(edge), e);
            } else {
                this.selectedEdges.add(edge);
                this.emit('edge:select', this.graph.getEdge(edge), e);
            }
            return;
        }
        for (const other of [...this.selectedEdges]) {
            if (other !== edge) {
                this.selectedEdges.delete(other);
                this.emit('edge:deselect', this.graph.getEdge(other) ?? { id: other }, e);
            }
        }
        if (!this.selectedEdges.has(edge)) {
            this.selectedEdges.add(edge);
            this.emit('edge:select', this.graph.getEdge(edge), e);
        }
    }

    private deselectEdges(e?: Event): void {
        for (const edge of [...this.selectedEdges]) {
            this.selectedEdges.delete(edge);
            this.emit('edge:deselect', this.graph.getEdge(edge) ?? { id: edge }, e);
        }
    }

    private updateEdgeHover(edge: EdgeId | null): void {
        if (edge === this.hoveredEdge) return;
        if (this.hoveredEdge !== null) {
            this.emit('edge:hoverout', this.graph.getEdge(this.hoveredEdge));
        }
        this.hoveredEdge = edge;
        if (edge !== null) {
            this.emit('edge:hover', this.graph.getEdge(edge));
            this.canvas.style.cursor = 'pointer';
        } else if (this.hoveredNode === null) {
            this.canvas.style.cursor = 'grab';
        }
    }

    private releasePointer(e: PointerEvent): void {
        this.pointers.delete(e.pointerId);
        if (this.pointers.size < 2) this.lastPinchDist = 0;
        try {
            if (this.canvas.hasPointerCapture(e.pointerId)) {
                this.canvas.releasePointerCapture(e.pointerId);
            }
        } catch {
            // Already released
        }
    }

    /** Abort a drag without click/select semantics */
    private cancelDrag(): void {
        if (!this.drag.active) return;
        if (this.drag.nodeId !== null) {
            const id = this.drag.nodeId;
            this.unpinDragged();
            if (this.drag.moved) this.emit('node:dragend', this.graph.getNode(id));
        }
        this.resetDrag();
    }

    // =========================================================
    // Box selection
    // =========================================================

    /** Canvas pixels per CSS pixel */
    private pixelScale(): number {
        const rect = this.canvas.getBoundingClientRect();
        return rect.width > 0 ? this.canvas.width / rect.width : 1;
    }

    private updateBox(sx: number, sy: number): void {
        if (!this.boxElement) {
            const box = document.createElement('div');
            box.style.cssText =
                'position:absolute;pointer-events:none;z-index:3;box-sizing:border-box;' +
                'border:1px solid rgba(100,140,255,0.9);background:rgba(100,140,255,0.15);';
            (this.canvas.parentElement ?? document.body).appendChild(box);
            this.boxElement = box;
        }
        const k = this.pixelScale();
        const box = this.boxElement.style;
        box.left = `${this.canvas.offsetLeft + Math.min(this.drag.startX, sx) / k}px`;
        box.top = `${this.canvas.offsetTop + Math.min(this.drag.startY, sy) / k}px`;
        box.width = `${Math.abs(sx - this.drag.startX) / k}px`;
        box.height = `${Math.abs(sy - this.drag.startY) / k}px`;
    }

    private removeBox(): void {
        this.boxElement?.remove();
        this.boxElement = null;
    }

    /** Add every node whose center is inside the box to the selection */
    private selectInBox(e: PointerEvent): void {
        const [ax, ay] = this.camera.screenToWorld(this.drag.startX, this.drag.startY);
        const [bx, by] = this.camera.screenToWorld(this.drag.lastX, this.drag.lastY);
        const minX = Math.min(ax, bx), maxX = Math.max(ax, bx);
        const minY = Math.min(ay, by), maxY = Math.max(ay, by);
        const pos = this.graph.positions;
        for (let id = 0; id < this.graph.nodeSlots; id++) {
            if (!this.graph.isNodeActive(id) || this.selectedNodes.has(id)) continue;
            const x = pos[id * 2], y = pos[id * 2 + 1];
            if (x < minX || x > maxX || y < minY || y > maxY) continue;
            this.selectedNodes.add(id);
            this.emit('node:select', this.graph.getNode(id), e);
        }
    }

    private unpinDragged(): void {
        // In animated mode, unpin the node so it rejoins physics
        if (this.drag.nodeId !== null && this.animatedMode && this.animatedLayout) {
            this.animatedLayout.unpin(this.drag.nodeId);
        }
    }

    private resetDrag(): void {
        this.removeBox();
        this.drag.isBox = false;
        this.drag.active = false;
        this.drag.pointerId = -1;
        this.drag.nodeId = null;
        this.canvas.style.cursor = this.hoveredNode !== null ? 'pointer' : 'grab';
    }

    // =========================================================
    // Wheel & double click
    // =========================================================

    private onWheel(e: WheelEvent): void {
        if (!this.opts.zoom) return;
        e.preventDefault();

        // Normalize to pixels (some mice report lines or pages)
        let dy = e.deltaY;
        if (e.deltaMode === 1) dy *= 16;
        else if (e.deltaMode === 2) dy *= this.canvas.clientHeight || 800;
        if (e.ctrlKey) dy *= PINCH_WHEEL_BOOST;

        // Proportional to the scroll amount, so trackpads (many tiny
        // events) and mouse wheels (few big ones) zoom at similar speeds
        const factor = Math.min(
            MAX_WHEEL_FACTOR,
            Math.max(1 / MAX_WHEEL_FACTOR, Math.exp(-dy * WHEEL_ZOOM_SPEED)),
        );

        const [sx, sy] = this.getCanvasCoords(e);
        this.camera.zoomBy(factor, sx, sy);
        this.emit('canvas:zoom', this.camera.getState());
    }

    private onDblClick(e: MouseEvent): void {
        const [sx, sy] = this.getCanvasCoords(e);
        const hitNode = this.hitTest(sx, sy);
        if (hitNode !== null) {
            this.emit('node:dblclick', this.graph.getNode(hitNode), e);
        }
    }

    // =========================================================
    // Event emission
    // =========================================================

    /** Dispatch an event to its listeners (also used for layout/render events) */
    emit(type: GraphGPUEventType, target: unknown, originalEvent?: Event): void {
        const handlers = this.listeners.get(type);
        if (!handlers) return;
        const event = { type, target, originalEvent };
        for (const handler of handlers) {
            handler(event);
        }
    }

    /** Clean up */
    destroy(): void {
        this.detach();
        this.listeners.clear();
    }
}
