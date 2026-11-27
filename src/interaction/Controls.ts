// ============================================================
// graphGPU - Interaction Controller
// ============================================================
// Handles pointer input (mouse, touch, pen) for pan, zoom,
// node picking, dragging, and selection. Spatial hit testing
// via brute force (upgrade to quadtree for >50k nodes).
//
// Uses Pointer Events with pointer capture, so a drag keeps
// tracking (and ends properly) even when released outside
// the canvas.
// ============================================================

import type { NodeId, InteractionOptions, GraphGPUEventType, GraphGPUEventHandler } from '../types';
import type { Graph } from '../core/Graph';
import type { ForceLayout } from '../layout/ForceLayout';
import { Camera } from '../core/Camera';

interface DragState {
    active: boolean;
    pointerId: number;
    nodeId: NodeId | null;
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
    isPan: boolean;
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
        moved: false,
    };

    private hoveredNode: NodeId | null = null;
    private selectedNodes: Set<NodeId> = new Set();

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

    /** Toggle selection on/off */
    setSelectionEnabled(enabled: boolean): void {
        this.opts.selection = enabled;
        if (!enabled) this.selectedNodes.clear();
    }

    // Multi-select always active (without needing shift)
    private alwaysMultiSelect: boolean = false;

    // Animated mode: layout ref for pin/unpin during drag
    private animatedMode: boolean = false;
    private animatedLayout: ForceLayout | null = null;

    /** Toggle multi-select always-on mode */
    setMultiSelectEnabled(enabled: boolean): void {
        this.alwaysMultiSelect = enabled;
    }

    /** Set animated mode + provide layout reference for pin/unpin */
    setAnimatedMode(enabled: boolean, layout: ForceLayout | null): void {
        this.animatedMode = enabled;
        this.animatedLayout = layout;
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

    /** Find the node at screen position (brute force) */
    hitTest(screenX: number, screenY: number): NodeId | null {
        const [wx, wy] = this.camera.screenToWorld(screenX, screenY);

        let closestId: NodeId | null = null;
        let closestDist = Infinity;

        const positions = this.graph.positions;
        const sizes = this.graph.sizes;

        for (const id of this.graph.activeNodeIds()) {
            const nx = positions[id * 2];
            const ny = positions[id * 2 + 1];
            const dx = wx - nx;
            const dy = wy - ny;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // World-space hit radius matching shader: size * nodeScale * 0.01
            const hitRadius = sizes[id] * this.nodeScale * 0.01 * 1.2; // 20% forgiveness

            if (dist < hitRadius && dist < closestDist) {
                closestDist = dist;
                closestId = id;
            }
        }

        return closestId;
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
            this.updateHover(this.hitTest(sx, sy));
        }
    }

    private onPointerUp(e: PointerEvent): void {
        this.releasePointer(e);

        if (!this.drag.active || e.pointerId !== this.drag.pointerId) return;

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
                    // Deselect all others
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
            // Clicked empty canvas - deselect all
            if (this.opts.selection) {
                for (const id of this.selectedNodes) {
                    this.emit('node:deselect', this.graph.getNode(id), e);
                }
                this.selectedNodes.clear();
            }
            this.emit('canvas:click', null, e);
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

        if (this.drag.isPan && this.opts.pan) {
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

    private unpinDragged(): void {
        // In animated mode, unpin the node so it rejoins physics
        if (this.drag.nodeId !== null && this.animatedMode && this.animatedLayout) {
            this.animatedLayout.unpin(this.drag.nodeId);
        }
    }

    private resetDrag(): void {
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

    private emit(type: GraphGPUEventType, target: unknown, originalEvent?: Event): void {
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
