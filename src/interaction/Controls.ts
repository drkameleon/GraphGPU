// ============================================================
// graphGPU - Interaction Controller
// ============================================================
// Handles mouse/touch input for pan, zoom, node picking,
// dragging, and selection. Spatial hit testing via brute
// force (upgrade to quadtree for >50k nodes).
// ============================================================

import type { NodeId, InteractionOptions, GraphGPUEventType, GraphGPUEventHandler } from '../types';
import type { Graph } from '../core/Graph';
import { Camera } from '../core/Camera';

interface DragState {
    active: boolean;
    nodeId: NodeId | null;
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
    isPan: boolean;
    moved: boolean;
}

export class InteractionController {
    private canvas: HTMLCanvasElement;
    private graph: Graph;
    private camera: Camera;
    private opts: Required<InteractionOptions>;
    private nodeScale: number;

    // State
    private drag: DragState = {
        active: false,
        nodeId: null,
        startX: 0, startY: 0,
        lastX: 0, lastY: 0,
        isPan: false,
        moved: false,
    };

    private hoveredNode: NodeId | null = null;
    private selectedNodes: Set<NodeId> = new Set();

    // Event listeners
    private listeners = new Map<GraphGPUEventType, Set<GraphGPUEventHandler>>();

    // Bound handlers (for cleanup)
    private boundMouseDown: (e: MouseEvent) => void;
    private boundMouseMove: (e: MouseEvent) => void;
    private boundMouseUp: (e: MouseEvent) => void;
    private boundWheel: (e: WheelEvent) => void;
    private boundDblClick: (e: MouseEvent) => void;

    // Touch handlers
    private boundTouchStart: (e: TouchEvent) => void;
    private boundTouchMove: (e: TouchEvent) => void;
    private boundTouchEnd: (e: TouchEvent) => void;

    // Pinch-zoom state
    private lastPinchDist: number = 0;

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
        this.boundMouseDown = this.onMouseDown.bind(this);
        this.boundMouseMove = this.onMouseMove.bind(this);
        this.boundMouseUp = this.onMouseUp.bind(this);
        this.boundWheel = this.onWheel.bind(this);
        this.boundDblClick = this.onDblClick.bind(this);

        // Touch handlers
        this.boundTouchStart = this.onTouchStart.bind(this);
        this.boundTouchMove = this.onTouchMove.bind(this);
        this.boundTouchEnd = this.onTouchEnd.bind(this);

        this.attach();
    }

    /** Attach event listeners to canvas */
    attach(): void {
        this.canvas.addEventListener('mousedown', this.boundMouseDown);
        this.canvas.addEventListener('mousemove', this.boundMouseMove);
        this.canvas.addEventListener('mouseup', this.boundMouseUp);
        this.canvas.addEventListener('wheel', this.boundWheel, { passive: false });
        this.canvas.addEventListener('dblclick', this.boundDblClick);

        // Touch events
        this.canvas.addEventListener('touchstart', this.boundTouchStart, { passive: false });
        this.canvas.addEventListener('touchmove', this.boundTouchMove, { passive: false });
        this.canvas.addEventListener('touchend', this.boundTouchEnd);
        this.canvas.addEventListener('touchcancel', this.boundTouchEnd);

        this.canvas.style.cursor = 'grab';
        // Prevent browser's default touch behaviors (scroll, zoom)
        this.canvas.style.touchAction = 'none';
    }

    /** Detach event listeners */
    detach(): void {
        this.canvas.removeEventListener('mousedown', this.boundMouseDown);
        this.canvas.removeEventListener('mousemove', this.boundMouseMove);
        this.canvas.removeEventListener('mouseup', this.boundMouseUp);
        this.canvas.removeEventListener('wheel', this.boundWheel);
        this.canvas.removeEventListener('dblclick', this.boundDblClick);

        this.canvas.removeEventListener('touchstart', this.boundTouchStart);
        this.canvas.removeEventListener('touchmove', this.boundTouchMove);
        this.canvas.removeEventListener('touchend', this.boundTouchEnd);
        this.canvas.removeEventListener('touchcancel', this.boundTouchEnd);
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
    private animatedLayout: import('../layout/ForceLayout').ForceLayout | null = null;

    /** Toggle multi-select always-on mode */
    setMultiSelectEnabled(enabled: boolean): void {
        this.alwaysMultiSelect = enabled;
    }

    /** Set animated mode + provide layout reference for pin/unpin */
    setAnimatedMode(enabled: boolean, layout: import('../layout/ForceLayout').ForceLayout | null): void {
        this.animatedMode = enabled;
        this.animatedLayout = layout;
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
    // Mouse handlers
    // =========================================================

    private getCanvasCoords(e: MouseEvent): [number, number] {
        const rect = this.canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio;
        return [
            (e.clientX - rect.left) * ratio,
            (e.clientY - rect.top) * ratio,
        ];
    }

    private onMouseDown(e: MouseEvent): void {
        const [sx, sy] = this.getCanvasCoords(e);

        const hitNode = this.opts.dragNodes ? this.hitTest(sx, sy) : null;

        this.drag.active = true;
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

    private onMouseMove(e: MouseEvent): void {
        const [sx, sy] = this.getCanvasCoords(e);

        if (this.drag.active) {
            const dx = sx - this.drag.lastX;
            const dy = sy - this.drag.lastY;
            const totalDx = Math.abs(sx - this.drag.startX);
            const totalDy = Math.abs(sy - this.drag.startY);

            if (totalDx > 3 || totalDy > 3) {
                this.drag.moved = true;
            }

            if (this.drag.isPan && this.opts.pan) {
                if (this.drag.moved) this.canvas.style.cursor = 'grabbing';
                this.camera.pan(dx, dy);
                this.emit('canvas:pan', this.camera.getState());
            } else if (this.drag.nodeId !== null && this.opts.dragNodes) {
                this.canvas.style.cursor = 'grabbing';
                // Move node in world space
                const [wx1, wy1] = this.camera.screenToWorld(this.drag.lastX, this.drag.lastY);
                const [wx2, wy2] = this.camera.screenToWorld(sx, sy);
                const id = this.drag.nodeId;
                const ddx = wx2 - wx1;
                const ddy = wy2 - wy1;
                const newX = this.graph.positions[id * 2] + ddx;
                const newY = this.graph.positions[id * 2 + 1] + ddy;
                this.graph.setNodePosition(id, newX, newY);

                this.emit('node:drag', this.graph.getNode(id));
            }

            this.drag.lastX = sx;
            this.drag.lastY = sy;
        } else if (this.opts.hover) {
            // Hover detection
            const hitNode = this.hitTest(sx, sy);

            if (hitNode !== this.hoveredNode) {
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
        }
    }

    private onMouseUp(_e: MouseEvent): void {
        if (this.drag.active) {
            const isClick = !this.drag.moved;

            if (this.drag.nodeId !== null) {
                // In animated mode, unpin the node so it rejoins physics
                if (this.animatedMode && this.animatedLayout) {
                    this.animatedLayout.unpin(this.drag.nodeId);
                }
                this.emit('node:dragend', this.graph.getNode(this.drag.nodeId));

                // Selection happens on mouseUp (click without drag)
                if (isClick && this.opts.selection) {
                    const id = this.drag.nodeId;
                    const isMulti = this.alwaysMultiSelect || _e.shiftKey;

                    if (isMulti) {
                        if (this.selectedNodes.has(id)) {
                            this.selectedNodes.delete(id);
                            this.emit('node:deselect', this.graph.getNode(id));
                        } else {
                            this.selectedNodes.add(id);
                            this.emit('node:select', this.graph.getNode(id));
                        }
                    } else {
                        // Deselect all others
                        for (const prev of this.selectedNodes) {
                            if (prev !== id) this.emit('node:deselect', this.graph.getNode(prev));
                        }
                        this.selectedNodes.clear();
                        this.selectedNodes.add(id);
                        this.emit('node:select', this.graph.getNode(id));
                    }

                    this.emit('node:click', this.graph.getNode(id));
                }
            } else {
                if (isClick) {
                    // Clicked empty canvas - deselect all
                    if (this.opts.selection) {
                        for (const id of this.selectedNodes) {
                            this.emit('node:deselect', this.graph.getNode(id));
                        }
                        this.selectedNodes.clear();
                    }
                    this.emit('canvas:click', null);
                }
            }

            this.drag.active = false;
            this.drag.nodeId = null;
            this.canvas.style.cursor = this.hoveredNode !== null ? 'pointer' : 'grab';
        }
    }

    // =========================================================
    // Touch handlers
    // =========================================================

    private getTouchCanvasCoords(touch: Touch): [number, number] {
        const rect = this.canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio;
        return [
            (touch.clientX - rect.left) * ratio,
            (touch.clientY - rect.top) * ratio,
        ];
    }

    private onTouchStart(e: TouchEvent): void {
        e.preventDefault();

        if (e.touches.length === 2) {
            // Start pinch-zoom
            const dx = e.touches[1].clientX - e.touches[0].clientX;
            const dy = e.touches[1].clientY - e.touches[0].clientY;
            this.lastPinchDist = Math.hypot(dx, dy);
            // Cancel any active single-finger drag
            this.drag.active = false;
            this.drag.nodeId = null;
            return;
        }

        if (e.touches.length !== 1) return;

        const [sx, sy] = this.getTouchCanvasCoords(e.touches[0]);
        const hitNode = this.opts.dragNodes ? this.hitTest(sx, sy) : null;

        this.drag.active = true;
        this.drag.startX = sx;
        this.drag.startY = sy;
        this.drag.lastX = sx;
        this.drag.lastY = sy;
        this.drag.moved = false;

        if (hitNode !== null) {
            this.drag.nodeId = hitNode;
            this.drag.isPan = false;
            if (this.animatedMode && this.animatedLayout) {
                this.animatedLayout.pin(hitNode);
            }
        } else {
            this.drag.nodeId = null;
            this.drag.isPan = true;
        }
    }

    private onTouchMove(e: TouchEvent): void {
        e.preventDefault();

        if (e.touches.length === 2 && this.opts.zoom) {
            // Pinch-zoom
            const dx = e.touches[1].clientX - e.touches[0].clientX;
            const dy = e.touches[1].clientY - e.touches[0].clientY;
            const dist = Math.hypot(dx, dy);

            if (this.lastPinchDist > 0) {
                const delta = (dist - this.lastPinchDist) * 3; // sensitivity
                // Zoom toward midpoint of two fingers
                const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                const rect = this.canvas.getBoundingClientRect();
                const ratio = window.devicePixelRatio;
                const sx = (midX - rect.left) * ratio;
                const sy = (midY - rect.top) * ratio;
                this.camera.zoom(delta, sx, sy);
                this.emit('canvas:zoom', this.camera.getState());
            }
            this.lastPinchDist = dist;
            return;
        }

        if (e.touches.length !== 1 || !this.drag.active) return;

        const [sx, sy] = this.getTouchCanvasCoords(e.touches[0]);
        const dx = sx - this.drag.lastX;
        const dy = sy - this.drag.lastY;
        const totalDx = Math.abs(sx - this.drag.startX);
        const totalDy = Math.abs(sy - this.drag.startY);

        if (totalDx > 3 || totalDy > 3) {
            this.drag.moved = true;
        }

        if (this.drag.isPan && this.opts.pan) {
            this.camera.pan(dx, dy);
            this.emit('canvas:pan', this.camera.getState());
        } else if (this.drag.nodeId !== null && this.opts.dragNodes) {
            const [wx1, wy1] = this.camera.screenToWorld(this.drag.lastX, this.drag.lastY);
            const [wx2, wy2] = this.camera.screenToWorld(sx, sy);
            const id = this.drag.nodeId;
            const ddx = wx2 - wx1;
            const ddy = wy2 - wy1;
            const newX = this.graph.positions[id * 2] + ddx;
            const newY = this.graph.positions[id * 2 + 1] + ddy;
            this.graph.setNodePosition(id, newX, newY);

            this.emit('node:drag', this.graph.getNode(id));
        }

        this.drag.lastX = sx;
        this.drag.lastY = sy;
    }

    private onTouchEnd(e: TouchEvent): void {
        // Reset pinch state when fingers lift
        if (e.touches.length < 2) {
            this.lastPinchDist = 0;
        }

        if (e.touches.length > 0) return; // still touching

        if (this.drag.active) {
            const isClick = !this.drag.moved;

            if (this.drag.nodeId !== null) {
                // Unpin in animated mode
                if (this.animatedMode && this.animatedLayout) {
                    this.animatedLayout.unpin(this.drag.nodeId);
                }
                this.emit('node:dragend', this.graph.getNode(this.drag.nodeId));

                if (isClick && this.opts.selection) {
                    const id = this.drag.nodeId;
                    // Touch always acts as single-select
                    for (const prev of this.selectedNodes) {
                        if (prev !== id) this.emit('node:deselect', this.graph.getNode(prev));
                    }
                    this.selectedNodes.clear();
                    this.selectedNodes.add(id);
                    this.emit('node:select', this.graph.getNode(id));
                    this.emit('node:click', this.graph.getNode(id));
                }
            } else {
                if (isClick) {
                    if (this.opts.selection) {
                        for (const id of this.selectedNodes) {
                            this.emit('node:deselect', this.graph.getNode(id));
                        }
                        this.selectedNodes.clear();
                    }
                    this.emit('canvas:click', null);
                }
            }

            this.drag.active = false;
            this.drag.nodeId = null;
        }
    }

    private onWheel(e: WheelEvent): void {
        if (!this.opts.zoom) return;
        e.preventDefault();

        const [sx, sy] = this.getCanvasCoords(e);
        this.camera.zoom(-e.deltaY, sx, sy);
        this.emit('canvas:zoom', this.camera.getState());
    }

    private onDblClick(e: MouseEvent): void {
        const [sx, sy] = this.getCanvasCoords(e);
        const hitNode = this.hitTest(sx, sy);
        if (hitNode !== null) {
            this.emit('node:dblclick', this.graph.getNode(hitNode));
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
