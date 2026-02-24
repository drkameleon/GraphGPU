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

        this.attach();
    }

    /** Attach event listeners to canvas */
    attach(): void {
        this.canvas.addEventListener('mousedown', this.boundMouseDown);
        this.canvas.addEventListener('mousemove', this.boundMouseMove);
        this.canvas.addEventListener('mouseup', this.boundMouseUp);
        this.canvas.addEventListener('wheel', this.boundWheel, { passive: false });
        this.canvas.addEventListener('dblclick', this.boundDblClick);
        this.canvas.style.cursor = 'grab';
    }

    /** Detach event listeners */
    detach(): void {
        this.canvas.removeEventListener('mousedown', this.boundMouseDown);
        this.canvas.removeEventListener('mousemove', this.boundMouseMove);
        this.canvas.removeEventListener('mouseup', this.boundMouseUp);
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

    // Gravity-pull mode: dragging one node pulls connected nodes
    private gravityPull: boolean = false;
    private gravityPullStrength: number = 0.15;

    /** Toggle multi-select always-on mode */
    setMultiSelectEnabled(enabled: boolean): void {
        this.alwaysMultiSelect = enabled;
    }

    /** Toggle gravity-pull mode */
    setGravityPull(enabled: boolean, strength?: number): void {
        this.gravityPull = enabled;
        if (strength !== undefined) this.gravityPullStrength = strength;
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

                // Gravity pull: connected nodes follow with spring force
                if (this.gravityPull) {
                    this.applyGravityPull(id, ddx, ddy);
                }

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

    // Velocity array for physics-based pull (lazy init)
    private pullVelocities: Float32Array | null = null;
    private pullAnimId: number = 0;

    /**
     * Gravity pull: apply impulse to all connected nodes via BFS.
     * Nodes accumulate velocity and coast with damping after release.
     */
    private applyGravityPull(sourceId: NodeId, dx: number, dy: number): void {
        const n = this.graph.numNodes;
        if (!this.pullVelocities || this.pullVelocities.length < n * 2) {
            this.pullVelocities = new Float32Array(n * 2);
        }

        const strength = this.gravityPullStrength;
        const indices = this.graph.edgeIndices;
        const edgeCount = this.graph.numEdges;

        // Build adjacency
        const adj = new Map<NodeId, NodeId[]>();
        for (let i = 0; i < edgeCount; i++) {
            if (!this.graph.isEdgeActive(i)) continue;
            const src = indices[i * 2];
            const tgt = indices[i * 2 + 1];
            if (!adj.has(src)) adj.set(src, []);
            if (!adj.has(tgt)) adj.set(tgt, []);
            adj.get(src)!.push(tgt);
            adj.get(tgt)!.push(src);
        }

        // BFS: apply velocity impulse with hop decay
        const visited = new Set<NodeId>([sourceId]);
        let frontier: NodeId[] = [sourceId];
        let hop = 0;
        const decay = 0.6;

        while (frontier.length > 0 && hop < 12) {
            hop++;
            const factor = strength * Math.pow(decay, hop);
            if (factor < 0.002) break;

            const next: NodeId[] = [];
            for (const node of frontier) {
                for (const nb of (adj.get(node) ?? [])) {
                    if (visited.has(nb) || !this.graph.isNodeActive(nb)) continue;
                    visited.add(nb);
                    next.push(nb);
                    // Add to velocity (accumulates during drag)
                    this.pullVelocities[nb * 2] += dx * factor;
                    this.pullVelocities[nb * 2 + 1] += dy * factor;
                }
            }
            frontier = next;
        }

        // Start coast animation if not already running
        if (!this.pullAnimId) {
            this.startPullCoast();
        }
    }

    /** Animate pull velocities with damping until settled */
    private startPullCoast(): void {
        const damping = 0.92;
        const minVel = 0.0001;

        const tick = () => {
            if (!this.pullVelocities) return;
            const positions = this.graph.positions;
            let maxV = 0;

            for (const id of this.graph.activeNodeIds()) {
                const vx = this.pullVelocities[id * 2] *= damping;
                const vy = this.pullVelocities[id * 2 + 1] *= damping;

                if (Math.abs(vx) > minVel || Math.abs(vy) > minVel) {
                    positions[id * 2] += vx;
                    positions[id * 2 + 1] += vy;
                    maxV = Math.max(maxV, Math.abs(vx), Math.abs(vy));
                }
            }

            if (maxV > minVel) {
                this.graph.dirtyNodes = true;
                this.graph.dirtyEdges = true;
                this.pullAnimId = requestAnimationFrame(tick);
            } else {
                this.pullAnimId = 0;
            }
        };

        this.pullAnimId = requestAnimationFrame(tick);
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
