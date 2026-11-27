// ============================================================
// graphGPU - Graph Data Structure
// ============================================================
// Struct-of-Arrays layout for GPU-friendly data.
// All node/edge data lives in contiguous typed arrays.
// Object accessors are compile-time only views.
// ============================================================

import type {
    NodeId, EdgeId, Node, Edge,
    NodeInput, EdgeInput,
} from '../types';
import { TagColorMap } from '../utils/palette';
import { parseColor, writeColorToBuffer, type RGBA } from '../utils/color';

// -----------------------------------------------------------
// Buffer growth strategy
// -----------------------------------------------------------

const GROWTH_FACTOR = 2;
const DEFAULT_MAX_NODES = 4096;
const DEFAULT_MAX_EDGES = 8192;

function growBuffer<T extends Float32Array | Uint32Array | Uint8Array>(
    old: T,
    newLength: number,
): T {
    const Ctor = old.constructor as { new(length: number): T };
    const next = new Ctor(newLength);
    next.set(old);
    return next;
}

/**
 * Pop a reusable slot off a free list, or -1 if there is none.
 * Entries go stale when the tail shrinks past them (and the slot
 * may since have been reused sequentially), so skip any slot
 * that is out of range or already active.
 */
function popFreeSlot(
    free: number[],
    tags: (string | undefined)[],
    count: number,
): number {
    while (free.length > 0) {
        const id = free.pop()!;
        if (id < count && tags[id] === undefined) return id;
    }
    return -1;
}

// -----------------------------------------------------------
// Graph
// -----------------------------------------------------------

export class Graph {
    // --- Node SoA buffers ---
    positions: Float32Array;   // [x0, y0, x1, y1, ...]
    colors: Float32Array;      // [r0, g0, b0, a0, r1, ...]
    sizes: Float32Array;       // [s0, s1, ...]

    /** 1 if a node's color was set explicitly (kept on palette changes) */
    private customColor: Uint8Array;

    // --- Node metadata (not uploaded to GPU) ---
    private nodeTags: string[];
    private nodeProperties: Record<string, unknown>[];
    private nodeCount: number = 0;     // slots in use (highest id + 1)
    private liveNodes: number = 0;     // active nodes
    private nodeCapacity: number;

    // --- Edge SoA buffers ---
    edgeIndices: Uint32Array;  // [src0, tgt0, src1, tgt1, ...]
    edgeWeights: Float32Array; // [w0, w1, ...]
    edgeColors: Float32Array;  // [r0, g0, b0, r1, g1, b1, ...]  per-edge RGB

    // --- Edge metadata ---
    private edgeTags: string[];
    private edgeProperties: Record<string, unknown>[];
    private edgeCount: number = 0;     // slots in use (highest id + 1)
    private liveEdges: number = 0;     // active edges
    private edgeCapacity: number;

    // --- Free lists for reuse after deletion ---
    private freeNodes: NodeId[] = [];
    private freeEdges: EdgeId[] = [];

    // --- Adjacency index: nodeId → Set of edgeIds ---
    // Avoids O(edges) scan when removing a node's edges
    // Built lazily (on first need) and maintained from then on, so
    // bulk-adding edges stays cheap. null = not built yet.
    // node id → its edge ids (plain arrays: much cheaper to build than Sets)
    private nodeAdj: (EdgeId[] | undefined)[] | null = null;

    // --- Color management ---
    readonly tagColors: TagColorMap;

    // --- Dirty tracking for GPU upload ---
    private _dirtyNodes: boolean = true;

    /** Anything about nodes changed (incl. positions) */
    get dirtyNodes(): boolean {
        return this._dirtyNodes;
    }
    set dirtyNodes(dirty: boolean) {
        this._dirtyNodes = dirty;
        if (dirty) this.nodesVersion++;
    }

    /**
     * Bumped whenever nodes are marked dirty (moved, added, removed,
     * resized...), so caches like the picking grid know to rebuild.
     */
    nodesVersion: number = 0;
    dirtyNodeAttributes: boolean = true;  // colors / sizes (not just positions)
    dirtyEdges: boolean = true;

    /**
     * Bumped whenever nodes/edges are added or removed (not on
     * position/color changes), so consumers like the GPU layout
     * know when to re-upload the graph structure.
     */
    structureVersion: number = 0;

    /** Bumped when edge colors change in bulk (e.g. palette switch) */
    edgeColorVersion: number = 0;

    /**
     * Position edits made while positions live on the GPU (GPU layout
     * running): node id → [x, y]. null when not tracking.
     */
    private positionEdits: Map<NodeId, [number, number]> | null = null;

    // --- Default node size ---
    private defaultNodeSize: number;

    constructor(opts?: {
        maxNodes?: number;
        maxEdges?: number;
        nodeSize?: number;
        palette?: string;
    }) {
        this.nodeCapacity = opts?.maxNodes ?? DEFAULT_MAX_NODES;
        this.edgeCapacity = opts?.maxEdges ?? DEFAULT_MAX_EDGES;
        this.defaultNodeSize = opts?.nodeSize ?? 6;

        // Allocate node buffers
        this.positions = new Float32Array(this.nodeCapacity * 2);
        this.colors = new Float32Array(this.nodeCapacity * 4);
        this.sizes = new Float32Array(this.nodeCapacity);
        this.customColor = new Uint8Array(this.nodeCapacity);
        this.nodeTags = new Array(this.nodeCapacity);
        this.nodeProperties = new Array(this.nodeCapacity);

        // Allocate edge buffers
        this.edgeIndices = new Uint32Array(this.edgeCapacity * 2);
        this.edgeWeights = new Float32Array(this.edgeCapacity);
        this.edgeColors = new Float32Array(this.edgeCapacity * 3);
        this.edgeTags = new Array(this.edgeCapacity);
        this.edgeProperties = new Array(this.edgeCapacity);

        // Color management
        this.tagColors = new TagColorMap(opts?.palette ?? 'default');
    }

    // =========================================================
    // Node operations
    // =========================================================

    /** Number of active nodes */
    get numNodes(): number {
        return this.liveNodes;
    }

    /** Number of active edges */
    get numEdges(): number {
        return this.liveEdges;
    }

    /**
     * Number of node slots in use (highest node id + 1), including
     * removed ones. This is the extent of the node buffers.
     */
    get nodeSlots(): number {
        return this.nodeCount;
    }

    /**
     * Number of edge slots in use (highest edge id + 1), including
     * removed ones. This is the extent of the edge buffers.
     */
    get edgeSlots(): number {
        return this.edgeCount;
    }

    /** Add a node, returns its ID */
    addNode(input: NodeInput): NodeId {
        let id: NodeId;

        const free = popFreeSlot(this.freeNodes, this.nodeTags, this.nodeCount);
        if (free >= 0) {
            id = free;
        } else {
            id = this.nodeCount;
            if (id >= this.nodeCapacity) {
                this.growNodes();
            }
        }

        this.nodeCount = Math.max(this.nodeCount, id + 1);
        this.liveNodes++;
        this.structureVersion++;

        // Position
        const px = input.x ?? (Math.random() - 0.5) * 2;
        const py = input.y ?? (Math.random() - 0.5) * 2;
        this.positions[id * 2] = px;
        this.positions[id * 2 + 1] = py;
        this.positionEdits?.set(id, [px, py]);

        // Size
        this.sizes[id] = input.size ?? this.defaultNodeSize;

        // Color (auto-assign from palette if not provided)
        const tagColor = this.tagColors.getColor(input.tag);
        const color: RGBA = input.color
            ? parseColor(input.color)
            : tagColor.bg;
        writeColorToBuffer(this.colors, id * 4, color);
        this.customColor[id] = input.color ? 1 : 0;

        // Metadata
        this.nodeTags[id] = input.tag;
        this.nodeProperties[id] = input.properties ?? {};

        this.dirtyNodes = true;
        this.dirtyNodeAttributes = true;
        return id;
    }

    /** Batch-add nodes (minimizes overhead) */
    addNodes(inputs: NodeInput[]): NodeId[] {
        const ids: NodeId[] = new Array(inputs.length);
        for (let i = 0; i < inputs.length; i++) {
            ids[i] = this.addNode(inputs[i]);
        }
        return ids;
    }

    /** Remove a node and all its edges */
    removeNode(id: NodeId): void {
        if (id < 0 || id >= this.nodeCount) return;
        if (this.nodeTags[id] === undefined) return; // already deleted

        // Remove all edges connected to this node
        this.removeNodeEdges(id);

        // Zero out buffers
        this.positions[id * 2] = 0;
        this.positions[id * 2 + 1] = 0;
        this.colors[id * 4] = 0;
        this.colors[id * 4 + 1] = 0;
        this.colors[id * 4 + 2] = 0;
        this.colors[id * 4 + 3] = 0;
        this.sizes[id] = 0;
        this.customColor[id] = 0;

        // Clear metadata
        this.nodeTags[id] = undefined!;
        this.nodeProperties[id] = undefined!;

        this.freeNodes.push(id);
        this.liveNodes--;
        this.structureVersion++;

        // Shrink nodeCount if we deleted from the tail
        // This avoids iterating/uploading dead trailing slots
        while (this.nodeCount > 0 && this.nodeTags[this.nodeCount - 1] === undefined) {
            this.nodeCount--;
        }
        // Free-list entries beyond nodeCount are now stale; they are
        // skipped lazily by popFreeSlot instead of filtered here (O(1))

        this.dirtyNodes = true;
        this.dirtyNodeAttributes = true;
    }

    /** Get a node view (allocates - use sparingly in hot paths) */
    getNode(id: NodeId): Node | null {
        if (id < 0 || id >= this.nodeCount || this.nodeTags[id] === undefined) return null;
        return {
            id,
            tag: this.nodeTags[id],
            x: this.positions[id * 2],
            y: this.positions[id * 2 + 1],
            size: this.sizes[id],
            r: this.colors[id * 4],
            g: this.colors[id * 4 + 1],
            b: this.colors[id * 4 + 2],
            a: this.colors[id * 4 + 3],
            properties: this.nodeProperties[id],
        };
    }

    /** Get node properties without allocating a view */
    getNodeProperties(id: NodeId): Record<string, unknown> | undefined {
        return this.nodeProperties[id];
    }

    /** Get edge tag without allocating */
    getEdgeTag(id: EdgeId): string | undefined {
        return this.edgeTags[id];
    }

    /** Get node tag without allocating */
    getNodeTag(id: NodeId): string | undefined {
        return this.nodeTags[id];
    }

    /** Check if a node ID is active */
    isNodeActive(id: NodeId): boolean {
        return id >= 0 && id < this.nodeCount && this.nodeTags[id] !== undefined;
    }

    /** Update node position (hot path - zero alloc) */
    setNodePosition(id: NodeId, x: number, y: number): void {
        this.positions[id * 2] = x;
        this.positions[id * 2 + 1] = y;
        this.positionEdits?.set(id, [x, y]);
        this.dirtyNodes = true;
        this.dirtyEdges = true;  // edges reference node positions
    }

    // --- Position edits (while positions live on the GPU) ---

    /** Start/stop recording position edits */
    trackPositionEdits(enabled: boolean): void {
        this.positionEdits = enabled ? new Map() : null;
    }

    /** Take (and clear) the recorded position edits */
    takePositionEdits(): [NodeId, [number, number]][] {
        if (!this.positionEdits || this.positionEdits.size === 0) return [];
        const edits = [...this.positionEdits];
        this.positionEdits.clear();
        return edits;
    }

    /** Re-apply pending edits over positions read back from the GPU */
    reapplyPositionEdits(): void {
        if (!this.positionEdits) return;
        for (const [id, [x, y]] of this.positionEdits) {
            this.positions[id * 2] = x;
            this.positions[id * 2 + 1] = y;
        }
    }

    /** Update node color */
    setNodeColor(id: NodeId, color: RGBA): void {
        writeColorToBuffer(this.colors, id * 4, color);
        this.customColor[id] = 1;
        this.dirtyNodes = true;
        this.dirtyNodeAttributes = true;
    }

    /** Update node size */
    setNodeSize(id: NodeId, size: number): void {
        this.sizes[id] = size;
        this.dirtyNodes = true;
        this.dirtyNodeAttributes = true;
    }

    /** Update node property */
    setNodeProperty(id: NodeId, key: string, value: unknown): void {
        if (this.nodeProperties[id]) {
            this.nodeProperties[id][key] = value;
        }
    }

    // =========================================================
    // Edge operations
    // =========================================================

    /** Add an edge, returns its ID */
    addEdge(input: EdgeInput): EdgeId {
        let id: EdgeId;

        const free = popFreeSlot(this.freeEdges, this.edgeTags, this.edgeCount);
        if (free >= 0) {
            id = free;
        } else {
            id = this.edgeCount;
            if (id >= this.edgeCapacity) {
                this.growEdges();
            }
        }

        this.edgeCount = Math.max(this.edgeCount, id + 1);
        this.liveEdges++;
        this.structureVersion++;

        this.edgeIndices[id * 2] = input.source;
        this.edgeIndices[id * 2 + 1] = input.target;
        this.edgeWeights[id] = input.weight ?? 1.0;

        this.edgeTags[id] = input.tag;
        this.edgeProperties[id] = input.properties ?? {};

        // Assign edge color from palette based on tag
        if (input.tag) {
            const tagColor = this.tagColors.getColor(input.tag);
            this.edgeColors[id * 3] = tagColor.bg[0];
            this.edgeColors[id * 3 + 1] = tagColor.bg[1];
            this.edgeColors[id * 3 + 2] = tagColor.bg[2];
        } else {
            // Default gray for untagged edges
            this.edgeColors[id * 3] = 0.55;
            this.edgeColors[id * 3 + 1] = 0.53;
            this.edgeColors[id * 3 + 2] = 0.68;
        }

        // Update adjacency index
        if (this.nodeAdj) {
            this.adjAdd(input.source, id);
            if (input.target !== input.source) this.adjAdd(input.target, id);
        }

        this.dirtyEdges = true;
        return id;
    }

    /** Batch-add edges */
    addEdges(inputs: EdgeInput[]): EdgeId[] {
        const ids: EdgeId[] = new Array(inputs.length);
        for (let i = 0; i < inputs.length; i++) {
            ids[i] = this.addEdge(inputs[i]);
        }
        return ids;
    }

    /** Remove an edge */
    removeEdge(id: EdgeId): void {
        if (id < 0 || id >= this.edgeCount) return;
        if (this.edgeTags[id] === undefined) return;

        // Clean adjacency index before zeroing
        const src = this.edgeIndices[id * 2];
        const tgt = this.edgeIndices[id * 2 + 1];
        if (this.nodeAdj) {
            this.adjRemove(src, id);
            if (tgt !== src) this.adjRemove(tgt, id);
        }

        this.edgeIndices[id * 2] = 0;
        this.edgeIndices[id * 2 + 1] = 0;
        this.edgeWeights[id] = 0;
        this.edgeTags[id] = undefined!;
        this.edgeProperties[id] = undefined!;

        this.freeEdges.push(id);
        this.liveEdges--;
        this.structureVersion++;

        // Shrink edgeCount if we deleted from the tail
        while (this.edgeCount > 0 && this.edgeTags[this.edgeCount - 1] === undefined) {
            this.edgeCount--;
        }

        this.dirtyEdges = true;
    }

    /** Remove all edges connected to a node — O(degree) via adjacency index */
    removeNodeEdges(nodeId: NodeId): void {
        const index = this.adjacency();
        const edgeIds = index[nodeId];
        if (!edgeIds) return;
        // Detach the node's own list first, so each removal only has to
        // update the other endpoint's list (no O(degree²) for hubs)
        index[nodeId] = undefined;
        for (const eid of edgeIds) {
            this.removeEdge(eid);
        }
    }

    // --- Adjacency index helpers ---

    /** The adjacency index, built on first use: O(edges) once */
    private adjacency(): (EdgeId[] | undefined)[] {
        if (!this.nodeAdj) {
            this.nodeAdj = [];
            for (let id = 0; id < this.edgeCount; id++) {
                if (this.edgeTags[id] === undefined) continue;
                const a = this.edgeIndices[id * 2];
                const b = this.edgeIndices[id * 2 + 1];
                this.adjAdd(a, id);
                if (b !== a) this.adjAdd(b, id);
            }
        }
        return this.nodeAdj;
    }

    private adjAdd(nodeId: NodeId, edgeId: EdgeId): void {
        const list = this.nodeAdj![nodeId];
        if (list) list.push(edgeId);
        else this.nodeAdj![nodeId] = [edgeId];
    }

    private adjRemove(nodeId: NodeId, edgeId: EdgeId): void {
        const list = this.nodeAdj![nodeId];
        if (!list) return;
        const i = list.lastIndexOf(edgeId);
        if (i < 0) return;
        // Swap-remove (order doesn't matter, queries sort)
        list[i] = list[list.length - 1];
        list.pop();
        if (list.length === 0) this.nodeAdj![nodeId] = undefined;
    }

    /** Get an edge view */
    getEdge(id: EdgeId): Edge | null {
        if (id < 0 || id >= this.edgeCount || this.edgeTags[id] === undefined) return null;
        return {
            id,
            tag: this.edgeTags[id],
            source: this.edgeIndices[id * 2],
            target: this.edgeIndices[id * 2 + 1],
            weight: this.edgeWeights[id],
            properties: this.edgeProperties[id],
        };
    }

    /** Check if an edge ID is active */
    isEdgeActive(id: EdgeId): boolean {
        return id >= 0 && id < this.edgeCount && this.edgeTags[id] !== undefined;
    }

    // =========================================================
    // Query operations (graphGPU-style)
    // =========================================================

    /** Fetch nodes by tag, optionally filtering by properties */
    fetchByTag(
        tag: string,
        filter?: Record<string, unknown>,
    ): Node[] {
        const results: Node[] = [];
        for (let i = 0; i < this.nodeCount; i++) {
            if (this.nodeTags[i] !== tag) continue;
            if (filter) {
                const props = this.nodeProperties[i];
                let match = true;
                for (const [k, v] of Object.entries(filter)) {
                    if (props[k] !== v) { match = false; break; }
                }
                if (!match) continue;
            }
            results.push(this.getNode(i)!);
        }
        return results;
    }

    /** Get all edges for a given node */
    /** Ids of a node's edges, in ascending order — O(degree) */
    getNodeEdgeIds(nodeId: NodeId): EdgeId[] {
        const list = this.adjacency()[nodeId];
        return list ? list.slice().sort((a, b) => a - b) : [];
    }

    getNodeEdges(nodeId: NodeId): Edge[] {
        return this.getNodeEdgeIds(nodeId).map(id => this.getEdge(id)!);
    }

    /** Get all related nodes (neighbors) for a given node — O(degree) */
    getRelatedNodes(nodeId: NodeId): Node[] {
        const neighborIds = new Set<NodeId>();
        for (const i of this.getNodeEdgeIds(nodeId)) {
            const src = this.edgeIndices[i * 2];
            const tgt = this.edgeIndices[i * 2 + 1];
            if (src === nodeId) neighborIds.add(tgt);
            if (tgt === nodeId) neighborIds.add(src);
        }
        return [...neighborIds]
            .map(id => this.getNode(id))
            .filter((n): n is Node => n !== null);
    }

    /** Get all unique node tags */
    getAllTags(): string[] {
        const tags = new Set<string>();
        for (let i = 0; i < this.nodeCount; i++) {
            if (this.nodeTags[i] !== undefined) tags.add(this.nodeTags[i]);
        }
        return [...tags];
    }

    /** Iterate over active node IDs (no allocation per node) */
    *activeNodeIds(): Generator<NodeId> {
        for (let i = 0; i < this.nodeCount; i++) {
            if (this.nodeTags[i] !== undefined) yield i;
        }
    }

    /** Iterate over active edge IDs */
    *activeEdgeIds(): Generator<EdgeId> {
        for (let i = 0; i < this.edgeCount; i++) {
            if (this.edgeTags[i] !== undefined) yield i;
        }
    }

    /** Clear entire graph */
    clear(): void {
        this.nodeCount = 0;
        this.edgeCount = 0;
        this.liveNodes = 0;
        this.liveEdges = 0;
        this.structureVersion++;
        this.freeNodes.length = 0;
        this.freeEdges.length = 0;
        this.positions.fill(0);
        this.colors.fill(0);
        this.sizes.fill(0);
        this.customColor.fill(0);
        this.edgeIndices.fill(0);
        this.edgeWeights.fill(0);
        this.edgeColors.fill(0);
        this.nodeAdj = null;
        this.nodeTags.length = 0;
        this.nodeProperties.length = 0;
        this.edgeTags.length = 0;
        this.edgeProperties.length = 0;
        this.tagColors.clear();
        this.dirtyNodes = true;
        this.dirtyNodeAttributes = true;
        this.dirtyEdges = true;
    }

    /**
     * Re-apply tag colors (e.g. after a palette change) to all nodes
     * and tagged edges. Nodes with an explicit color keep it.
     */
    applyTagColors(): void {
        for (let id = 0; id < this.nodeCount; id++) {
            const tag = this.nodeTags[id];
            if (tag === undefined || this.customColor[id]) continue;
            writeColorToBuffer(this.colors, id * 4, this.tagColors.getColor(tag).bg);
        }
        for (let id = 0; id < this.edgeCount; id++) {
            const tag = this.edgeTags[id];
            if (!tag) continue; // removed, or untagged (stays gray)
            const bg = this.tagColors.getColor(tag).bg;
            this.edgeColors[id * 3] = bg[0];
            this.edgeColors[id * 3 + 1] = bg[1];
            this.edgeColors[id * 3 + 2] = bg[2];
        }
        this.edgeColorVersion++;
        this.dirtyNodes = true;
        this.dirtyNodeAttributes = true;
        this.dirtyEdges = true;
    }

    /** Re-randomize positions of all active nodes (for layout restart) */
    resetPositions(): void {
        for (const id of this.activeNodeIds()) {
            const x = (Math.random() - 0.5) * 2;
            const y = (Math.random() - 0.5) * 2;
            this.positions[id * 2] = x;
            this.positions[id * 2 + 1] = y;
            this.positionEdits?.set(id, [x, y]);
        }
        this.dirtyNodes = true;
        this.dirtyEdges = true; // edges depend on node positions
    }

    // =========================================================
    // Internal: buffer growth
    // =========================================================

    private growNodes(): void {
        const newCapacity = this.nodeCapacity * GROWTH_FACTOR;
        this.positions = growBuffer(this.positions, newCapacity * 2);
        this.colors = growBuffer(this.colors, newCapacity * 4);
        this.sizes = growBuffer(this.sizes, newCapacity);
        this.customColor = growBuffer(this.customColor, newCapacity);
        this.nodeTags.length = newCapacity;
        this.nodeProperties.length = newCapacity;
        this.nodeCapacity = newCapacity;
    }

    private growEdges(): void {
        const newCapacity = this.edgeCapacity * GROWTH_FACTOR;
        this.edgeIndices = growBuffer(this.edgeIndices, newCapacity * 2);
        this.edgeWeights = growBuffer(this.edgeWeights, newCapacity);
        this.edgeColors = growBuffer(this.edgeColors, newCapacity * 3);
        this.edgeTags.length = newCapacity;
        this.edgeProperties.length = newCapacity;
        this.edgeCapacity = newCapacity;
    }
}
