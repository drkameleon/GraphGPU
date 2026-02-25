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

function growBuffer<T extends Float32Array | Uint32Array>(
    old: T,
    newLength: number,
): T {
    const Ctor = old.constructor as { new(length: number): T };
    const next = new Ctor(newLength);
    next.set(old);
    return next;
}

// -----------------------------------------------------------
// Graph
// -----------------------------------------------------------

export class Graph {
    // --- Node SoA buffers ---
    positions: Float32Array;   // [x0, y0, x1, y1, ...]
    colors: Float32Array;      // [r0, g0, b0, a0, r1, ...]
    sizes: Float32Array;       // [s0, s1, ...]

    // --- Node metadata (not uploaded to GPU) ---
    private nodeTags: string[];
    private nodeProperties: Record<string, unknown>[];
    private nodeCount: number = 0;
    private nodeCapacity: number;

    // --- Edge SoA buffers ---
    edgeIndices: Uint32Array;  // [src0, tgt0, src1, tgt1, ...]
    edgeWeights: Float32Array; // [w0, w1, ...]
    edgeColors: Float32Array;  // [r0, g0, b0, r1, g1, b1, ...]  per-edge RGB

    // --- Edge metadata ---
    private edgeTags: string[];
    private edgeProperties: Record<string, unknown>[];
    private edgeCount: number = 0;
    private edgeCapacity: number;

    // --- Free lists for reuse after deletion ---
    private freeNodes: NodeId[] = [];
    private freeEdges: EdgeId[] = [];

    // --- Adjacency index: nodeId → Set of edgeIds ---
    // Avoids O(edges) scan when removing a node's edges
    private nodeAdj: Map<NodeId, Set<EdgeId>> = new Map();

    // --- Color management ---
    readonly tagColors: TagColorMap;

    // --- Dirty tracking for GPU upload ---
    dirtyNodes: boolean = true;
    dirtyEdges: boolean = true;

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
        return this.nodeCount;
    }

    /** Number of active edges */
    get numEdges(): number {
        return this.edgeCount;
    }

    /** Add a node, returns its ID */
    addNode(input: NodeInput): NodeId {
        let id: NodeId;

        if (this.freeNodes.length > 0) {
            id = this.freeNodes.pop()!;
        } else {
            id = this.nodeCount;
            if (id >= this.nodeCapacity) {
                this.growNodes();
            }
        }

        this.nodeCount = Math.max(this.nodeCount, id + 1);

        // Position
        const px = input.x ?? (Math.random() - 0.5) * 2;
        const py = input.y ?? (Math.random() - 0.5) * 2;
        this.positions[id * 2] = px;
        this.positions[id * 2 + 1] = py;

        // Size
        this.sizes[id] = input.size ?? this.defaultNodeSize;

        // Color (auto-assign from palette if not provided)
        const tagColor = this.tagColors.getColor(input.tag);
        const color: RGBA = input.color
            ? parseColor(input.color)
            : tagColor.bg;
        writeColorToBuffer(this.colors, id * 4, color);

        // Metadata
        this.nodeTags[id] = input.tag;
        this.nodeProperties[id] = input.properties ?? {};

        this.dirtyNodes = true;
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

        // Clear metadata
        this.nodeTags[id] = undefined!;
        this.nodeProperties[id] = undefined!;

        this.freeNodes.push(id);

        // Shrink nodeCount if we deleted from the tail
        // This avoids iterating/uploading dead trailing slots
        while (this.nodeCount > 0 && this.nodeTags[this.nodeCount - 1] === undefined) {
            this.nodeCount--;
        }
        // Remove any free-list entries that are now beyond nodeCount
        // (they'll be naturally reassigned by addNode's sequential path)
        if (this.freeNodes.length > 0) {
            this.freeNodes = this.freeNodes.filter(fid => fid < this.nodeCount);
        }

        this.dirtyNodes = true;
    }

    /** Get a node view (allocates - use sparingly in hot paths) */
    getNode(id: NodeId): Node | null {
        if (id < 0 || id >= this.nodeCount || !this.nodeTags[id]) return null;
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
        this.dirtyNodes = true;
        this.dirtyEdges = true;  // edges reference node positions
    }

    /** Update node color */
    setNodeColor(id: NodeId, color: RGBA): void {
        writeColorToBuffer(this.colors, id * 4, color);
        this.dirtyNodes = true;
    }

    /** Update node size */
    setNodeSize(id: NodeId, size: number): void {
        this.sizes[id] = size;
        this.dirtyNodes = true;
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

        if (this.freeEdges.length > 0) {
            id = this.freeEdges.pop()!;
        } else {
            id = this.edgeCount;
            if (id >= this.edgeCapacity) {
                this.growEdges();
            }
        }

        this.edgeCount = Math.max(this.edgeCount, id + 1);

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
        this.adjAdd(input.source, id);
        this.adjAdd(input.target, id);

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
        this.adjRemove(src, id);
        this.adjRemove(tgt, id);

        this.edgeIndices[id * 2] = 0;
        this.edgeIndices[id * 2 + 1] = 0;
        this.edgeWeights[id] = 0;
        this.edgeTags[id] = undefined!;
        this.edgeProperties[id] = undefined!;

        this.freeEdges.push(id);

        // Shrink edgeCount if we deleted from the tail
        while (this.edgeCount > 0 && this.edgeTags[this.edgeCount - 1] === undefined) {
            this.edgeCount--;
        }
        if (this.freeEdges.length > 0) {
            this.freeEdges = this.freeEdges.filter(fid => fid < this.edgeCount);
        }

        this.dirtyEdges = true;
    }

    /** Remove all edges connected to a node — O(degree) via adjacency index */
    removeNodeEdges(nodeId: NodeId): void {
        const adj = this.nodeAdj.get(nodeId);
        if (!adj) return;
        // Copy to array because removeEdge mutates the set
        const edgeIds = [...adj];
        for (const eid of edgeIds) {
            this.removeEdge(eid);
        }
        this.nodeAdj.delete(nodeId);
    }

    // --- Adjacency index helpers ---

    private adjAdd(nodeId: NodeId, edgeId: EdgeId): void {
        let set = this.nodeAdj.get(nodeId);
        if (!set) {
            set = new Set();
            this.nodeAdj.set(nodeId, set);
        }
        set.add(edgeId);
    }

    private adjRemove(nodeId: NodeId, edgeId: EdgeId): void {
        const set = this.nodeAdj.get(nodeId);
        if (set) {
            set.delete(edgeId);
            if (set.size === 0) this.nodeAdj.delete(nodeId);
        }
    }

    /** Get an edge view */
    getEdge(id: EdgeId): Edge | null {
        if (id < 0 || id >= this.edgeCount || !this.edgeTags[id]) return null;
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
    getNodeEdges(nodeId: NodeId): Edge[] {
        const results: Edge[] = [];
        for (let i = 0; i < this.edgeCount; i++) {
            if (this.edgeTags[i] === undefined) continue;
            const src = this.edgeIndices[i * 2];
            const tgt = this.edgeIndices[i * 2 + 1];
            if (src === nodeId || tgt === nodeId) {
                results.push(this.getEdge(i)!);
            }
        }
        return results;
    }

    /** Get all related nodes (neighbors) for a given node */
    getRelatedNodes(nodeId: NodeId): Node[] {
        const neighborIds = new Set<NodeId>();
        for (let i = 0; i < this.edgeCount; i++) {
            if (this.edgeTags[i] === undefined) continue;
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
            if (this.nodeTags[i]) tags.add(this.nodeTags[i]);
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
        this.freeNodes.length = 0;
        this.freeEdges.length = 0;
        this.positions.fill(0);
        this.colors.fill(0);
        this.sizes.fill(0);
        this.edgeIndices.fill(0);
        this.edgeWeights.fill(0);
        this.nodeTags.length = 0;
        this.nodeProperties.length = 0;
        this.edgeTags.length = 0;
        this.edgeProperties.length = 0;
        this.tagColors.clear();
        this.dirtyNodes = true;
        this.dirtyEdges = true;
    }

    // =========================================================
    // Internal: buffer growth
    // =========================================================

    private growNodes(): void {
        const newCapacity = this.nodeCapacity * GROWTH_FACTOR;
        this.positions = growBuffer(this.positions, newCapacity * 2);
        this.colors = growBuffer(this.colors, newCapacity * 4);
        this.sizes = growBuffer(this.sizes, newCapacity);
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
