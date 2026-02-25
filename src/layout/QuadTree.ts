// ============================================================
// graphGPU - Barnes-Hut Quadtree
// ============================================================
// Spatial index for O(n log n) N-body repulsion.
//
// Uses a flat pre-allocated node pool (no heap allocations
// during simulation). Rebuilt from scratch each tick — this
// is standard practice and is O(n), cheaper than incremental
// updates for dynamic layouts.
//
// Quadrants:
//   0 = NW (+x, +y)
//   1 = NE (-x, +y)
//   2 = SW (+x, -y)
//   3 = SE (-x, -y)
// ============================================================

// Each tree node occupies a fixed-size struct in the pool.
// Fields are accessed via offset from the node's base index.

const CHILD_0 = 0;   // index of child 0 (NW), -1 if empty
const CHILD_1 = 1;   // NE
const CHILD_2 = 2;   // SW
const CHILD_3 = 3;   // SE
const BODY_ID = 4;    // node ID stored here (-1 if internal)
const MASS = 5;       // total mass of this subtree
const COM_X = 6;      // center of mass X
const COM_Y = 7;      // center of mass Y
const CX = 8;         // cell center X
const CY = 9;         // cell center Y
const HALF = 10;      // cell half-size
const NODE_SIZE = 11; // floats per tree node

// Default pool size — will grow if needed
const INITIAL_POOL = 4096;

export class QuadTree {
    /** Flat node pool: each tree node = NODE_SIZE contiguous floats */
    private pool: Float32Array;
    /** Number of tree nodes currently allocated */
    private count: number = 0;
    /** Pool capacity (in tree nodes) */
    private capacity: number;
    /** Root node index (always 0 after build) */
    private root: number = -1;
    /** Pre-allocated stack for iterative tree walk (avoids per-call allocation) */
    private walkStack: Int32Array;

    constructor() {
        this.capacity = INITIAL_POOL;
        this.pool = new Float32Array(this.capacity * NODE_SIZE);
        this.walkStack = new Int32Array(256);
    }

    // =========================================================
    // Build
    // =========================================================

    /**
     * Build the quadtree from scratch for the given positions.
     * Called once per tick — O(n) amortized.
     *
     * @param positions  Interleaved [x0, y0, x1, y1, ...] array
     * @param sizes      Per-node sizes (used to skip inactive nodes with size <= 0)
     * @param nodeCount  Number of nodes (not array length)
     */
    build(positions: Float32Array, sizes: Float32Array, nodeCount: number): void {
        this.count = 0;

        if (nodeCount === 0) {
            this.root = -1;
            return;
        }

        // Find bounding box
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (let i = 0; i < nodeCount; i++) {
            if (sizes[i] <= 0) continue;
            const x = positions[i * 2];
            const y = positions[i * 2 + 1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        // Make it square (quadtree needs square cells)
        const w = maxX - minX;
        const h = maxY - minY;
        const size = Math.max(w, h, 0.001);
        const cx = (minX + maxX) * 0.5;
        const cy = (minY + maxY) * 0.5;
        const halfSize = size * 0.55; // slight padding

        // Allocate root
        this.root = this.allocNode(cx, cy, halfSize);

        // Insert all active nodes
        for (let i = 0; i < nodeCount; i++) {
            if (sizes[i] <= 0) continue;
            this.insert(this.root, i, positions[i * 2], positions[i * 2 + 1]);
        }

        // Compute centers of mass (bottom-up)
        this.computeMass(this.root);
    }

    // =========================================================
    // Barnes-Hut force calculation
    // =========================================================

    /**
     * Calculate the repulsive force on a single node using
     * the Barnes-Hut approximation.
     *
     * @param nodeId    The node to compute forces for
     * @param px        Node's X position
     * @param py        Node's Y position
     * @param G         Gravitational constant (negative = repulsive)
     * @param theta     Approximation parameter (0.3–0.5 typical)
     * @returns         [fx, fy] force components
     */
    computeForce(
        nodeId: number,
        px: number,
        py: number,
        G: number,
        theta: number,
    ): [number, number] {
        if (this.root === -1) return [0, 0];

        let fx = 0;
        let fy = 0;

        // Distance softening: prevents division by zero for coincident nodes.
        // Must be tiny relative to typical inter-node distances in [-1,1] space.
        const softening = 1e-8;

        // Iterative tree walk using pre-allocated stack
        let stack = this.walkStack;
        stack[0] = this.root;
        let sp = 1;

        while (sp > 0) {
            const ni = stack[--sp];
            const base = ni * NODE_SIZE;

            const mass = this.pool[base + MASS];
            if (mass === 0) continue;

            const comX = this.pool[base + COM_X];
            const comY = this.pool[base + COM_Y];
            const halfSize = this.pool[base + HALF];
            const bodyId = this.pool[base + BODY_ID];

            const dx = px - comX;
            const dy = py - comY;
            const distSq = dx * dx + dy * dy + softening;
            const dist = Math.sqrt(distSq);

            // Skip self (leaf that IS the querying node)
            if (bodyId === nodeId) continue;

            // Leaf node with a single body
            if (bodyId >= 0) {
                const forceMag = G / distSq;
                fx -= (dx / dist) * forceMag;
                fy -= (dy / dist) * forceMag;
                continue;
            }

            // Internal node: check Barnes-Hut criterion
            // If cell is far enough away relative to its size, approximate
            if ((2 * halfSize) / dist < theta) {
                const forceMag = G * mass / distSq;
                fx -= (dx / dist) * forceMag;
                fy -= (dy / dist) * forceMag;
                continue;
            }

            // Otherwise, recurse into children
            // Grow stack if needed
            if (sp + 4 >= stack.length) {
                const bigger = new Int32Array(stack.length * 2);
                bigger.set(stack);
                this.walkStack = bigger;
                stack = bigger;
            }
            const c0 = this.pool[base + CHILD_0];
            const c1 = this.pool[base + CHILD_1];
            const c2 = this.pool[base + CHILD_2];
            const c3 = this.pool[base + CHILD_3];
            if (c0 >= 0) stack[sp++] = c0;
            if (c1 >= 0) stack[sp++] = c1;
            if (c2 >= 0) stack[sp++] = c2;
            if (c3 >= 0) stack[sp++] = c3;
        }

        return [fx, fy];
    }

    // =========================================================
    // Internal: tree construction
    // =========================================================

    private allocNode(cx: number, cy: number, half: number): number {
        if (this.count >= this.capacity) {
            this.grow();
        }

        const idx = this.count++;
        const base = idx * NODE_SIZE;

        this.pool[base + CHILD_0] = -1;
        this.pool[base + CHILD_1] = -1;
        this.pool[base + CHILD_2] = -1;
        this.pool[base + CHILD_3] = -1;
        this.pool[base + BODY_ID] = -1;
        this.pool[base + MASS] = 0;
        this.pool[base + COM_X] = 0;
        this.pool[base + COM_Y] = 0;
        this.pool[base + CX] = cx;
        this.pool[base + CY] = cy;
        this.pool[base + HALF] = half;

        return idx;
    }

    private insert(nodeIdx: number, bodyId: number, bx: number, by: number, depth: number = 0): void {
        const base = nodeIdx * NODE_SIZE;
        const existingBody = this.pool[base + BODY_ID];
        const hasChildren =
            this.pool[base + CHILD_0] >= 0 || this.pool[base + CHILD_1] >= 0 ||
            this.pool[base + CHILD_2] >= 0 || this.pool[base + CHILD_3] >= 0;

        // Case 1: Empty leaf — place body here
        if (existingBody < 0 && !hasChildren) {
            this.pool[base + BODY_ID] = bodyId;
            this.pool[base + COM_X] = bx;
            this.pool[base + COM_Y] = by;
            this.pool[base + MASS] = 1;
            return;
        }

        // Max depth guard: if two bodies share (nearly) the same position,
        // stop subdividing and accumulate mass into this node.
        if (depth >= 40) {
            this.pool[base + MASS] += 1;
            return;
        }

        // Case 2: Leaf with existing body — subdivide
        if (existingBody >= 0) {
            const oldId = existingBody;
            const oldX = this.pool[base + COM_X];
            const oldY = this.pool[base + COM_Y];

            // Mark as internal
            this.pool[base + BODY_ID] = -1;
            this.pool[base + MASS] = 0;

            // Re-insert the old body
            this.insertIntoChild(nodeIdx, oldId, oldX, oldY, depth);
        }

        // Case 3: Internal node — insert into correct child
        this.insertIntoChild(nodeIdx, bodyId, bx, by, depth);
    }

    private insertIntoChild(
        parentIdx: number,
        bodyId: number,
        bx: number,
        by: number,
        depth: number = 0,
    ): void {
        const base = parentIdx * NODE_SIZE;
        const cx = this.pool[base + CX];
        const cy = this.pool[base + CY];
        const half = this.pool[base + HALF];
        const qHalf = half * 0.5;

        // Determine quadrant
        const right = bx >= cx;
        const top = by >= cy;
        let quadrant: number;
        let childCX: number;
        let childCY: number;

        if (right && top) {
            quadrant = CHILD_0; // NE
            childCX = cx + qHalf;
            childCY = cy + qHalf;
        } else if (!right && top) {
            quadrant = CHILD_1; // NW
            childCX = cx - qHalf;
            childCY = cy + qHalf;
        } else if (right && !top) {
            quadrant = CHILD_2; // SE
            childCX = cx + qHalf;
            childCY = cy - qHalf;
        } else {
            quadrant = CHILD_3; // SW
            childCX = cx - qHalf;
            childCY = cy - qHalf;
        }

        let childIdx = this.pool[base + quadrant];
        if (childIdx < 0) {
            childIdx = this.allocNode(childCX, childCY, qHalf);
            this.pool[base + quadrant] = childIdx;
        }

        this.insert(childIdx, bodyId, bx, by, depth + 1);
    }

    /**
     * Bottom-up pass to compute aggregate mass and center of mass
     * for internal nodes.
     */
    private computeMass(nodeIdx: number): void {
        const base = nodeIdx * NODE_SIZE;

        // Leaf with body — mass already set during insert
        if (this.pool[base + BODY_ID] >= 0) return;

        let totalMass = 0;
        let comX = 0;
        let comY = 0;

        for (let q = 0; q < 4; q++) {
            const childIdx = this.pool[base + CHILD_0 + q];
            if (childIdx < 0) continue;

            this.computeMass(childIdx);

            const cBase = childIdx * NODE_SIZE;
            const m = this.pool[cBase + MASS];
            comX += this.pool[cBase + COM_X] * m;
            comY += this.pool[cBase + COM_Y] * m;
            totalMass += m;
        }

        if (totalMass > 0) {
            this.pool[base + COM_X] = comX / totalMass;
            this.pool[base + COM_Y] = comY / totalMass;
        }
        this.pool[base + MASS] = totalMass;
    }

    // =========================================================
    // Pool growth
    // =========================================================

    private grow(): void {
        const newCapacity = this.capacity * 2;
        const newPool = new Float32Array(newCapacity * NODE_SIZE);
        newPool.set(this.pool);
        this.pool = newPool;
        this.capacity = newCapacity;
    }
}
