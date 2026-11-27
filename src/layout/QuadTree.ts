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
// Bodies that can't be separated (identical positions, or
// MAX_DEPTH reached) share a leaf as a "bucket", chained
// through a per-body linked list. This bounds the recursion
// depth no matter what the input looks like.
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
const BODY_ID = 4;    // first body of this leaf (-1 if internal)
const MASS = 5;       // total mass of this subtree
const COM_X = 6;      // center of mass X
const COM_Y = 7;      // center of mass Y
const CX = 8;         // cell center X
const CY = 9;         // cell center Y
const HALF = 10;      // cell half-size
const NODE_SIZE = 11; // floats per tree node

// Default pool size — will grow if needed
const INITIAL_POOL = 4096;

// Max subdivision depth. Float32 positions can't be told apart
// beyond ~2^-24 of the bounding box anyway, so anything deeper
// goes into a bucket instead.
const MAX_DEPTH = 32;

// Distance (squared) below which two bodies count as coincident
const COINCIDENT_DIST_SQ = 1e-12;

// Effective distance used to push coincident bodies apart
const COINCIDENT_PUSH_DIST = 0.01;

// Distance softening: prevents division by zero for very close
// nodes. Must be tiny relative to typical inter-node distances.
const SOFTENING = 1e-8;

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
    /** Per-body link to the next body in the same leaf (-1 = end) */
    private nextBody: Int32Array = new Int32Array(0);

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
     * Bodies with size <= 0 (inactive) or non-finite positions
     * are skipped.
     *
     * @param positions  Interleaved [x0, y0, x1, y1, ...] array
     * @param sizes      Per-node sizes (used to skip inactive nodes with size <= 0)
     * @param nodeCount  Number of nodes (not array length)
     */
    build(positions: Float32Array, sizes: Float32Array, nodeCount: number): void {
        this.count = 0;
        this.root = -1;

        if (this.nextBody.length < nodeCount) {
            this.nextBody = new Int32Array(nodeCount);
        }

        // Find bounding box
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        let active = 0;

        for (let i = 0; i < nodeCount; i++) {
            if (sizes[i] <= 0) continue;
            const x = positions[i * 2];
            const y = positions[i * 2 + 1];
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            active++;
        }

        if (active === 0) return;

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
            const x = positions[i * 2];
            const y = positions[i * 2 + 1];
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
            this.insert(this.root, i, x, y, 0);
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
            const firstBody = this.pool[base + BODY_ID];

            const dx = px - comX;
            const dy = py - comY;
            const rawDistSq = dx * dx + dy * dy;
            const distSq = rawDistSq + SOFTENING;
            const dist = Math.sqrt(distSq);

            // Leaf: one body, or a bucket of bodies sharing its position
            if (firstBody >= 0) {
                for (let j = firstBody; j >= 0; j = this.nextBody[j]) {
                    if (j === nodeId) continue;

                    if (rawDistSq < COINCIDENT_DIST_SQ) {
                        // Coincident: push apart along a fixed per-pair direction
                        const [ux, uy] = coincidentDirection(nodeId, j);
                        const forceMag = G / (COINCIDENT_PUSH_DIST * COINCIDENT_PUSH_DIST);
                        fx -= ux * forceMag;
                        fy -= uy * forceMag;
                    } else {
                        const forceMag = G / distSq;
                        fx -= (dx / dist) * forceMag;
                        fy -= (dy / dist) * forceMag;
                    }
                }
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

    private insert(nodeIdx: number, bodyId: number, bx: number, by: number, depth: number): void {
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
            this.nextBody[bodyId] = -1;
            return;
        }

        // Case 2: Leaf with existing body (or bucket of bodies)
        if (existingBody >= 0) {
            const oldX = this.pool[base + COM_X];
            const oldY = this.pool[base + COM_Y];

            // Can't (or shouldn't) separate them — add to the bucket
            if (depth >= MAX_DEPTH || (oldX === bx && oldY === by)) {
                const m = this.pool[base + MASS];
                this.nextBody[bodyId] = this.nextBody[existingBody];
                this.nextBody[existingBody] = bodyId;
                this.pool[base + COM_X] = (oldX * m + bx) / (m + 1);
                this.pool[base + COM_Y] = (oldY * m + by) / (m + 1);
                this.pool[base + MASS] = m + 1;
                return;
            }

            // Subdivide: mark as internal, push the existing bodies down.
            // Below MAX_DEPTH a bucket only holds identical positions,
            // so all its bodies sit exactly at (oldX, oldY).
            this.pool[base + BODY_ID] = -1;
            this.pool[base + MASS] = 0;

            let j = existingBody;
            while (j >= 0) {
                const next = this.nextBody[j];
                this.insertIntoChild(nodeIdx, j, oldX, oldY, depth);
                j = next;
            }
        }

        // Case 3: Internal node — insert into correct child
        this.insertIntoChild(nodeIdx, bodyId, bx, by, depth);
    }

    private insertIntoChild(
        parentIdx: number,
        bodyId: number,
        bx: number,
        by: number,
        depth: number,
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

// -----------------------------------------------------------
// Helpers
// -----------------------------------------------------------

/**
 * Deterministic unit vector for a pair of coincident bodies.
 * Antisymmetric: (a, b) and (b, a) get opposite directions,
 * so the pair is pushed apart with equal and opposite forces.
 */
function coincidentDirection(a: number, b: number): [number, number] {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    let h = Math.imul(lo, 0x9E3779B1) ^ Math.imul(hi + 1, 0x85EBCA77);
    h = Math.imul(h ^ (h >>> 15), 0x2C1B3C6D);
    h ^= h >>> 13;
    const angle = ((h >>> 0) / 4294967296) * Math.PI * 2;
    const sign = a === lo ? 1 : -1;
    return [Math.cos(angle) * sign, Math.sin(angle) * sign];
}
