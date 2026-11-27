// ============================================================
// graphGPU - Spatial Index (node picking)
// ============================================================
// Uniform grid over node centers in world space. Cells are at
// least one node diameter wide, so a point can only hit nodes
// whose centers lie in its own or the 8 neighboring cells.
//
// Rebuilt from scratch (O(n), allocation-free once warm) only
// when nodes changed; queries are O(1) on average.
// ============================================================

import type { Graph } from '../core/Graph';
import type { NodeId } from '../types';

/** Upper bound on grid cells, relative to node count */
const CELLS_PER_NODE = 2;

export class NodeGrid {
    private cellStart: Int32Array = new Int32Array(0);
    private items: Int32Array = new Int32Array(0);
    private cellOf: Int32Array = new Int32Array(0);
    private cols = 0;
    private rows = 0;
    private cell = 1;
    private minX = 0;
    private minY = 0;
    private radiusScale = 0;
    private graph: Graph | null = null;

    /**
     * Index the active nodes of `graph`. A node's hit radius is
     * `size * radiusScale` (world units).
     */
    build(graph: Graph, radiusScale: number): void {
        this.graph = graph;
        this.radiusScale = radiusScale;

        const slots = graph.nodeSlots;
        const pos = graph.positions;
        const sizes = graph.sizes;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        let maxSize = 0;
        let count = 0;
        for (let id = 0; id < slots; id++) {
            if (!graph.isNodeActive(id)) continue;
            const x = pos[id * 2], y = pos[id * 2 + 1];
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (sizes[id] > maxSize) maxSize = sizes[id];
            count++;
        }

        if (count === 0) {
            this.cols = this.rows = 0;
            return;
        }

        // At least one diameter per cell, and not more cells than needed
        const w = Math.max(maxX - minX, 1e-9);
        const h = Math.max(maxY - minY, 1e-9);
        let cell = Math.max(2 * maxSize * radiusScale, 1e-9);
        const maxCells = Math.max(1, count * CELLS_PER_NODE);
        if ((w / cell + 1) * (h / cell + 1) > maxCells) {
            cell = Math.max(cell, Math.sqrt((w * h) / maxCells));
            while ((Math.floor(w / cell) + 1) * (Math.floor(h / cell) + 1) > maxCells) cell *= 1.5;
        }

        const cols = Math.floor(w / cell) + 1;
        const rows = Math.floor(h / cell) + 1;
        const cells = cols * rows;
        this.cols = cols;
        this.rows = rows;
        this.cell = cell;
        this.minX = minX;
        this.minY = minY;

        if (this.cellStart.length < cells + 1) this.cellStart = new Int32Array((cells + 1) * 2);
        if (this.items.length < count) this.items = new Int32Array(count * 2);
        if (this.cellOf.length < slots) this.cellOf = new Int32Array(slots * 2);
        const start = this.cellStart;
        const cellOf = this.cellOf;
        start.fill(0, 0, cells + 1);

        // Count per cell, prefix-sum, then fill (CSR)
        for (let id = 0; id < slots; id++) {
            cellOf[id] = -1;
            if (!graph.isNodeActive(id)) continue;
            const x = pos[id * 2], y = pos[id * 2 + 1];
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
            const c = Math.floor((y - minY) / cell) * cols + Math.floor((x - minX) / cell);
            cellOf[id] = c;
            start[c + 1]++;
        }
        for (let c = 0; c < cells; c++) start[c + 1] += start[c];
        const fill = start.slice(0, cells);
        for (let id = 0; id < slots; id++) {
            const c = cellOf[id];
            if (c >= 0) this.items[fill[c]++] = id;
        }
    }

    /**
     * The node whose hit circle contains (x, y), closest center first
     * (lowest id on ties), or null.
     */
    query(x: number, y: number): NodeId | null {
        const graph = this.graph;
        if (!graph || this.cols === 0) return null;

        const pos = graph.positions;
        const sizes = graph.sizes;
        const cx = Math.floor((x - this.minX) / this.cell);
        const cy = Math.floor((y - this.minY) / this.cell);

        let best: NodeId | null = null;
        let bestDist = Infinity;

        for (let gy = Math.max(0, cy - 1); gy <= Math.min(this.rows - 1, cy + 1); gy++) {
            for (let gx = Math.max(0, cx - 1); gx <= Math.min(this.cols - 1, cx + 1); gx++) {
                const c = gy * this.cols + gx;
                for (let k = this.cellStart[c], end = this.cellStart[c + 1]; k < end; k++) {
                    const id = this.items[k];
                    const dx = x - pos[id * 2];
                    const dy = y - pos[id * 2 + 1];
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist >= sizes[id] * this.radiusScale) continue;
                    if (dist < bestDist || (dist === bestDist && best !== null && id < best)) {
                        bestDist = dist;
                        best = id;
                    }
                }
            }
        }
        return best;
    }
}
