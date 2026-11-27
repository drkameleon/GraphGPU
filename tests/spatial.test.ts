// ============================================================
// graphGPU - Spatial index tests
// ============================================================

import { describe, it, expect } from 'vitest';
import { Graph } from '../src/core/Graph';
import { NodeGrid } from '../src/interaction/SpatialIndex';
import { seededRandom } from './helpers';

/** Reference: the old brute-force hit test */
function bruteForce(g: Graph, x: number, y: number, radiusScale: number): number | null {
    let best: number | null = null;
    let bestDist = Infinity;
    for (const id of g.activeNodeIds()) {
        const dist = Math.hypot(x - g.positions[id * 2], y - g.positions[id * 2 + 1]);
        if (dist < g.sizes[id] * radiusScale && dist < bestDist) {
            bestDist = dist;
            best = id;
        }
    }
    return best;
}

function randomGraph(n: number, seed: number, spread: number) {
    const rand = seededRandom(seed);
    const g = new Graph();
    for (let i = 0; i < n; i++) {
        g.addNode({ tag: 'n', x: (rand() * 2 - 1) * spread, y: (rand() * 2 - 1) * spread, size: 1 + rand() * 9 });
    }
    return { g, rand };
}

describe('NodeGrid', () => {
    for (const [n, spread, scale] of [[2000, 1, 0.012], [500, 50, 0.05], [3000, 0.2, 0.002], [50, 1, 1]]) {
        it(`matches brute force (${n} nodes, spread ${spread}, radius scale ${scale})`, () => {
            const { g, rand } = randomGraph(n, n, spread);
            for (let i = 0; i < n; i += 7) g.removeNode(i);
            const grid = new NodeGrid();
            grid.build(g, scale);

            let hits = 0;
            for (let q = 0; q < 3000; q++) {
                const x = (rand() * 2.2 - 1.1) * spread;
                const y = (rand() * 2.2 - 1.1) * spread;
                const expected = bruteForce(g, x, y, scale);
                expect(grid.query(x, y)).toBe(expected);
                if (expected !== null) hits++;
            }
            expect(hits).toBeGreaterThan(0);
        });
    }

    it('hits nodes exactly at their centers, even when stacked', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n', x: 0.5, y: 0.5 });
        g.addNode({ tag: 'n', x: 0.5, y: 0.5 });
        const grid = new NodeGrid();
        grid.build(g, 0.01);
        expect(grid.query(0.5, 0.5)).toBe(a); // lowest id on ties
    });

    it('handles empty graphs and far outliers', () => {
        const grid = new NodeGrid();
        grid.build(new Graph(), 0.01);
        expect(grid.query(0, 0)).toBeNull();

        const g = new Graph();
        g.addNode({ tag: 'n', x: 0, y: 0 });
        const far = g.addNode({ tag: 'n', x: 1e6, y: -1e6 });
        grid.build(g, 0.01);
        expect(grid.query(1e6, -1e6)).toBe(far);
        expect(grid.query(0, 0)).toBe(0);
    });

    it('skips non-finite positions', () => {
        const g = new Graph();
        g.addNode({ tag: 'n', x: NaN, y: 0 });
        const b = g.addNode({ tag: 'n', x: 0.2, y: 0.2 });
        const grid = new NodeGrid();
        grid.build(g, 0.01);
        expect(grid.query(0.2, 0.2)).toBe(b);
    });
});
