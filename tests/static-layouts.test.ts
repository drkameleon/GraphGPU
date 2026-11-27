// ============================================================
// graphGPU - Static layouts tests
// ============================================================

import { describe, it, expect } from 'vitest';
import { Graph } from '../src/core/Graph';
import { circularLayout, gridLayout } from '../src/layout/StaticLayouts';
import { nodeDistance } from './helpers';

function graphOf(n: number, tags = ['a', 'b']) {
    const g = new Graph();
    for (let i = 0; i < n; i++) g.addNode({ tag: tags[i % tags.length] });
    return g;
}

describe('circularLayout', () => {
    it('puts every node at the same distance from the center', () => {
        const g = graphOf(40);
        circularLayout(g, { center: [1, -1] });
        const r = Math.hypot(g.positions[0] - 1, g.positions[1] + 1);
        for (const id of g.activeNodeIds()) {
            expect(Math.hypot(g.positions[id * 2] - 1, g.positions[id * 2 + 1] + 1)).toBeCloseTo(r);
        }
    });

    it('keeps neighbors at least `spacing` apart', () => {
        const g = graphOf(500);
        circularLayout(g, { spacing: 0.05 });
        let minDist = Infinity;
        const ids = [...g.activeNodeIds()];
        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < ids.length; j++) minDist = Math.min(minDist, nodeDistance(g, ids[i], ids[j]));
        }
        expect(minDist).toBeGreaterThanOrEqual(0.05 * 0.99);
    });

    it('groups nodes by tag', () => {
        const g = graphOf(10); // a, b, a, b, ...
        circularLayout(g);
        // All 'a' nodes come first around the circle: the first 5 slots
        const angleOf = (id: number) => (Math.atan2(g.positions[id * 2 + 1], g.positions[id * 2]) + Math.PI * 2.5) % (Math.PI * 2);
        const aAngles = [0, 2, 4, 6, 8].map(angleOf);
        const bAngles = [1, 3, 5, 7, 9].map(angleOf);
        expect(Math.max(...aAngles)).toBeLessThan(Math.min(...bAngles));
    });

    it('handles a single node and an empty graph', () => {
        const g = graphOf(1);
        circularLayout(g);
        expect([g.positions[0], g.positions[1]]).toEqual([0, 0]);
        expect(() => circularLayout(new Graph())).not.toThrow();
    });
});

describe('gridLayout', () => {
    it('lays nodes out on a square grid, centered', () => {
        const g = graphOf(9, ['a']);
        gridLayout(g, { spacing: 1 });
        const coords = [...g.activeNodeIds()].map(id => [g.positions[id * 2], g.positions[id * 2 + 1]]);
        expect(coords).toEqual([
            [-1, 1], [0, 1], [1, 1],
            [-1, 0], [0, 0], [1, 0],
            [-1, -1], [0, -1], [1, -1],
        ]);
    });

    it('skips removed nodes and marks the graph dirty', () => {
        const g = graphOf(5, ['a']);
        g.removeNode(2);
        g.dirtyNodes = false;
        gridLayout(g, { spacing: 1 });
        expect(g.dirtyNodes).toBe(true);
        expect(nodeDistance(g, 0, 1)).toBeCloseTo(1);
        expect([g.positions[4], g.positions[5]]).toEqual([0, 0]); // removed slot untouched
    });
});
