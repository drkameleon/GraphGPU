// ============================================================
// graphGPU - ForceLayout benchmarks
// ============================================================

import type { Graph } from '../src/core/Graph';
import { ForceLayout } from '../src/layout/ForceLayout';
import { randomGraph } from '../tests/helpers';
import type { BenchCase, BenchSuite } from './types';

/**
 * One simulation tick on a fixed graph. Positions are restored
 * before every iteration so each sample does the same work.
 */
function tick(name: string, nodes: number, edges: number, barnesHutTheta: number): BenchCase {
    const g: Graph = randomGraph(nodes, edges, nodes);
    const initial = g.positions.slice();
    let layout: ForceLayout;

    return {
        name,
        beforeEach: () => {
            g.positions.set(initial);
            layout = new ForceLayout(g, { type: 'force', barnesHutTheta, minVelocity: 0 });
        },
        fn: () => { layout.step(1); },
    };
}

export default {
    name: 'ForceLayout',
    cases: [
        tick('step (1k/2k, Barnes-Hut)', 1_000, 2_000, 0.3),
        tick('step (10k/20k, Barnes-Hut)', 10_000, 20_000, 0.3),
        tick('step (2k/4k, brute-force)', 2_000, 4_000, 0),
    ],
} satisfies BenchSuite;
