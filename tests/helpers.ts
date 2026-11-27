// ============================================================
// graphGPU - Test Helpers
// ============================================================

import { Graph } from '../src/core/Graph';

/** Deterministic PRNG (mulberry32) so tests never flake */
export function seededRandom(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Build a random graph with deterministic positions and edges */
export function randomGraph(nodes: number, edges: number, seed: number = 1): Graph {
    const rand = seededRandom(seed);
    const g = new Graph({ maxNodes: nodes, maxEdges: edges });
    for (let i = 0; i < nodes; i++) {
        g.addNode({
            tag: `t${i % 5}`,
            x: rand() * 2 - 1,
            y: rand() * 2 - 1,
        });
    }
    for (let i = 0; i < edges; i++) {
        g.addEdge({
            tag: 'e',
            source: Math.floor(rand() * nodes),
            target: Math.floor(rand() * nodes),
        });
    }
    return g;
}

/** Euclidean distance between two nodes */
export function nodeDistance(g: Graph, a: number, b: number): number {
    const p = g.positions;
    return Math.hypot(p[a * 2] - p[b * 2], p[a * 2 + 1] - p[b * 2 + 1]);
}
