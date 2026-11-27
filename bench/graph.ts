// ============================================================
// graphGPU - Graph benchmarks
// ============================================================

import { Graph } from '../src/core/Graph';
import { randomGraph, seededRandom } from '../tests/helpers';
import type { BenchSuite } from './types';

const N = 10_000;
const M = 20_000;

// Pre-generated edge endpoints, so RNG cost isn't measured
const rand = seededRandom(1);
const edgePairs = new Uint32Array(M * 2);
for (let i = 0; i < M * 2; i++) edgePairs[i] = Math.floor(rand() * N);

let g: Graph;
const queryGraph = randomGraph(N, M, 2);

export default {
    name: 'Graph',
    cases: [
        {
            name: 'addNode ×10k',
            fn: () => {
                const graph = new Graph();
                for (let i = 0; i < N; i++) graph.addNode({ tag: 'node' });
            },
        },
        {
            name: 'addEdge ×20k',
            beforeEach: () => { g = randomGraph(N, 0); },
            fn: () => {
                for (let i = 0; i < M; i++) {
                    g.addEdge({ tag: 'edge', source: edgePairs[i * 2], target: edgePairs[i * 2 + 1] });
                }
            },
        },
        {
            name: 'removeNode ×2k (of 4k)',
            beforeEach: () => { g = randomGraph(4_000, 8_000); },
            fn: () => {
                for (let i = 0; i < 4_000; i += 2) g.removeNode(i);
            },
        },
        {
            name: 'getNodeEdges ×100 (10k/20k)',
            fn: () => {
                for (let i = 0; i < 100; i++) queryGraph.getNodeEdges(i);
            },
        },
        {
            name: 'getRelatedNodes ×100 (10k/20k)',
            fn: () => {
                for (let i = 0; i < 100; i++) queryGraph.getRelatedNodes(i);
            },
        },
        {
            name: 'fetchByTag (10k)',
            fn: () => { queryGraph.fetchByTag('t3'); },
        },
        {
            name: 'iterate activeNodeIds (10k)',
            fn: () => {
                let sum = 0;
                for (const id of queryGraph.activeNodeIds()) sum += id;
                if (sum < 0) throw new Error('unreachable');
            },
        },
    ],
} satisfies BenchSuite;
