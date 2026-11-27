// ============================================================
// graphGPU - QuadTree benchmarks
// ============================================================

import { QuadTree } from '../src/layout/QuadTree';
import { seededRandom } from '../tests/helpers';
import type { BenchSuite } from './types';

function bodies(n: number, seed: number) {
    const rand = seededRandom(seed);
    const positions = new Float32Array(n * 2);
    for (let i = 0; i < n * 2; i++) positions[i] = rand() * 2 - 1;
    return { positions, sizes: new Float32Array(n).fill(1), n };
}

const small = bodies(10_000, 1);
const large = bodies(100_000, 2);
const tree = new QuadTree();

export default {
    name: 'QuadTree',
    cases: [
        {
            name: 'build (10k)',
            fn: () => { tree.build(small.positions, small.sizes, small.n); },
        },
        {
            name: 'build (100k)',
            fn: () => { tree.build(large.positions, large.sizes, large.n); },
        },
        {
            name: 'build + forces (10k, θ=0.5)',
            fn: () => {
                const { positions, sizes, n } = small;
                tree.build(positions, sizes, n);
                let fx = 0;
                for (let i = 0; i < n; i++) {
                    fx += tree.computeForce(i, positions[i * 2], positions[i * 2 + 1], -0.25, 0.5)[0];
                }
                if (Number.isNaN(fx)) throw new Error('NaN force');
            },
        },
    ],
} satisfies BenchSuite;
