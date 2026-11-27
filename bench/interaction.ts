// ============================================================
// graphGPU - Interaction benchmarks
// ============================================================

import { Camera } from '../src/core/Camera';
import { InteractionController } from '../src/interaction/Controls';
import { randomGraph, seededRandom } from '../tests/helpers';
import { installBrowserEnv, createCanvas } from './dom';
import type { BenchSuite } from './types';

let controls: InteractionController;
const points = new Float32Array(2000);

export default {
    name: 'Interaction',
    setup: () => {
        installBrowserEnv();
        const graph = randomGraph(100_000, 0, 5);
        const camera = new Camera(1280, 800);
        controls = new InteractionController(createCanvas(), graph, camera, undefined, 1);
        const rand = seededRandom(9);
        for (let i = 0; i < points.length; i++) points[i] = rand() * (i % 2 ? 800 : 1280);
        controls.hitTest(0, 0); // warm up (builds any index)
    },
    cases: [
        {
            name: 'hitTest ×1000 (100k nodes, static)',
            fn: () => {
                let hits = 0;
                for (let i = 0; i < points.length; i += 2) {
                    if (controls.hitTest(points[i], points[i + 1]) !== null) hits++;
                }
                if (hits < 0) throw new Error('unreachable');
            },
        },
    ],
} satisfies BenchSuite;
