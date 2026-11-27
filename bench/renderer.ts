// ============================================================
// graphGPU - Renderer benchmarks
// ============================================================
// Per-frame CPU cost of the renderer (with a fake GPU device):
// buffer uploads, edge instance rebuilds and the label overlay.
// ============================================================

import { Renderer } from '../src/core/Renderer';
import type { Graph } from '../src/core/Graph';
import { randomGraph } from '../tests/helpers';
import { installBrowserEnv, createCanvas } from './dom';
import type { BenchCase, BenchSuite } from './types';

interface Scene {
    graph: Graph;
    renderer: Renderer;
}

const scenes: Record<string, Scene> = {};

async function scene(name: string, nodes: number, edges: number, labels: boolean): Promise<void> {
    const graph = randomGraph(nodes, edges, nodes);
    const renderer = new Renderer(graph, { canvas: createCanvas(), pixelRatio: 1 });
    await renderer.init();
    renderer.initLabelOverlay();
    renderer.setLabelsVisible(labels);
    renderer.renderFrame(0);
    scenes[name] = { graph, renderer };
}

let t = 0;

function frame(name: string, sceneName: string, change: (s: Scene) => void): BenchCase {
    return {
        name,
        beforeEach: () => change(scenes[sceneName]),
        fn: () => { scenes[sceneName].renderer.renderFrame(t += 16); },
    };
}

const positionsChanged = (s: Scene) => {
    s.graph.dirtyNodes = true;
    s.graph.dirtyEdges = true;
};

const cameraMoved = (s: Scene) => {
    s.renderer.getCamera().pan(1, 0);
};

export default {
    name: 'Renderer',
    setup: async () => {
        installBrowserEnv();
        await scene('10k', 10_000, 20_000, false);
        await scene('100k', 100_000, 200_000, false);
        await scene('2k+labels', 2_000, 4_000, true);
    },
    cases: [
        frame('frame, positions changed (10k/20k)', '10k', positionsChanged),
        frame('frame, positions changed (100k/200k)', '100k', positionsChanged),
        frame('frame, camera moved (100k/200k)', '100k', cameraMoved),
        frame('frame, labels on (2k/4k)', '2k+labels', cameraMoved),
    ],
} satisfies BenchSuite;
