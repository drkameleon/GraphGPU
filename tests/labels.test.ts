// @vitest-environment happy-dom
// ============================================================
// graphGPU - Label overlay tests
// ============================================================
// Records every label the overlay draws (text, position, angle,
// font, color) and compares against a snapshot, so performance
// work on the overlay can't change what ends up on screen.
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Graph } from '../src/core/Graph';
import { Renderer } from '../src/core/Renderer';
import { installFakeWebGPU } from './fake-gpu';
import { seededRandom } from './helpers';

// -----------------------------------------------------------
// A 2D context that records drawn text
// -----------------------------------------------------------

interface DrawnText {
    text: string;
    x: number;
    y: number;
    angle: number;
    font: string;
    fill: string;
}

let drawn: DrawnText[] = [];

function recordingContext() {
    let tx = 0, ty = 0, angle = 0;
    const stack: [number, number, number][] = [];
    const ctx = {
        font: '10px sans-serif',
        fillStyle: '#000',
        textAlign: 'start',
        textBaseline: 'alphabetic',
        save() { stack.push([tx, ty, angle]); },
        restore() { [tx, ty, angle] = stack.pop() ?? [0, 0, 0]; },
        scale() {},
        clearRect() {},
        translate(x: number, y: number) {
            tx += x * Math.cos(angle) - y * Math.sin(angle);
            ty += x * Math.sin(angle) + y * Math.cos(angle);
        },
        rotate(a: number) { angle += a; },
        measureText(text: string) {
            const size = parseFloat(/([\d.]+)px/.exec(ctx.font)?.[1] ?? '10');
            return { width: text.length * size * 0.55 };
        },
        fillText(text: string, x: number, y: number) {
            const r = (v: number) => Math.round(v * 100) / 100;
            drawn.push({
                text,
                x: r(tx + x * Math.cos(angle) - y * Math.sin(angle)),
                y: r(ty + x * Math.sin(angle) + y * Math.cos(angle)),
                angle: r(angle),
                font: ctx.font,
                fill: String(ctx.fillStyle),
            });
        },
    };
    return ctx;
}

// -----------------------------------------------------------
// Scenes
// -----------------------------------------------------------

const W = 1000;
const H = 700;

async function scene(
    nodes: number, edges: number, seed: number, nodeSize: number,
    labels?: import('../src/types').LabelOptions,
) {
    const rand = seededRandom(seed);
    const graph = new Graph();
    const words = ['alpha', 'beta', 'gamma', 'a fairly long node label', 'δέλτα', 'x'];
    for (let i = 0; i < nodes; i++) {
        graph.addNode({
            tag: `t${i % 4}`,
            properties: i % 7 === 0 ? {} : { name: `${words[i % words.length]} ${i}` },
            x: rand() * 2 - 1,
            y: rand() * 2 - 1,
            size: 3 + (i % 5) * 2,
        });
    }
    for (let i = 0; i < edges; i++) {
        graph.addEdge({
            tag: i % 9 === 0 ? '' : ['knows', 'works with', 'r'][i % 3],
            source: Math.floor(rand() * nodes),
            target: Math.floor(rand() * nodes),
        });
    }

    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
        left: 0, top: 0, right: W, bottom: H, width: W, height: H, x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect);
    Object.defineProperty(canvas, 'clientWidth', { value: W });
    Object.defineProperty(canvas, 'clientHeight', { value: H });

    installFakeWebGPU(canvas);
    const renderer = new Renderer(graph, { canvas, pixelRatio: 1, nodeSize, labels });
    await renderer.init();
    renderer.initLabelOverlay();
    return { graph, renderer };
}

beforeEach(() => {
    drawn = [];
    const original = HTMLCanvasElement.prototype.getContext;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
        this: HTMLCanvasElement, type: string, ...args: unknown[]
    ) {
        if (type === '2d') return recordingContext() as unknown as CanvasRenderingContext2D;
        return (original as (...a: unknown[]) => unknown).call(this, type, ...args) as RenderingContext;
    } as typeof HTMLCanvasElement.prototype.getContext);
});

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
});

// -----------------------------------------------------------
// Tests
// -----------------------------------------------------------

describe('Label overlay', () => {
    const cases: [string, number, number, number, number, number][] = [
        // name, nodes, edges, seed, nodeSize, zoom
        ['default zoom', 120, 200, 1, 1, 1],
        ['zoomed in', 120, 200, 2, 1, 4],
        ['zoomed in more, off-center', 200, 300, 3, 1, 9],
        ['big nodes', 60, 90, 4, 6, 1],
        ['zoomed out (no labels)', 120, 200, 5, 1, 0.2],
    ];

    for (const [name, nodes, edges, seed, nodeSize, zoom] of cases) {
        it(`draws the same labels: ${name}`, async () => {
            const { renderer } = await scene(nodes, edges, seed, nodeSize);
            const cam = renderer.getCamera();
            cam.setZoom(zoom);
            if (name.includes('off-center')) cam.setPosition(0.3, -0.2);
            renderer.renderFrame(16);
            expect(drawn).toMatchSnapshot();
            renderer.destroy();
        });
    }

    it('draws nothing when labels are hidden', async () => {
        const { renderer } = await scene(50, 80, 6, 1);
        renderer.setLabelsVisible(false);
        renderer.renderFrame(16);
        expect(drawn).toEqual([]);
        renderer.destroy();
    });

    it('follows node movement and removal', async () => {
        const { graph, renderer } = await scene(40, 60, 7, 1);
        renderer.getCamera().setZoom(4);
        renderer.renderFrame(16);
        const first = drawn.length;
        expect(first).toBeGreaterThan(0);

        drawn = [];
        for (let i = 0; i < 20; i++) graph.removeNode(i);
        graph.setNodePosition(25, 0, 0);
        renderer.renderFrame(32);
        expect(drawn).toMatchSnapshot();
        renderer.destroy();
    });
});

describe('Label options', () => {
    async function nodeLabels(labels: import('../src/types').LabelOptions, zoom = 4) {
        drawn = [];
        const { renderer } = await scene(60, 0, 8, 1, labels);
        renderer.getCamera().setZoom(zoom);
        renderer.renderFrame(16);
        renderer.destroy();
        return [...drawn];
    }

    it('uses the given property as label', async () => {
        const texts = (await nodeLabels({ property: 'name' })).map(d => d.text);
        const byTag = (await nodeLabels({ property: 'nope' })).map(d => d.text);
        expect(texts.length).toBeGreaterThan(0);
        expect(byTag).toEqual([]); // no such property: no labels
    });

    it('truncates to maxLength', async () => {
        const texts = (await nodeLabels({ maxLength: 4 })).map(d => d.text);
        expect(texts.length).toBeGreaterThan(0);
        for (const t of texts) expect(t.replace('…', '').length).toBeLessThanOrEqual(4);
    });

    it('uses the given font family and color', async () => {
        const labels = await nodeLabels({ font: 'Inter, sans-serif', color: '#ff0000' });
        expect(labels.length).toBeGreaterThan(0);
        for (const l of labels) {
            expect(l.font).toMatch(/px Inter, sans-serif$/);
            expect(l.fill).toBe('#ff0000');
        }
        const rgb = await nodeLabels({ color: [0, 0.5, 1] });
        expect(rgb[0].fill).toBe('rgba(0,128,255,1)');
    });

    it('starts hidden with enabled: false', async () => {
        expect(await nodeLabels({ enabled: false })).toEqual([]);
    });

    it('hides labels below visibleAtZoom', async () => {
        expect(await nodeLabels({ visibleAtZoom: 5 }, 4)).toEqual([]);
        expect((await nodeLabels({ visibleAtZoom: 5 }, 6)).length).toBeGreaterThan(0);
    });
});
