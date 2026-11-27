// @vitest-environment happy-dom
// ============================================================
// graphGPU - InteractionController tests
// ============================================================

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Graph } from '../src/core/Graph';
import { Camera } from '../src/core/Camera';
import { InteractionController } from '../src/interaction/Controls';
import type { ForceLayout } from '../src/layout/ForceLayout';
import type { GraphGPUEventType } from '../src/types';

const W = 800;
const H = 600;

// World (0,0) → screen (400,300); one world unit = 300px
let canvas: HTMLCanvasElement;
let graph: Graph;
let camera: Camera;
let controls: InteractionController;
let a: number;
let b: number;

function fire(
    type: string,
    x: number,
    y: number,
    opts: { pointerId?: number; pointerType?: string; button?: number; shiftKey?: boolean } = {},
): void {
    canvas.dispatchEvent(new PointerEvent(type, {
        clientX: x,
        clientY: y,
        pointerId: opts.pointerId ?? 1,
        pointerType: opts.pointerType ?? 'mouse',
        button: opts.button ?? 0,
        shiftKey: opts.shiftKey ?? false,
        bubbles: true,
        cancelable: true,
    }));
}

function click(x: number, y: number, opts: { shiftKey?: boolean } = {}): void {
    fire('pointerdown', x, y, opts);
    fire('pointerup', x, y, opts);
}

function record(...types: GraphGPUEventType[]): GraphGPUEventType[] {
    const log: GraphGPUEventType[] = [];
    for (const t of types) controls.on(t, () => log.push(t));
    return log;
}

beforeEach(() => {
    Object.defineProperty(window, 'devicePixelRatio', { value: 1, configurable: true });
    canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    document.body.appendChild(canvas);

    graph = new Graph();
    a = graph.addNode({ tag: 'n', x: 0, y: 0 });
    b = graph.addNode({ tag: 'n', x: 1, y: 0 });
    camera = new Camera(W, H);
    controls = new InteractionController(canvas, graph, camera);
});

afterEach(() => {
    controls.destroy();
    canvas.remove();
});

// -----------------------------------------------------------
// Selection
// -----------------------------------------------------------

describe('InteractionController: selection', () => {
    it('selects a node on click', () => {
        const log = record('node:select', 'node:click');
        click(400, 300);
        expect(log).toEqual(['node:select', 'node:click']);
        expect([...controls.getSelectedNodes()]).toEqual([a]);
    });

    it('replaces the selection on a plain click', () => {
        click(400, 300);
        const log = record('node:select', 'node:deselect');
        click(700, 300);
        expect(log).toEqual(['node:deselect', 'node:select']);
        expect([...controls.getSelectedNodes()]).toEqual([b]);
    });

    it('toggles nodes with shift-click', () => {
        click(400, 300);
        click(700, 300, { shiftKey: true });
        expect([...controls.getSelectedNodes()].sort()).toEqual([a, b]);
        click(400, 300, { shiftKey: true });
        expect([...controls.getSelectedNodes()]).toEqual([b]);
    });

    it('deselects everything when clicking empty canvas', () => {
        click(400, 300);
        const log = record('node:deselect', 'canvas:click');
        click(50, 50);
        expect(log).toEqual(['node:deselect', 'canvas:click']);
        expect(controls.getSelectedNodes().size).toBe(0);
    });

    it('ignores non-primary mouse buttons', () => {
        const log = record('node:select', 'node:click', 'canvas:click');
        fire('pointerdown', 400, 300, { button: 2 });
        fire('pointerup', 400, 300, { button: 2 });
        expect(log).toEqual([]);
    });
});

// -----------------------------------------------------------
// Dragging & panning
// -----------------------------------------------------------

describe('InteractionController: dragging', () => {
    it('drags a node with the cursor', () => {
        const log = record('node:dragstart', 'node:drag', 'node:dragend', 'node:click', 'node:select');
        fire('pointerdown', 400, 300);
        fire('pointermove', 430, 300);
        fire('pointermove', 460, 240);
        fire('pointerup', 460, 240);

        // 60px right, 60px up = 0.2 world units each
        expect(graph.positions[a * 2]).toBeCloseTo(0.2);
        expect(graph.positions[a * 2 + 1]).toBeCloseTo(0.2);
        expect(log).toEqual(['node:dragstart', 'node:drag', 'node:drag', 'node:dragend']);
    });

    it('treats tiny movements as a click, without moving the node', () => {
        const log = record('node:drag', 'node:click');
        fire('pointerdown', 400, 300);
        fire('pointermove', 402, 301);
        fire('pointerup', 402, 301);
        expect(graph.positions[a * 2]).toBe(0);
        expect(log).toEqual(['node:click']);
    });

    it('pans the camera when dragging empty space', () => {
        const log = record('canvas:pan', 'canvas:click');
        fire('pointerdown', 100, 100);
        fire('pointermove', 150, 100);
        fire('pointerup', 150, 100);

        const [sx, sy] = camera.worldToScreen(0, 0);
        expect(sx).toBeCloseTo(450);
        expect(sy).toBeCloseTo(300);
        expect(log).toContain('canvas:pan');
        expect(log).not.toContain('canvas:click');
    });

    it('captures the pointer, so releasing outside the canvas still ends the drag', () => {
        const capture = vi.spyOn(canvas, 'setPointerCapture');
        fire('pointerdown', 400, 300, { pointerId: 7 });
        expect(capture).toHaveBeenCalledWith(7);
    });

    it('ends a drag on pointercancel, without clicking', () => {
        const log = record('node:dragend', 'node:click');
        fire('pointerdown', 400, 300);
        fire('pointermove', 450, 300);
        fire('pointercancel', 450, 300);
        const x = graph.positions[a * 2];

        fire('pointermove', 500, 300);
        expect(graph.positions[a * 2]).toBe(x);
        expect(log).toEqual(['node:dragend']);
    });

    it('pins the dragged node in animated mode and unpins it after', () => {
        const layout = { pin: vi.fn(), unpin: vi.fn() };
        controls.setAnimatedMode(true, layout as unknown as ForceLayout);
        fire('pointerdown', 400, 300);
        expect(layout.pin).toHaveBeenCalledWith(a);
        fire('pointermove', 450, 300);
        fire('pointerup', 450, 300);
        expect(layout.unpin).toHaveBeenCalledWith(a);
    });

    it('ignores moves from other pointers during a drag', () => {
        fire('pointerdown', 400, 300, { pointerId: 1 });
        fire('pointermove', 450, 300, { pointerId: 5, pointerType: 'pen' });
        fire('pointerup', 400, 300, { pointerId: 1 });
        expect(graph.positions[a * 2]).toBe(0);
    });
});

// -----------------------------------------------------------
// Hover
// -----------------------------------------------------------

describe('InteractionController: hover', () => {
    it('emits hover and hoverout', () => {
        const log = record('node:hover', 'node:hoverout');
        fire('pointermove', 400, 300);
        fire('pointermove', 405, 300);
        fire('pointermove', 50, 50);
        expect(log).toEqual(['node:hover', 'node:hoverout']);
        expect(controls.getHoveredNode()).toBeNull();
    });

    it('emits hoverout when the pointer leaves the canvas', () => {
        const log = record('node:hoverout');
        fire('pointermove', 400, 300);
        fire('pointerleave', 400, 300);
        expect(log).toEqual(['node:hoverout']);
    });

    it('does not hover with touch', () => {
        const log = record('node:hover');
        fire('pointermove', 400, 300, { pointerType: 'touch' });
        expect(log).toEqual([]);
    });
});

// -----------------------------------------------------------
// Pinch-zoom
// -----------------------------------------------------------

describe('InteractionController: pinch', () => {
    it('zooms in when two fingers spread apart', () => {
        const touch = { pointerType: 'touch' };
        fire('pointerdown', 300, 300, { ...touch, pointerId: 1 });
        fire('pointerdown', 500, 300, { ...touch, pointerId: 2 });
        fire('pointermove', 250, 300, { ...touch, pointerId: 1 });
        fire('pointermove', 550, 300, { ...touch, pointerId: 2 });
        expect(camera.getZoom()).toBeGreaterThan(1);

        fire('pointerup', 250, 300, { ...touch, pointerId: 1 });
        fire('pointerup', 550, 300, { ...touch, pointerId: 2 });
    });

    it('cancels a node drag (and unpins it) when a second finger lands', () => {
        const layout = { pin: vi.fn(), unpin: vi.fn() };
        controls.setAnimatedMode(true, layout as unknown as ForceLayout);
        const log = record('node:click', 'node:select');

        const touch = { pointerType: 'touch' };
        fire('pointerdown', 400, 300, { ...touch, pointerId: 1 });
        fire('pointerdown', 600, 300, { ...touch, pointerId: 2 });
        expect(layout.unpin).toHaveBeenCalledWith(a);

        fire('pointerup', 400, 300, { ...touch, pointerId: 1 });
        fire('pointerup', 600, 300, { ...touch, pointerId: 2 });
        expect(log).toEqual([]);
    });
});

// -----------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------

describe('InteractionController: lifecycle', () => {
    it('stops reacting after destroy', () => {
        const log = record('node:click');
        controls.destroy();
        click(400, 300);
        expect(log).toEqual([]);
    });
});

// -----------------------------------------------------------
// Coordinate scale & node size
// -----------------------------------------------------------

describe('InteractionController: hit testing scale', () => {
    function layoutCanvas(cssW: number, cssH: number, pixelW: number, pixelH: number): void {
        canvas.width = pixelW;
        canvas.height = pixelH;
        camera.resize(pixelW, pixelH);
        vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
            left: 0, top: 0, right: cssW, bottom: cssH,
            width: cssW, height: cssH, x: 0, y: 0,
            toJSON: () => ({}),
        } as DOMRect);
    }

    it('maps CSS pixels to canvas pixels using the canvas size, not devicePixelRatio', () => {
        // Rendered at 2x while the window reports 1x (e.g. options.pixelRatio = 2)
        layoutCanvas(400, 300, 800, 600);
        const log = record('node:select');

        // Node b sits at canvas pixel (700, 300) → CSS pixel (350, 150)
        click(350, 150);
        expect(log).toEqual(['node:select']);
        expect([...controls.getSelectedNodes()]).toEqual([b]);
    });

    it('pans by the right amount on a high-density canvas', () => {
        layoutCanvas(400, 300, 800, 600);
        fire('pointerdown', 20, 20);
        fire('pointermove', 70, 20);
        fire('pointerup', 70, 20);
        // 50 CSS px = 100 canvas px
        const [sx] = camera.worldToScreen(0, 0);
        expect(sx).toBeCloseTo(500);
    });

    it('uses the current node scale for hit testing', () => {
        // ~60px from node a: inside the default radius (~130px)
        click(460, 300);
        expect([...controls.getSelectedNodes()]).toEqual([a]);

        // Shrink nodes: radius ~22px, now it's a miss
        controls.setNodeScale(1);
        const log = record('canvas:click');
        click(460, 300);
        expect(log).toEqual(['canvas:click']);
    });
});

// -----------------------------------------------------------
// Wheel zoom
// -----------------------------------------------------------

describe('InteractionController: wheel zoom', () => {
    // happy-dom's WheelEvent drops the MouseEvent fields, so set them by hand
    function wheel(
        deltaY: number,
        opts: { deltaMode?: number; ctrlKey?: boolean; x?: number; y?: number } = {},
    ): void {
        const e = new WheelEvent('wheel', {
            deltaY,
            deltaMode: opts.deltaMode ?? 0,
            bubbles: true,
            cancelable: true,
        });
        Object.defineProperties(e, {
            clientX: { value: opts.x ?? 400 },
            clientY: { value: opts.y ?? 300 },
            ctrlKey: { value: opts.ctrlKey ?? false },
        });
        canvas.dispatchEvent(e);
    }

    it('zooms in on scroll up and out on scroll down', () => {
        wheel(-100);
        expect(camera.getZoom()).toBeGreaterThan(1);
        camera.reset();
        wheel(100);
        expect(camera.getZoom()).toBeLessThan(1);
    });

    it('zooms proportionally to the scroll amount', () => {
        // Many tiny trackpad events ≈ one big mouse-wheel event
        for (let i = 0; i < 50; i++) wheel(-2);
        const trackpad = camera.getZoom();
        camera.reset();
        wheel(-100);
        expect(trackpad).toBeCloseTo(camera.getZoom(), 5);

        // ...and a tiny event zooms only a tiny bit
        camera.reset();
        wheel(-2);
        expect(camera.getZoom()).toBeLessThan(1.01);
    });

    it('normalizes line-based deltas', () => {
        wheel(-3, { deltaMode: 1 });
        const lines = camera.getZoom();
        camera.reset();
        wheel(-48);
        expect(lines).toBeCloseTo(camera.getZoom(), 5);
    });

    it('zooms faster for trackpad pinch (ctrl + wheel)', () => {
        wheel(-10, { ctrlKey: true });
        const pinch = camera.getZoom();
        camera.reset();
        wheel(-10);
        expect(pinch).toBeGreaterThan(camera.getZoom());
    });

    it('caps the zoom of a single huge event', () => {
        wheel(-100000);
        expect(camera.getZoom()).toBeLessThanOrEqual(2);
    });

    it('zooms toward the cursor', () => {
        wheel(-200, { x: 700, y: 300 });
        const [sx, sy] = camera.worldToScreen(1, 0);
        expect(sx).toBeCloseTo(700);
        expect(sy).toBeCloseTo(300);
    });

    it('pinches by the actual finger distance ratio', () => {
        const touch = { pointerType: 'touch' };
        fire('pointerdown', 300, 300, { ...touch, pointerId: 1 });
        fire('pointerdown', 500, 300, { ...touch, pointerId: 2 });
        fire('pointermove', 200, 300, { ...touch, pointerId: 1 });
        fire('pointermove', 600, 300, { ...touch, pointerId: 2 });
        // 200px → 300px → 400px apart: 2× in total
        expect(camera.getZoom()).toBeCloseTo(2);
    });
});

// -----------------------------------------------------------
// Picking after changes & edge events
// -----------------------------------------------------------

describe('InteractionController: picking', () => {
    it('picks nodes at their new position after they move', () => {
        graph.setNodePosition(a, -1, -0.5); // screen (100, 450)
        const log = record('node:select');
        click(100, 450);
        expect(log).toEqual(['node:select']);
        expect([...controls.getSelectedNodes()]).toEqual([a]);
    });

    it('picks newly added nodes and ignores removed ones', () => {
        const c = graph.addNode({ tag: 'n', x: -1, y: 0.5 }); // screen (100, 150)
        click(100, 150);
        expect([...controls.getSelectedNodes()]).toEqual([c]);

        graph.removeNode(c);
        const log = record('canvas:click');
        click(100, 150);
        expect(log).toEqual(['canvas:click']);
    });
});

describe('InteractionController: edge events', () => {
    // Edge from a (400,300) to b (700,300); its midpoint (550,300) is
    // outside both node circles once nodes are small
    beforeEach(() => {
        controls.setNodeScale(1);
        graph.addEdge({ tag: 'link', source: a, target: b });
    });

    it('emits edge:click instead of canvas:click when clicking an edge', () => {
        const edges: unknown[] = [];
        controls.on('edge:click', (e) => edges.push(e.target));
        const log = record('canvas:click');
        click(550, 301);
        expect(edges).toHaveLength(1);
        expect((edges[0] as { tag: string }).tag).toBe('link');
        expect(log).toEqual([]);

        click(550, 340); // well off the edge
        expect(log).toEqual(['canvas:click']);
    });

    it('emits edge:hover and edge:hoverout', () => {
        const log = record('edge:hover', 'edge:hoverout', 'node:hover');
        fire('pointermove', 550, 300);
        fire('pointermove', 560, 301);
        fire('pointermove', 550, 360);
        expect(log).toEqual(['edge:hover', 'edge:hoverout']);
    });

    it('prefers nodes over the edges under them', () => {
        const log = record('edge:hover', 'node:hover');
        fire('pointermove', 400, 300);
        expect(log).toEqual(['node:hover']);
    });

    it('picks edges within a fixed edgeWidth', () => {
        const edges: unknown[] = [];
        controls.on('edge:click', (e) => edges.push(e.target));
        click(550, 309); // ~9px off: only within a wide edge
        expect(edges).toHaveLength(0);
        controls.setEdgeWidth(16);
        click(550, 309);
        expect(edges).toHaveLength(1);
    });

    it('does not pick edges when nobody listens', () => {
        const spy = vi.spyOn(controls, 'hitTestEdge');
        fire('pointermove', 550, 300);
        click(550, 300);
        expect(spy).not.toHaveBeenCalled();
    });
});

// -----------------------------------------------------------
// Box selection (interaction.lasso)
// -----------------------------------------------------------

describe('InteractionController: box selection', () => {
    function withLasso(): void {
        controls.destroy();
        controls = new InteractionController(canvas, graph, camera, { lasso: true });
    }

    it('selects the nodes inside a shift-dragged box', () => {
        withLasso();
        const log = record('node:select', 'canvas:pan');
        // Box around node a (400,300), not b (700,300); starts off any node
        fire('pointerdown', 240, 140, { shiftKey: true });
        fire('pointermove', 280, 180, { shiftKey: true });
        expect(document.querySelectorAll('div').length).toBe(1); // the box overlay
        fire('pointermove', 460, 360, { shiftKey: true });
        fire('pointerup', 460, 360, { shiftKey: true });

        expect(log).toEqual(['node:select']);
        expect([...controls.getSelectedNodes()]).toEqual([a]);
        expect(document.querySelectorAll('div').length).toBe(0);
        // The camera didn't pan
        const [sx] = camera.worldToScreen(0, 0);
        expect(sx).toBeCloseTo(400);
    });

    it('adds to the current selection', () => {
        withLasso();
        click(700, 300);
        fire('pointerdown', 240, 140, { shiftKey: true });
        fire('pointermove', 460, 360, { shiftKey: true });
        fire('pointerup', 460, 360, { shiftKey: true });
        expect([...controls.getSelectedNodes()].sort()).toEqual([a, b].sort());
    });

    it('removes the box when cancelled', () => {
        withLasso();
        fire('pointerdown', 240, 140, { shiftKey: true });
        fire('pointermove', 460, 360, { shiftKey: true });
        fire('pointercancel', 460, 360);
        expect(document.querySelectorAll('div').length).toBe(0);
        expect(controls.getSelectedNodes().size).toBe(0);
    });

    it('pans as usual when lasso is off', () => {
        const log = record('canvas:pan', 'node:select');
        fire('pointerdown', 100, 100, { shiftKey: true });
        fire('pointermove', 150, 100, { shiftKey: true });
        fire('pointerup', 150, 100, { shiftKey: true });
        expect(log).toContain('canvas:pan');
        expect(log).not.toContain('node:select');
    });
});
