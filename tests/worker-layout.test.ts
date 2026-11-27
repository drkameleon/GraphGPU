// ============================================================
// graphGPU - WorkerForceLayout tests
// ============================================================
// The worker is faked in-process: messages go through
// structuredClone (with transfers, so buffers really detach)
// to the real worker-side handler, delivered asynchronously.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Graph } from '../src/core/Graph';
import { ForceLayout } from '../src/layout/ForceLayout';
import { WorkerForceLayout, type WorkerLike } from '../src/layout/WorkerForceLayout';
import { createForceWorker, type ForceWorkerRequest } from '../src/layout/ForceWorker';
import { randomGraph } from './helpers';

let frames = new Map<number, FrameRequestCallback>();
let nextFrameId = 1;

function flushFrames(count: number): void {
    for (let i = 0; i < count && frames.size > 0; i++) {
        const pending = [...frames.values()];
        frames = new Map();
        for (const cb of pending) cb(0);
    }
}

beforeEach(() => {
    frames = new Map();
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
        const id = nextFrameId++;
        frames.set(id, cb);
        return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
});

afterEach(() => {
    vi.unstubAllGlobals();
});

class FakeWorker implements WorkerLike {
    sent: ForceWorkerRequest[] = [];
    /** Positions of each frame sent, as the worker got them */
    framePositions: Float32Array[] = [];
    terminated = false;
    private inbox: unknown[] = [];
    private listeners: { message: ((e: MessageEvent) => void)[]; error: ((e: Event) => void)[] } =
        { message: [], error: [] };
    private handle = createForceWorker((msg, transfer) => {
        this.inbox.push(structuredClone(msg, { transfer }));
    });

    postMessage(msg: ForceWorkerRequest, transfer: Transferable[] = []): void {
        if (this.terminated) return;
        const copy = structuredClone(msg, { transfer });
        this.sent.push(copy);
        if (copy.type === 'frame') this.framePositions.push(new Float32Array(copy.positions).slice());
        this.handle(copy);
    }

    addEventListener(type: 'message' | 'error', cb: (e: any) => void): void {
        this.listeners[type].push(cb);
    }

    terminate(): void {
        this.terminated = true;
    }

    /** Deliver the worker's replies (they're async in a real worker) */
    deliver(): number {
        const msgs = this.inbox;
        this.inbox = [];
        for (const data of msgs) {
            if (this.terminated) break;
            for (const cb of this.listeners.message) cb({ data } as MessageEvent);
        }
        return msgs.length;
    }

    fail(): void {
        for (const cb of this.listeners.error) cb(new Event('error'));
    }
}

/** Deliver replies and run animation frames, `count` times */
function pump(worker: FakeWorker, count: number): void {
    for (let i = 0; i < count; i++) {
        worker.deliver();
        flushFrames(1);
    }
}

function setup(nodes = 50, edges = 80, opts = {}) {
    const g = randomGraph(nodes, edges, 7);
    const worker = new FakeWorker();
    const layout = new WorkerForceLayout(g, { type: 'force', ...opts }, () => worker);
    return { g, worker, layout };
}

describe('WorkerForceLayout', () => {
    it('runs the same physics as the main-thread layout', () => {
        const { g, worker, layout } = setup();
        const ref = randomGraph(50, 80, 7);
        const refLayout = new ForceLayout(ref, { type: 'force' });

        layout.start();
        expect(layout.inWorker).toBe(true);
        pump(worker, 10);
        for (let i = 0; i < 10; i++) refLayout.advance();

        expect(Array.from(g.positions.subarray(0, 100)))
            .toEqual(Array.from(ref.positions.subarray(0, 100)));
        layout.destroy();
    });

    it('hands the positions buffer over instead of copying it', () => {
        const { worker, layout } = setup();
        layout.start();
        pump(worker, 3);
        // The worker got the buffer, then gave it back (detached here)
        const frames = worker.sent.filter((m) => m.type === 'frame');
        expect(frames.length).toBeGreaterThan(1);
        expect(worker.framePositions.every((p) => p.length === 100)).toBe(true);
        for (const m of frames) expect((m as { positions: ArrayBuffer }).positions.byteLength).toBe(0);
        layout.destroy();
    });

    it('only sends the graph structure when it changes', () => {
        const { worker, layout } = setup();
        layout.start();
        pump(worker, 4);
        const withEdges = () => worker.sent.filter((m) => m.type === 'frame' && m.edges).length;
        expect(withEdges()).toBe(1);
        layout.destroy();
    });

    it('fires start, tick and stop', () => {
        const { worker, layout } = setup(20, 20, { maxIterations: 9 });
        const events: string[] = [];
        layout.onStart = () => events.push('start');
        layout.onTick = () => events.push('tick');
        layout.onStop = () => events.push('stop');

        layout.start();
        pump(worker, 10);
        expect(events).toEqual(['start', 'tick', 'tick', 'tick', 'stop']);
        expect(layout.isRunning()).toBe(false);
        expect(frames.size).toBe(0);
    });

    it('drops a frame that arrives after stop()', () => {
        const { g, worker, layout } = setup();
        layout.start();
        layout.stop();
        const before = g.positions.slice(0, 100);
        worker.deliver();
        expect(g.positions.subarray(0, 100)).toEqual(before);
        expect(frames.size).toBe(0);
    });

    it('can be restarted', () => {
        const { g, worker, layout } = setup();
        layout.start();
        pump(worker, 2);
        layout.stop();
        worker.deliver();

        const before = g.positions.slice(0, 100);
        layout.start();
        pump(worker, 2);
        expect(g.positions.subarray(0, 100)).not.toEqual(before);
        layout.destroy();
    });

    it('keeps nodes moved on the main thread while a frame is in flight', () => {
        const { g, worker, layout } = setup();
        layout.start();
        g.setNodePosition(3, 0.5, -0.5);
        worker.deliver();
        expect(g.positions[6]).toBe(0.5);
        expect(g.positions[7]).toBe(-0.5);

        // ...and the worker carries on from there
        flushFrames(1);
        const last = worker.framePositions.at(-1)!;
        expect(last[6]).toBe(0.5);
        expect(last[7]).toBe(-0.5);
        layout.destroy();
    });

    it('holds pinned nodes in place', () => {
        const { g, worker, layout } = setup();
        layout.pin(4);
        layout.start();
        pump(worker, 5);
        const x = g.positions[8], y = g.positions[9];
        pump(worker, 5);
        expect(g.positions[8]).toBe(x);
        expect(g.positions[9]).toBe(y);
        expect(worker.sent.some((m) => m.type === 'pin' && m.node === 4)).toBe(true);
        layout.destroy();
    });

    it('picks up nodes added while running', () => {
        const { g, worker, layout } = setup();
        layout.start();
        pump(worker, 2);
        const id = g.addNode({ tag: 'new', x: 0, y: 0 });
        g.addEdge({ tag: 'e', source: id, target: 0 });
        pump(worker, 3);
        expect(g.positions[id * 2] !== 0 || g.positions[id * 2 + 1] !== 0).toBe(true);
        layout.destroy();
    });

    it('never stops in animated mode', () => {
        const { worker, layout } = setup(20, 20, { maxIterations: 3 });
        layout.setAnimated(true);
        expect(layout.isRunning()).toBe(true);
        pump(worker, 10);
        expect(layout.isRunning()).toBe(true);
        layout.destroy();
        expect(worker.terminated).toBe(true);
    });

    it('runs on the main thread when workers are unavailable', () => {
        const g = randomGraph(50, 80, 7);
        const layout = new WorkerForceLayout(g, { type: 'force' }, () => null);
        const before = g.positions.slice(0, 100);
        layout.start();
        expect(layout.inWorker).toBe(false);
        expect(layout.isRunning()).toBe(true);
        flushFrames(3);
        expect(g.positions.subarray(0, 100)).not.toEqual(before);
        layout.destroy();
        expect(frames.size).toBe(0);
    });

    it('falls back to the main thread if the worker fails', () => {
        const { g, worker, layout } = setup();
        const starts = vi.fn();
        layout.onStart = starts;
        layout.start();
        worker.fail();
        expect(worker.terminated).toBe(true);
        expect(layout.isRunning()).toBe(true);
        expect(starts).toHaveBeenCalledTimes(1);

        const before = g.positions.slice(0, 100);
        flushFrames(3);
        expect(g.positions.subarray(0, 100)).not.toEqual(before);
        layout.destroy();
    });

    it('steps synchronously on the main thread', () => {
        const { g, layout } = setup();
        const before = g.positions.slice(0, 100);
        layout.step(5);
        expect(g.positions.subarray(0, 100)).not.toEqual(before);
    });
});

describe('WorkerForceLayout: no Worker support', () => {
    it('still lays out (Node has no Worker)', () => {
        const g = new Graph();
        expect(typeof Worker).toBe('undefined');
        const layout = new WorkerForceLayout(g, { type: 'force' });
        g.addNode({ tag: 'a', x: 0.01, y: 0 });
        g.addNode({ tag: 'a', x: -0.01, y: 0 });
        layout.start();
        expect(layout.inWorker).toBe(false);
        flushFrames(2);
        expect(Math.abs(g.positions[0])).toBeGreaterThan(0.01);
        layout.destroy();
    });
});
