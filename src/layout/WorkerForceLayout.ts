// ============================================================
// graphGPU - Force Layout in a Web Worker (main thread side)
// ============================================================
// Same API as ForceLayout, but the physics run in a worker, so a
// big CPU layout doesn't block the main thread. Each frame the
// worker hands the positions back, the main thread merges them
// into the graph and returns the buffer on the next animation
// frame (see ForceWorker.ts for the protocol).
//
// Edits made on the main thread win: a node moved (e.g. dragged)
// while a frame was in flight keeps its new position, and the
// worker picks it up with the next frame. Graph changes (nodes or
// edges added/removed, sizes changed) are synced the same way.
//
// Falls back to running on the main thread when workers aren't
// available or the worker script fails to load.
// ============================================================

import type { LayoutOptions } from '../types';
import type { Graph } from '../core/Graph';
import { ForceLayout } from './ForceLayout';
import type { ForceWorkerFrame, ForceWorkerRequest } from './ForceWorker';
import './ForceWorker';

/** The bits of a Worker we use (so tests can fake one) */
export interface WorkerLike {
    postMessage(msg: ForceWorkerRequest, transfer?: Transferable[]): void;
    addEventListener(type: 'message', cb: (e: MessageEvent<ForceWorkerFrame>) => void): void;
    addEventListener(type: 'error', cb: (e: Event) => void): void;
    terminate(): void;
}

// The standalone (IIFE) build has no import.meta.url: it runs the
// worker from its own script instead. Remember it while it's current.
const scriptSrc = typeof document !== 'undefined'
    ? (document.currentScript as HTMLScriptElement | null)?.src
    : undefined;

/** Load a script in a worker; cross-origin scripts go through a blob */
function workerFor(url: string, type: WorkerType): Worker {
    try {
        return new Worker(url, { type });
    } catch {
        const code = type === 'module'
            ? `import ${JSON.stringify(url)};`
            : `importScripts(${JSON.stringify(url)});`;
        const blob = new Blob([code], { type: 'text/javascript' });
        return new Worker(URL.createObjectURL(blob), { type });
    }
}

/** Start the layout worker, or null if that isn't possible here */
export function spawnForceWorker(): WorkerLike | null {
    if (typeof Worker === 'undefined') return null;
    try {
        // Literal `new Worker(new URL(...))`, so bundlers (Vite,
        // webpack) spot the worker and bundle it
        return new Worker(new URL('./ForceWorker.js', import.meta.url), { type: 'module' });
    } catch { /* cross-origin, or no import.meta.url */ }
    try {
        const url = new URL('./ForceWorker.js', import.meta.url).href;
        return workerFor(url, 'module');
    } catch { /* no import.meta.url: standalone build */ }
    try {
        // Only our own bundle, never some app bundle that happens to include us
        if (scriptSrc && /graphgpu[^/]*\.js(\?|#|$)/i.test(scriptSrc)) {
            return workerFor(scriptSrc, 'classic');
        }
    } catch { /* give up */ }
    return null;
}

export class WorkerForceLayout {
    /** Main-thread layout: fallback, and synchronous step() */
    private local: ForceLayout;
    private worker: WorkerLike | null = null;
    /** Worker unusable: everything runs on `local` */
    private fallback: boolean = false;

    private running: boolean = false;
    private animated: boolean = false;
    private pinned: Set<number> = new Set();
    private frameId: number = 0;
    /** A frame is in flight (the worker holds the buffer) */
    private pending: boolean = false;
    /** Next frame starts a new run */
    private restartNext: boolean = false;

    /** Positions as sent with the in-flight frame (to spot main-thread edits) */
    private sent: Float32Array = new Float32Array(0);
    /** The ping-pong buffer, while it's on this side */
    private buffer: ArrayBuffer | null = null;
    /** What the worker has of the graph */
    private syncedStructure: number = -1;
    private syncedSizes: Float32Array = new Float32Array(0);

    onStart?: () => void;
    onTick?: (iteration: number, energy: number) => void;
    onStop?: () => void;

    constructor(
        private graph: Graph,
        private opts: LayoutOptions,
        private spawn: () => WorkerLike | null = spawnForceWorker,
    ) {
        this.local = new ForceLayout(graph, opts);
        this.local.onStart = () => this.onStart?.();
        this.local.onTick = (i, e) => this.onTick?.(i, e);
        this.local.onStop = () => this.onStop?.();
    }

    /** Whether the physics run in a worker (false once fallen back) */
    get inWorker(): boolean {
        return !this.fallback && this.worker !== null;
    }

    // =========================================================
    // Animated mode + Pinning
    // =========================================================

    setAnimated(enabled: boolean): void {
        this.animated = enabled;
        if (this.fallback) {
            this.local.setAnimated(enabled);
            return;
        }
        if (enabled && !this.running) this.start();
    }

    isAnimated(): boolean {
        return this.fallback ? this.local.isAnimated() : this.animated;
    }

    pin(nodeId: number): void {
        this.pinned.add(nodeId);
        this.local.pin(nodeId);
        this.worker?.postMessage({ type: 'pin', node: nodeId });
    }

    unpin(nodeId: number): void {
        this.pinned.delete(nodeId);
        this.local.unpin(nodeId);
        this.worker?.postMessage({ type: 'unpin', node: nodeId });
    }

    // =========================================================
    // Step / Start / Stop
    // =========================================================

    /** Synchronous steps run on the main thread */
    step(steps: number = 1): number {
        return this.local.step(steps);
    }

    start(): void {
        if (this.fallback) {
            this.local.start();
            return;
        }
        if (this.running) return;
        if (!this.worker && !this.startWorker()) {
            this.useFallback(true);
            return;
        }

        this.running = true;
        this.restartNext = !this.animated;
        this.onStart?.();
        if (!this.pending) this.sendFrame();
    }

    stop(): void {
        if (this.fallback) {
            this.local.stop();
            return;
        }
        if (this.frameId) {
            cancelAnimationFrame(this.frameId);
            this.frameId = 0;
        }
        if (!this.running) return;
        this.running = false;
        this.onStop?.();
    }

    isRunning(): boolean {
        return this.fallback ? this.local.isRunning() : this.running;
    }

    reset(): void {
        this.local.reset();
        this.worker?.postMessage({ type: 'reset' });
    }

    destroy(): void {
        this.stop();
        this.local.destroy();
        this.worker?.terminate();
        this.worker = null;
    }

    // =========================================================
    // Worker
    // =========================================================

    private startWorker(): boolean {
        let worker: WorkerLike | null = null;
        try {
            worker = this.spawn();
        } catch {
            worker = null;
        }
        if (!worker) return false;

        this.worker = worker;
        worker.addEventListener('message', (e) => this.receive(e.data));
        // Failed to load (or crashed): carry on, on the main thread
        worker.addEventListener('error', () => this.useFallback(this.running));
        worker.postMessage({ type: 'init', options: this.opts });
        for (const node of this.pinned) worker.postMessage({ type: 'pin', node });
        return true;
    }

    /** Switch to the main thread for good, resuming if `resume` */
    private useFallback(resume: boolean): void {
        if (this.fallback) return;
        if (this.frameId) cancelAnimationFrame(this.frameId);
        this.frameId = 0;
        this.pending = false;
        this.worker?.terminate();
        this.worker = null;
        this.fallback = true;

        // Mid-run: carry on without a second onStart
        const wasRunning = this.running;
        this.running = false;
        if (wasRunning) this.local.onStart = undefined;
        if (this.animated) this.local.setAnimated(true);
        else if (resume) this.local.start();
        this.local.onStart = () => this.onStart?.();
    }

    /** Hand the positions (and any graph changes) to the worker */
    private sendFrame(): void {
        this.frameId = 0;
        if (!this.running || !this.worker) return;

        const g = this.graph;
        const len = g.nodeSlots * 2;

        // Reuse the ping-pong buffer when it still fits
        const positions = this.buffer && this.buffer.byteLength === len * 4
            ? new Float32Array(this.buffer)
            : new Float32Array(len);
        this.buffer = null;
        positions.set(g.positions.subarray(0, len));
        if (this.sent.length !== len) this.sent = new Float32Array(len);
        this.sent.set(positions);

        const msg: ForceWorkerRequest = { type: 'frame', positions: positions.buffer };
        const transfer: Transferable[] = [positions.buffer];

        if (g.structureVersion !== this.syncedStructure) {
            this.syncedStructure = g.structureVersion;
            msg.edges = this.activeEdges();
            transfer.push(msg.edges.buffer);
        }
        const sizes = g.sizes.subarray(0, g.nodeSlots);
        if (!sameValues(sizes, this.syncedSizes)) {
            this.syncedSizes = sizes.slice();
            msg.sizes = sizes.slice();
            transfer.push(msg.sizes.buffer);
        }
        if (this.restartNext) {
            msg.restart = true;
            this.restartNext = false;
        }

        this.pending = true;
        this.worker.postMessage(msg, transfer);
    }

    private receive(frame: ForceWorkerFrame): void {
        if (frame?.type !== 'frame') return;
        this.pending = false;
        this.buffer = frame.positions;

        // Stopped meanwhile: keep the buffer, drop the result
        if (!this.running) return;

        // Merge, keeping nodes moved on this side since the frame was sent
        const g = this.graph;
        const result = new Float32Array(frame.positions);
        const pos = g.positions;
        const sent = this.sent;
        const len = Math.min(result.length, g.nodeSlots * 2);
        for (let i = 0; i < len; i += 2) {
            if (pos[i] === sent[i] && pos[i + 1] === sent[i + 1]) {
                pos[i] = result[i];
                pos[i + 1] = result[i + 1];
            }
        }
        g.dirtyNodes = true;
        g.dirtyEdges = true;

        this.onTick?.(frame.iteration, frame.energy);

        if (frame.finished && !this.animated) {
            this.stop();
            return;
        }
        this.frameId = requestAnimationFrame(() => this.sendFrame());
    }

    /** Active edges, packed as [src, tgt, ...] */
    private activeEdges(): Uint32Array {
        const g = this.graph;
        const idx = g.edgeIndices;
        const out = new Uint32Array(g.numEdges * 2);
        let k = 0;
        for (let e = 0; e < g.edgeSlots && k < out.length; e++) {
            if (!g.isEdgeActive(e)) continue;
            out[k++] = idx[e * 2];
            out[k++] = idx[e * 2 + 1];
        }
        return k === out.length ? out : out.slice(0, k);
    }
}

function sameValues(a: Float32Array, b: Float32Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}
