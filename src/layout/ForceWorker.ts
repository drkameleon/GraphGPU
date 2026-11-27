// ============================================================
// graphGPU - Force Layout Worker (worker side)
// ============================================================
// Runs the CPU ForceLayout inside a Web Worker. The main thread
// (WorkerForceLayout) and the worker play ping-pong with one
// positions buffer: each 'frame' message hands the buffer over
// (transferred, not copied), the worker runs a frame of ticks on
// it and hands it back. So the worker is idle between frames, and
// never runs ahead of the main thread.
//
// This module installs itself when loaded as a worker script
// (the ESM build loads it directly; the standalone build loads
// the whole bundle), and does nothing on the main thread.
// ============================================================

import type { LayoutOptions } from '../types';
import { ForceLayout, type LayoutGraph } from './ForceLayout';

/** Main thread → worker */
export type ForceWorkerRequest =
    | { type: 'init'; options: LayoutOptions }
    | {
        type: 'frame';
        /** Positions for the first `nodeSlots` nodes */
        positions: ArrayBuffer;
        /** Node sizes, when they changed since the last frame */
        sizes?: Float32Array;
        /** Active edges as [src, tgt, ...], when they changed */
        edges?: Uint32Array;
        /** First frame of a run: reset the run counters */
        restart?: boolean;
    }
    | { type: 'pin' | 'unpin'; node: number }
    | { type: 'reset' };

/** Worker → main thread */
export interface ForceWorkerFrame {
    type: 'frame';
    positions: ArrayBuffer;
    iteration: number;
    energy: number;
    finished: boolean;
}

type Post = (msg: ForceWorkerFrame, transfer: Transferable[]) => void;

/**
 * Worker-side message handler: owns a ForceLayout over a plain
 * graph copy, kept in sync by the main thread.
 */
export function createForceWorker(post: Post): (msg: ForceWorkerRequest) => void {
    const graph: { -readonly [K in keyof LayoutGraph]: LayoutGraph[K] } = {
        nodeSlots: 0,
        edgeSlots: 0,
        positions: new Float32Array(0),
        sizes: new Float32Array(0),
        edgeIndices: new Uint32Array(0),
        isEdgeActive: () => true,
        dirtyNodes: false,
        dirtyEdges: false,
    };
    let layout: ForceLayout | null = null;

    return (msg) => {
        switch (msg.type) {
            case 'init':
                layout = new ForceLayout(graph, msg.options);
                break;
            case 'frame': {
                if (!layout) return;
                const positions = new Float32Array(msg.positions);
                graph.positions = positions;
                graph.nodeSlots = positions.length / 2;
                if (msg.sizes) graph.sizes = msg.sizes;
                if (msg.edges) {
                    graph.edgeIndices = msg.edges;
                    graph.edgeSlots = msg.edges.length / 2;
                }
                if (msg.restart) layout.restart();

                const { energy, finished } = layout.advance();
                post({
                    type: 'frame',
                    positions: positions.buffer,
                    iteration: layout.iterations,
                    energy,
                    finished,
                }, [positions.buffer]);
                break;
            }
            case 'pin':
                layout?.pin(msg.node);
                break;
            case 'unpin':
                layout?.unpin(msg.node);
                break;
            case 'reset':
                layout?.reset();
                break;
        }
    };
}

// Install when running as a worker script
const scope = globalThis as unknown as {
    WorkerGlobalScope?: new () => unknown;
    postMessage(msg: unknown, transfer: Transferable[]): void;
    addEventListener(type: 'message', cb: (e: MessageEvent) => void): void;
};
if (typeof scope.WorkerGlobalScope === 'function' && scope instanceof scope.WorkerGlobalScope) {
    const handle = createForceWorker((msg, transfer) => scope.postMessage(msg, transfer));
    scope.addEventListener('message', (e) => {
        // Ignore anything that isn't our protocol
        if (e.data && typeof e.data.type === 'string') handle(e.data);
    });
}
