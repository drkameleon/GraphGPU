// ============================================================
// graphGPU - Static Layouts
// ============================================================
// Instant, deterministic placements (no simulation): useful as
// a starting point, or for graphs where structure doesn't matter.
// Nodes are grouped by tag, then ordered by id.
// ============================================================

import type { Graph } from '../core/Graph';
import type { NodeId } from '../types';

export interface StaticLayoutOptions {
    /** Distance between neighboring nodes (world units). Default: 0.08 */
    spacing?: number;
    /** Center of the layout. Default: [0, 0] */
    center?: [number, number];
}

/** Active nodes, grouped by tag (in order of first appearance), then by id */
function orderedNodes(graph: Graph): NodeId[] {
    const groups = new Map<string, NodeId[]>();
    for (const id of graph.activeNodeIds()) {
        const tag = graph.getNodeTag(id) ?? '';
        let list = groups.get(tag);
        if (!list) groups.set(tag, list = []);
        list.push(id);
    }
    return [...groups.values()].flat();
}

function place(graph: Graph, ids: NodeId[], at: (index: number) => [number, number]): void {
    for (let i = 0; i < ids.length; i++) {
        const [x, y] = at(i);
        graph.setNodePosition(ids[i], x, y);
    }
}

/** Nodes evenly spaced on a circle (big enough to keep them `spacing` apart) */
export function circularLayout(graph: Graph, opts: StaticLayoutOptions = {}): void {
    const ids = orderedNodes(graph);
    const n = ids.length;
    if (n === 0) return;
    const spacing = opts.spacing ?? 0.08;
    const [cx, cy] = opts.center ?? [0, 0];
    const radius = n === 1 ? 0 : Math.max(0.5, (n * spacing) / (2 * Math.PI));
    place(graph, ids, (i) => {
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2; // start at 12 o'clock
        return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius];
    });
}

/** Nodes on a square grid, row by row from the top left */
export function gridLayout(graph: Graph, opts: StaticLayoutOptions = {}): void {
    const ids = orderedNodes(graph);
    const n = ids.length;
    if (n === 0) return;
    const spacing = opts.spacing ?? 0.08;
    const [cx, cy] = opts.center ?? [0, 0];
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const x0 = cx - ((cols - 1) * spacing) / 2;
    const y0 = cy + ((rows - 1) * spacing) / 2;
    place(graph, ids, (i) => [x0 + (i % cols) * spacing, y0 - Math.floor(i / cols) * spacing]);
}
