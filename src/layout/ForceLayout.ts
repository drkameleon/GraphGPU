// ============================================================
// graphGPU - Force-Directed Layout (CPU)
// ============================================================
// Simple force-directed layout that runs on the main thread
// or in a Web Worker. GPU compute version is separate.
// ============================================================

import type { LayoutOptions } from '../types';
import type { Graph } from '../core/Graph';

export interface ForceLayoutConfig {
    repulsion: number;
    attraction: number;
    gravity: number;
    damping: number;
    maxIterations: number;
    convergenceThreshold: number;
}

const DEFAULTS: ForceLayoutConfig = {
    repulsion: 1.0,
    attraction: 0.01,
    gravity: 0.05,
    damping: 0.92,
    maxIterations: 500,
    convergenceThreshold: 0.001,
};

export class ForceLayout {
    private config: ForceLayoutConfig;
    private velocities: Float32Array;
    private running: boolean = false;
    private iteration: number = 0;

    /** Animated mode: layout never auto-stops, physics are always live */
    private animated: boolean = false;

    /** Pinned nodes: positions set externally (e.g. by dragging), skip integration */
    private pinned: Set<number> = new Set();

    // Callbacks
    onTick?: (iteration: number, energy: number) => void;
    onStop?: () => void;

    constructor(
        private graph: Graph,
        opts?: LayoutOptions,
    ) {
        this.config = {
            repulsion: opts?.repulsion ?? DEFAULTS.repulsion,
            attraction: opts?.attraction ?? DEFAULTS.attraction,
            gravity: opts?.gravity ?? DEFAULTS.gravity,
            damping: opts?.damping ?? DEFAULTS.damping,
            maxIterations: opts?.maxIterations ?? DEFAULTS.maxIterations,
            convergenceThreshold: DEFAULTS.convergenceThreshold,
        };

        this.velocities = new Float32Array(graph.numNodes * 2);
    }

    /** Enable/disable animated mode (live physics, never auto-stops) */
    setAnimated(enabled: boolean): void {
        this.animated = enabled;
        // If turning on and not running, restart
        if (enabled && !this.running) {
            this.start();
        }
    }

    /** Is animated mode active? */
    isAnimated(): boolean {
        return this.animated;
    }

    /** Pin a node (its position is controlled externally, e.g. by dragging) */
    pin(nodeId: number): void {
        this.pinned.add(nodeId);
        // Zero out velocity for pinned node
        this.velocities[nodeId * 2] = 0;
        this.velocities[nodeId * 2 + 1] = 0;
    }

    /** Unpin a node (rejoin physics simulation) */
    unpin(nodeId: number): void {
        this.pinned.delete(nodeId);
    }

    /** Run layout synchronously for N steps */
    step(steps: number = 1): number {
        let energy = 0;

        for (let s = 0; s < steps; s++) {
            energy = this.tick();
            this.iteration++;

            if (energy < this.config.convergenceThreshold) {
                break;
            }
        }

        return energy;
    }

    /** Start async layout loop using requestAnimationFrame */
    start(): void {
        if (this.running) return;
        this.running = true;
        if (!this.animated) this.iteration = 0;

        // Ensure velocity buffer matches node count
        if (this.velocities.length < this.graph.numNodes * 2) {
            const old = this.velocities;
            this.velocities = new Float32Array(this.graph.numNodes * 2);
            this.velocities.set(old.subarray(0, Math.min(old.length, this.velocities.length)));
        }

        const loop = () => {
            if (!this.running) return;

            // Run multiple ticks per frame for faster convergence
            const ticksPerFrame = 3;
            let energy = 0;
            for (let i = 0; i < ticksPerFrame; i++) {
                energy = this.tick();
                this.iteration++;
            }

            this.onTick?.(this.iteration, energy);

            // In animated mode, never auto-stop
            if (!this.animated) {
                if (
                    energy < this.config.convergenceThreshold ||
                    this.iteration >= this.config.maxIterations
                ) {
                    this.stop();
                    return;
                }
            }

            requestAnimationFrame(loop);
        };

        requestAnimationFrame(loop);
    }

    /** Stop the layout */
    stop(): void {
        this.running = false;
        this.onStop?.();
    }

    /** Is layout currently running? */
    isRunning(): boolean {
        return this.running;
    }

    /** Reset velocities */
    reset(): void {
        this.velocities.fill(0);
        this.iteration = 0;
    }

    // =========================================================
    // Core simulation tick
    // =========================================================

    private tick(): number {
        const n = this.graph.numNodes;
        const pos = this.graph.positions;
        const vel = this.velocities;
        const edges = this.graph.edgeIndices;
        const edgeCount = this.graph.numEdges;
        const sizes = this.graph.sizes;
        const pinned = this.pinned;

        const { repulsion, attraction, gravity, damping } = this.config;

        // --- Repulsion (O(n²) brute force) ---
        for (let i = 0; i < n; i++) {
            if (sizes[i] <= 0) continue; // skip deleted
            if (pinned.has(i)) continue; // pinned: don't accumulate forces

            const ix = i * 2;
            const iy = ix + 1;
            let fx = 0, fy = 0;

            for (let j = 0; j < n; j++) {
                if (i === j || sizes[j] <= 0) continue;

                const jx = j * 2;
                const jy = jx + 1;
                const dx = pos[ix] - pos[jx];
                const dy = pos[iy] - pos[jy];
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.001);

                // Coulomb repulsion
                const force = repulsion / (dist * dist);
                fx += (dx / dist) * force;
                fy += (dy / dist) * force;
            }

            // Gravity toward origin
            fx -= pos[ix] * gravity;
            fy -= pos[iy] * gravity;

            vel[ix] = (vel[ix] + fx * 0.016) * damping;
            vel[iy] = (vel[iy] + fy * 0.016) * damping;
        }

        // --- Attraction along edges ---
        for (let e = 0; e < edgeCount; e++) {
            if (!this.graph.isEdgeActive(e)) continue;

            const src = edges[e * 2];
            const tgt = edges[e * 2 + 1];
            if (sizes[src] <= 0 || sizes[tgt] <= 0) continue;

            const sx = src * 2, sy = sx + 1;
            const tx = tgt * 2, ty = tx + 1;

            const dx = pos[tx] - pos[sx];
            const dy = pos[ty] - pos[sy];
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.001);

            const force = (dist - 0.5) * attraction;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            // Only apply force to unpinned end(s)
            if (!pinned.has(src)) {
                vel[sx] += fx * 0.016;
                vel[sy] += fy * 0.016;
            }
            if (!pinned.has(tgt)) {
                vel[tx] -= fx * 0.016;
                vel[ty] -= fy * 0.016;
            }
        }

        // --- Integration + energy measurement ---
        let totalEnergy = 0;
        for (let i = 0; i < n; i++) {
            if (sizes[i] <= 0) continue;
            if (pinned.has(i)) continue; // pinned: position set externally

            const ix = i * 2;
            const iy = ix + 1;

            pos[ix] += vel[ix] * 0.016;
            pos[iy] += vel[iy] * 0.016;

            totalEnergy += vel[ix] * vel[ix] + vel[iy] * vel[iy];
        }

        this.graph.dirtyNodes = true;
        this.graph.dirtyEdges = true;

        return totalEnergy / Math.max(n, 1);
    }
}
