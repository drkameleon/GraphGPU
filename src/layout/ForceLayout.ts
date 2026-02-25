// ============================================================
// graphGPU - Force-Directed Layout (CPU)
// ============================================================
// Physics simulation inspired by vis.js network:
//   - Gravitational repulsion between all nodes (N-body)
//     → Barnes-Hut quadtree approximation: O(n log n)
//     → Falls back to brute-force O(n²) when theta = 0
//   - Hooke's law springs along edges (with rest length)
//   - Central gravity pulling toward origin
//   - Proper velocity integration: a = (F - damping*v) / mass
//   - maxVelocity clamping to prevent explosions
//   - Stabilization detection (minVelocity threshold)
//   - Animated mode: physics never auto-stops, pinned nodes
//     for drag interaction
// ============================================================

import type { LayoutOptions } from '../types';
import type { Graph } from '../core/Graph';
import { QuadTree } from './QuadTree';

export interface ForceLayoutConfig {
    // ── Repulsion (gravitational) ──
    gravitationalConstant: number; // negative = repulsive (vis.js default: -2000)
    // ── Barnes-Hut ──
    barnesHutTheta: number;        // approximation threshold (0 = brute-force, 0.5 = typical)
    // ── Springs (edges) ──
    springLength: number;          // rest length of edge springs (vis.js: 95)
    springConstant: number;        // spring stiffness (vis.js: 0.04)
    // ── Central gravity ──
    centralGravity: number;        // pull toward origin (vis.js: 0.3)
    // ── Integration ──
    damping: number;               // velocity damping coefficient (vis.js: 0.09)
    timestep: number;              // integration timestep (vis.js: 0.5)
    maxVelocity: number;           // velocity clamp (vis.js: 50)
    minVelocity: number;           // stabilization threshold (vis.js: 0.75)
    // ── Limits ──
    maxIterations: number;
}

const DEFAULTS: ForceLayoutConfig = {
    // Tuned for GraphGPU coordinate space (positions in [-1, 1] range).
    gravitationalConstant: -0.25,  // moderate repulsion
    barnesHutTheta: 0.5,           // Barnes-Hut approximation (0 = brute-force)
    springLength: 0.2,             // short rest length → compact clusters
    springConstant: 0.06,          // moderate spring stiffness
    centralGravity: 0.012,         // strong pull → keeps graph centered and compact
    damping: 0.18,                 // higher damping → smooth elastic settling
    timestep: 0.35,                // smaller steps → stable integration
    maxVelocity: 0.06,             // prevent sudden jumps
    minVelocity: 0.0005,           // stabilization threshold
    maxIterations: 1000,
};

export class ForceLayout {
    private config: ForceLayoutConfig;
    // Per-node: vx, vy (interleaved)
    private velocities: Float32Array;
    // Per-node: fx, fy (interleaved, reset each tick)
    private forces: Float32Array;
    private running: boolean = false;
    private iteration: number = 0;

    /** Barnes-Hut quadtree — reused across ticks (no re-allocation) */
    private quadTree: QuadTree;

    /** Animated mode: layout never auto-stops, physics always live */
    private animated: boolean = false;

    /** Pinned nodes: position set externally (e.g. by drag), skip integration */
    private pinned: Set<number> = new Set();

    // Callbacks
    onTick?: (iteration: number, energy: number) => void;
    onStop?: () => void;

    constructor(
        private graph: Graph,
        opts?: LayoutOptions,
    ) {
        this.config = {
            gravitationalConstant: opts?.gravitationalConstant ?? DEFAULTS.gravitationalConstant,
            barnesHutTheta: opts?.barnesHutTheta ?? DEFAULTS.barnesHutTheta,
            springLength: opts?.springLength ?? DEFAULTS.springLength,
            springConstant: opts?.springConstant ?? DEFAULTS.springConstant,
            centralGravity: opts?.centralGravity ?? DEFAULTS.centralGravity,
            damping: opts?.damping ?? DEFAULTS.damping,
            timestep: opts?.timestep ?? DEFAULTS.timestep,
            maxVelocity: opts?.maxVelocity ?? DEFAULTS.maxVelocity,
            minVelocity: opts?.minVelocity ?? DEFAULTS.minVelocity,
            maxIterations: opts?.maxIterations ?? DEFAULTS.maxIterations,
        };

        const n = graph.numNodes;
        this.velocities = new Float32Array(n * 2);
        this.forces = new Float32Array(n * 2);
        this.quadTree = new QuadTree();
    }

    // =========================================================
    // Animated mode + Pinning
    // =========================================================

    setAnimated(enabled: boolean): void {
        this.animated = enabled;
        if (enabled && !this.running) {
            this.start();
        }
    }

    isAnimated(): boolean {
        return this.animated;
    }

    pin(nodeId: number): void {
        this.pinned.add(nodeId);
        this.velocities[nodeId * 2] = 0;
        this.velocities[nodeId * 2 + 1] = 0;
    }

    unpin(nodeId: number): void {
        this.pinned.delete(nodeId);
    }

    // =========================================================
    // Step / Start / Stop
    // =========================================================

    step(steps: number = 1): number {
        let maxVel = 0;
        for (let s = 0; s < steps; s++) {
            maxVel = this.tick();
            this.iteration++;
            if (!this.animated && maxVel < this.config.minVelocity) break;
        }
        return maxVel;
    }

    start(): void {
        if (this.running) return;
        this.running = true;
        if (!this.animated) this.iteration = 0;

        this.ensureBuffers();

        const loop = () => {
            if (!this.running) return;

            // 3 ticks per frame for faster convergence
            let maxVel = 0;
            for (let i = 0; i < 3; i++) {
                maxVel = this.tick();
                this.iteration++;
            }

            this.onTick?.(this.iteration, maxVel);

            if (!this.animated) {
                if (
                    maxVel < this.config.minVelocity ||
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

    stop(): void {
        this.running = false;
        this.onStop?.();
    }

    isRunning(): boolean {
        return this.running;
    }

    reset(): void {
        this.velocities.fill(0);
        this.forces.fill(0);
        this.iteration = 0;
    }

    // =========================================================
    // Core simulation tick (vis.js-style physics)
    // =========================================================

    private tick(): number {
        const n = this.graph.numNodes;
        const pos = this.graph.positions;
        const vel = this.velocities;
        const forces = this.forces;
        const edgeIdx = this.graph.edgeIndices;
        const edgeCount = this.graph.numEdges;
        const sizes = this.graph.sizes;
        const pinned = this.pinned;
        const cfg = this.config;

        this.ensureBuffers();

        // Reset forces
        forces.fill(0);

        // ── 1. Node-node repulsion ──
        const G = cfg.gravitationalConstant;
        if (G !== 0) {
            if (cfg.barnesHutTheta > 0) {
                // Barnes-Hut: O(n log n)
                this.repulsionBarnesHut(n, pos, sizes, forces, G, cfg.barnesHutTheta);
            } else {
                // Brute-force: O(n²)
                this.repulsionBruteForce(n, pos, sizes, forces, G);
            }
        }

        // ── 2. Edge springs (Hooke's law) ──
        // vis.js: springForce = springConstant * (springLength - distance) / distance
        const springK = cfg.springConstant;
        const springL = cfg.springLength;

        for (let e = 0; e < edgeCount; e++) {
            if (!this.graph.isEdgeActive(e)) continue;

            const src = edgeIdx[e * 2];
            const tgt = edgeIdx[e * 2 + 1];
            if (sizes[src] <= 0 || sizes[tgt] <= 0) continue;

            const sx = src * 2, sy = sx + 1;
            const tx = tgt * 2, ty = tx + 1;

            const dx = pos[sx] - pos[tx];
            const dy = pos[sy] - pos[ty];
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);

            // Hooke's law with rest length
            const springForce = springK * (springL - dist) / dist;
            const fx = dx * springForce;
            const fy = dy * springForce;

            forces[sx] += fx;
            forces[sy] += fy;
            forces[tx] -= fx;
            forces[ty] -= fy;
        }

        // ── 3. Central gravity ──
        // vis.js: gravityForce = centralGravity / distance (toward origin)
        const cg = cfg.centralGravity;
        if (cg !== 0) {
            for (let i = 0; i < n; i++) {
                if (sizes[i] <= 0) continue;
                const ix = i * 2;
                const iy = ix + 1;
                const dx = -pos[ix];
                const dy = -pos[iy];
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
                const gForce = cg / dist;
                forces[ix] += dx * gForce;
                forces[iy] += dy * gForce;
            }
        }

        // ── 4. Velocity integration + position update ──
        // vis.js: v += (F - damping*v) / mass * dt; x += v * dt
        const dt = cfg.timestep;
        const damp = cfg.damping;
        const maxV = cfg.maxVelocity;
        let maxNodeVel = 0;

        for (let i = 0; i < n; i++) {
            if (sizes[i] <= 0) continue;
            if (pinned.has(i)) continue;

            const ix = i * 2;
            const iy = ix + 1;

            // acceleration = (force - damping*velocity) / mass
            const ax = forces[ix] - damp * vel[ix];
            const ay = forces[iy] - damp * vel[iy];

            vel[ix] += ax * dt;
            vel[iy] += ay * dt;

            // Clamp velocity
            if (Math.abs(vel[ix]) > maxV) vel[ix] = vel[ix] > 0 ? maxV : -maxV;
            if (Math.abs(vel[iy]) > maxV) vel[iy] = vel[iy] > 0 ? maxV : -maxV;

            // Update position
            pos[ix] += vel[ix] * dt;
            pos[iy] += vel[iy] * dt;

            const nodeVel = Math.sqrt(vel[ix] * vel[ix] + vel[iy] * vel[iy]);
            if (nodeVel > maxNodeVel) maxNodeVel = nodeVel;
        }

        this.graph.dirtyNodes = true;
        this.graph.dirtyEdges = true;

        return maxNodeVel;
    }

    // =========================================================
    // Repulsion strategies
    // =========================================================

    /**
     * Barnes-Hut approximation: O(n log n).
     * Builds a quadtree, then walks it per-node, approximating
     * distant clusters as single point masses.
     */
    private repulsionBarnesHut(
        n: number,
        pos: Float32Array,
        sizes: Float32Array,
        forces: Float32Array,
        G: number,
        theta: number,
    ): void {
        // Rebuild quadtree (O(n), the tree object is reused)
        this.quadTree.build(pos, sizes, n);

        for (let i = 0; i < n; i++) {
            if (sizes[i] <= 0) continue;
            const ix = i * 2;
            const iy = ix + 1;

            const [fx, fy] = this.quadTree.computeForce(
                i, pos[ix], pos[iy], G, theta,
            );

            forces[ix] += fx;
            forces[iy] += fy;
        }
    }

    /**
     * Brute-force O(n²) repulsion. Used when barnesHutTheta = 0
     * or for very small graphs where the overhead isn't worth it.
     */
    private repulsionBruteForce(
        n: number,
        pos: Float32Array,
        sizes: Float32Array,
        forces: Float32Array,
        G: number,
    ): void {
        for (let i = 0; i < n; i++) {
            if (sizes[i] <= 0) continue;
            const ix = i * 2;
            const iy = ix + 1;

            for (let j = i + 1; j < n; j++) {
                if (sizes[j] <= 0) continue;
                const jx = j * 2;
                const jy = jx + 1;

                let dx = pos[ix] - pos[jx];
                let dy = pos[iy] - pos[jy];
                let dist = Math.sqrt(dx * dx + dy * dy);

                if (dist === 0) {
                    dx = 0.1 * Math.random();
                    dy = 0.1 * Math.random();
                    dist = Math.sqrt(dx * dx + dy * dy);
                }

                // G is negative → force pushes nodes apart
                const forceMag = G / (dist * dist);
                const fx = (dx / dist) * forceMag;
                const fy = (dy / dist) * forceMag;

                // Newton's 3rd law
                forces[ix] -= fx;
                forces[iy] -= fy;
                forces[jx] += fx;
                forces[jy] += fy;
            }
        }
    }

    // =========================================================
    // Buffer management
    // =========================================================

    private ensureBuffers(): void {
        const needed = this.graph.numNodes * 2;
        if (this.velocities.length < needed) {
            const oldVel = this.velocities;
            this.velocities = new Float32Array(needed);
            this.velocities.set(oldVel.subarray(0, Math.min(oldVel.length, needed)));
        }
        if (this.forces.length < needed) {
            this.forces = new Float32Array(needed);
        }
    }
}
