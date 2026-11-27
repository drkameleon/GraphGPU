// ============================================================
// graphGPU - QuadTree tests
// ============================================================

import { describe, it, expect } from 'vitest';
import { QuadTree } from '../src/layout/QuadTree';
import { seededRandom } from './helpers';

const G = -0.25;

/** Exact O(n²) reference: same formula as the tree's leaf case */
function exactForce(
    i: number,
    pos: Float32Array,
    sizes: Float32Array,
    n: number,
): [number, number] {
    let fx = 0, fy = 0;
    for (let j = 0; j < n; j++) {
        if (j === i || sizes[j] <= 0) continue;
        const dx = pos[i * 2] - pos[j * 2];
        const dy = pos[i * 2 + 1] - pos[j * 2 + 1];
        const distSq = dx * dx + dy * dy + 1e-8;
        const dist = Math.sqrt(distSq);
        const mag = G / distSq;
        fx -= (dx / dist) * mag;
        fy -= (dy / dist) * mag;
    }
    return [fx, fy];
}

function randomBodies(n: number, seed: number) {
    const rand = seededRandom(seed);
    const pos = new Float32Array(n * 2);
    const sizes = new Float32Array(n).fill(1);
    for (let i = 0; i < n * 2; i++) pos[i] = rand() * 2 - 1;
    return { pos, sizes };
}

describe('QuadTree', () => {
    it('returns zero force on an empty tree', () => {
        const qt = new QuadTree();
        qt.build(new Float32Array(0), new Float32Array(0), 0);
        expect(qt.computeForce(0, 0, 0, G, 0.5)).toEqual([0, 0]);
    });

    it('returns zero force on a lone body', () => {
        const qt = new QuadTree();
        qt.build(new Float32Array([0.3, 0.4]), new Float32Array([1]), 1);
        expect(qt.computeForce(0, 0.3, 0.4, G, 0.5)).toEqual([0, 0]);
    });

    it('pushes two bodies apart with equal and opposite force', () => {
        const pos = new Float32Array([-0.5, 0, 0.5, 0]);
        const sizes = new Float32Array([1, 1]);
        const qt = new QuadTree();
        qt.build(pos, sizes, 2);

        const [f0x, f0y] = qt.computeForce(0, -0.5, 0, G, 0.5);
        const [f1x, f1y] = qt.computeForce(1, 0.5, 0, G, 0.5);

        // Body 0 is on the left, so it is pushed further left
        expect(f0x).toBeLessThan(0);
        expect(f1x).toBeGreaterThan(0);
        expect(f0x).toBeCloseTo(-f1x);
        expect(f0y).toBeCloseTo(0);
        expect(f1y).toBeCloseTo(0);

        // |F| = |G| / d², d = 1
        expect(Math.abs(f0x)).toBeCloseTo(Math.abs(G), 5);
    });

    it('ignores inactive bodies (size <= 0)', () => {
        const pos = new Float32Array([-0.5, 0, 0.5, 0, 0.49, 0.01]);
        const qt = new QuadTree();

        qt.build(pos, new Float32Array([1, 1, 0]), 3);
        const withInactive = qt.computeForce(0, -0.5, 0, G, 0);

        qt.build(pos, new Float32Array([1, 1]), 2);
        const withoutInactive = qt.computeForce(0, -0.5, 0, G, 0);

        expect(withInactive[0]).toBeCloseTo(withoutInactive[0]);
        expect(withInactive[1]).toBeCloseTo(withoutInactive[1]);
    });

    it('matches the exact sum when theta = 0', () => {
        const n = 200;
        const { pos, sizes } = randomBodies(n, 7);
        const qt = new QuadTree();
        qt.build(pos, sizes, n);

        for (let i = 0; i < n; i += 17) {
            const [fx, fy] = qt.computeForce(i, pos[i * 2], pos[i * 2 + 1], G, 0);
            const [ex, ey] = exactForce(i, pos, sizes, n);
            expect(fx).toBeCloseTo(ex, 3);
            expect(fy).toBeCloseTo(ey, 3);
        }
    });

    it('approximates the exact sum when theta > 0', () => {
        const n = 2000;
        const { pos, sizes } = randomBodies(n, 42);
        const qt = new QuadTree();
        qt.build(pos, sizes, n);

        let errSum = 0, magSum = 0;
        for (let i = 0; i < n; i += 13) {
            const [fx, fy] = qt.computeForce(i, pos[i * 2], pos[i * 2 + 1], G, 0.5);
            const [ex, ey] = exactForce(i, pos, sizes, n);
            errSum += Math.hypot(fx - ex, fy - ey);
            magSum += Math.hypot(ex, ey);
        }
        // Aggregate relative error stays small
        expect(errSum / magSum).toBeLessThan(0.05);
    });

    it('handles more bodies than the initial pool size', () => {
        const n = 20000;
        const { pos, sizes } = randomBodies(n, 3);
        const qt = new QuadTree();
        expect(() => qt.build(pos, sizes, n)).not.toThrow();

        const [fx, fy] = qt.computeForce(0, pos[0], pos[1], G, 0.5);
        expect(Number.isFinite(fx)).toBe(true);
        expect(Number.isFinite(fy)).toBe(true);
    });

    it('handles bodies at the exact same position', () => {
        const pos = new Float32Array([0.5, 0.5, 0.5, 0.5]);
        const sizes = new Float32Array([1, 1]);
        const qt = new QuadTree();
        expect(() => qt.build(pos, sizes, 2)).not.toThrow();

        const [f0x, f0y] = qt.computeForce(0, 0.5, 0.5, G, 0.5);
        const [f1x, f1y] = qt.computeForce(1, 0.5, 0.5, G, 0.5);

        // Pushed apart: finite, non-zero, equal and opposite
        expect(Number.isFinite(f0x) && Number.isFinite(f0y)).toBe(true);
        expect(Math.hypot(f0x, f0y)).toBeGreaterThan(0);
        expect(f1x).toBeCloseTo(-f0x);
        expect(f1y).toBeCloseTo(-f0y);
    });

    it('handles many coincident bodies among others', () => {
        const n = 500;
        const { pos, sizes } = randomBodies(n, 5);
        for (let i = 0; i < 200; i++) {
            pos[i * 2] = 0.1;
            pos[i * 2 + 1] = -0.2;
        }
        const qt = new QuadTree();
        expect(() => qt.build(pos, sizes, n)).not.toThrow();

        for (const i of [0, 199, 200, 499]) {
            const [fx, fy] = qt.computeForce(i, pos[i * 2], pos[i * 2 + 1], G, 0.5);
            expect(Number.isFinite(fx)).toBe(true);
            expect(Number.isFinite(fy)).toBe(true);
        }

        // The group as a whole repels a body outside of it
        const [fx] = qt.computeForce(n, 0.1 + 0.5, -0.2, G, 0);
        expect(fx).toBeGreaterThan(0);
    });

    it('handles bodies closer than float precision can split', () => {
        const a = Math.fround(0.3);
        const b = a + 2 ** -25; // rounds to a neighbor of a in float32
        const pos = new Float32Array([a, a, b, b, -0.5, 0.5]);
        const qt = new QuadTree();
        expect(() => qt.build(pos, new Float32Array([1, 1, 1]), 3)).not.toThrow();

        const [fx, fy] = qt.computeForce(2, -0.5, 0.5, G, 0);
        expect(Number.isFinite(fx)).toBe(true);
        expect(Number.isFinite(fy)).toBe(true);
    });

    it('skips bodies with non-finite positions', () => {
        const pos = new Float32Array([-0.5, 0, 0.5, 0, NaN, NaN, Infinity, 0, NaN, 1]);
        const qt = new QuadTree();
        expect(() => qt.build(pos, new Float32Array(5).fill(1), 5)).not.toThrow();

        const [fx, fy] = qt.computeForce(0, -0.5, 0, G, 0);
        expect(fx).toBeCloseTo(G, 5);
        expect(fy).toBeCloseTo(0);
    });

    it('returns zero force when every body is inactive or non-finite', () => {
        const qt = new QuadTree();
        qt.build(new Float32Array([NaN, NaN, 0, 0]), new Float32Array([1, 0]), 2);
        expect(qt.computeForce(0, 0, 0, G, 0.5)).toEqual([0, 0]);
    });

    it('can be rebuilt repeatedly with different inputs', () => {
        const qt = new QuadTree();
        const big = randomBodies(5000, 9);
        qt.build(big.pos, big.sizes, 5000);

        const pos = new Float32Array([-0.5, 0, 0.5, 0]);
        qt.build(pos, new Float32Array([1, 1]), 2);
        const [fx] = qt.computeForce(0, -0.5, 0, G, 0.5);
        expect(Math.abs(fx)).toBeCloseTo(Math.abs(G), 5);
    });
});
