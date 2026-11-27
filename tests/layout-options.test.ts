// ============================================================
// graphGPU - Layout option resolution tests
// ============================================================

import { describe, it, expect } from 'vitest';
import { Graph } from '../src/core/Graph';
import { ForceLayout } from '../src/layout/ForceLayout';
import { resolveLayoutOptions } from '../src/layout/options';

describe('resolveLayoutOptions', () => {
    it('lets later sources win', () => {
        expect(resolveLayoutOptions({ damping: 0.1, timestep: 1 }, { damping: 0.2 }))
            .toEqual({ damping: 0.2, timestep: 1 });
    });

    it('never lets an unset value override', () => {
        expect(resolveLayoutOptions({ damping: 0.1 }, { damping: undefined }))
            .toEqual({ damping: 0.1 });
        expect(resolveLayoutOptions({ damping: undefined })).toEqual({});
    });

    it('skips missing sources', () => {
        expect(resolveLayoutOptions(undefined, { damping: 0.1 }, undefined)).toEqual({ damping: 0.1 });
    });

    it('maps the legacy names', () => {
        expect(resolveLayoutOptions({ repulsion: 0.5, attraction: 0.02, gravity: 0.001 })).toEqual({
            gravitationalConstant: -0.5,
            springConstant: 0.02,
            centralGravity: 0.001,
        });
    });

    it('prefers the current names over the legacy ones', () => {
        expect(resolveLayoutOptions({ gravity: 0.5, centralGravity: 0.1 })).toEqual({ centralGravity: 0.1 });
    });
});

describe('ForceLayout: legacy options', () => {
    function drift(opts: object): number {
        const g = new Graph();
        g.addNode({ tag: 'n', x: 0.5, y: 0 });
        new ForceLayout(g, { type: 'force', ...opts }).step(10);
        return Math.abs(g.positions[0] - 0.5);
    }

    it('honors gravity (as centralGravity)', () => {
        expect(drift({})).toBeGreaterThan(0);
        expect(drift({ gravity: 0 })).toBe(0);
    });

    it('ignores unset values', () => {
        expect(drift({ centralGravity: undefined, damping: undefined })).toBe(drift({}));
    });
});
