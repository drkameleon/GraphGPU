// ============================================================
// graphGPU - Palette System
// ============================================================
// Color palettes for automatic node coloring by tag.
// Ported from graphGPU's .art palette files.
// ============================================================

import type { Palette } from '../types';
import { parseColor, idealForeground, darken, type RGBA } from './color';

// -----------------------------------------------------------
// Built-in palettes (ported from graphGPU's ui/palettes/*.art)
// -----------------------------------------------------------

export const PALETTES: Record<string, Palette> = {
    default: {
        name: 'default',
        colors: [
            '#130075', '#3d0075', '#bd8ed1', '#75250e', '#756c10',
            '#567513', '#A5D180', '#00752B', '#005f73', '#0a9396',
            '#94d2bd', '#e9d8a6', '#d1ad94', '#ee9b00', '#ca6702',
            '#bb3e03', '#ae2012', '#9b2226', '#750041',
        ],
    },
    pastel: {
        name: 'pastel',
        colors: [
            '#3bb2d0', '#3887be', '#8a8acb', '#56b881', '#50667f',
            '#41afa5', '#f9886c', '#e55e5e', '#ed6498', '#fbb03b',
        ],
    },
    vibrant: {
        name: 'vibrant',
        colors: [
            '#ff6050', '#ff0e83', '#d54262', '#813867', '#66336e',
            '#5677fc', '#341539', '#9013fe', '#ffc208', '#00cc99',
        ],
    },
    earthy: {
        name: 'earthy',
        colors: [
            '#c27e88', '#a9505c', '#c3887d', '#aa5d4e', '#7a4238',
            '#b2c77a', '#94b049', '#6a7d34', '#90b1b1', '#679292',
            '#496868', '#ae9d92', '#8f786a', '#66564b', '#3d332d',
        ],
    },
    inferno: {
        name: 'inferno',
        colors: [
            '#fcffa4', '#f2ea69', '#f8cd37', '#fcb014', '#fa9407',
            '#f47918', '#e9612b', '#d94d3d', '#c63d4d', '#b0315b',
            '#982766', '#801f6c', '#69166e', '#510e6c', '#380962',
        ],
    },
    playful: {
        name: 'playful',
        colors: [
            '#7ac70c', '#8ee000', '#faa918', '#ffc715', '#d33131',
            '#e53838', '#1cb0f6', '#14d4f4', '#8549ba', '#a560e8',
            '#4c4c4c', '#6f6f6f',
        ],
    },
    viridis: {
        name: 'viridis',
        colors: [
            '#fde725', '#d0e11c', '#a0da39', '#73d056', '#4ac16d',
            '#2db27d', '#1fa187', '#21918c', '#277f8e', '#2e6e8e',
            '#365c8d', '#3f4788', '#46327e', '#481b6d', '#440154',
        ],
    },
    rainbow: {
        name: 'rainbow',
        colors: [
            '#4aae20', '#6cc751', '#225f1e', '#f8cc07', '#ff9300',
            '#e44436', '#cd3292', '#37aef3', '#0076ba', '#005888',
            '#8732cd',
        ],
    },
};

// -----------------------------------------------------------
// Tag → Color assignment (mirrors graphGPU's nodeColors logic)
// -----------------------------------------------------------

export interface TagColorAssignment {
    bg: RGBA;
    fg: RGBA;        // auto-computed ideal foreground
    border: RGBA;    // darkened bg
}

export class TagColorMap {
    private assignments = new Map<string, TagColorAssignment>();
    private usedIndices = new Set<number>();
    private palette: Palette;

    constructor(palette: Palette | string = 'default') {
        this.palette = typeof palette === 'string'
            ? (PALETTES[palette] ?? PALETTES.default)
            : palette;
    }

    /** Get or auto-assign a color for a tag */
    getColor(tag: string): TagColorAssignment {
        const existing = this.assignments.get(tag);
        if (existing) return existing;

        // Find next unused color (cycle if exhausted)
        const colors = this.palette.colors;
        let idx = 0;
        for (let i = 0; i < colors.length; i++) {
            if (!this.usedIndices.has(i)) {
                idx = i;
                break;
            }
            if (i === colors.length - 1) {
                // All used - cycle with offset
                idx = this.assignments.size % colors.length;
            }
        }
        this.usedIndices.add(idx);

        const bg = parseColor(colors[idx]);
        const assignment: TagColorAssignment = {
            bg,
            fg: idealForeground(bg),
            border: darken(bg, 0.2),
        };

        this.assignments.set(tag, assignment);
        return assignment;
    }

    /** Manually assign a color to a tag */
    setColor(tag: string, color: string): void {
        const bg = parseColor(color);
        this.assignments.set(tag, {
            bg,
            fg: idealForeground(bg),
            border: darken(bg, 0.2),
        });
    }

    /** Switch to a different palette (resets assignments) */
    setPalette(palette: Palette | string): void {
        this.palette = typeof palette === 'string'
            ? (PALETTES[palette] ?? PALETTES.default)
            : palette;
        this.assignments.clear();
        this.usedIndices.clear();
    }

    /** Get current palette */
    getPalette(): Palette {
        return this.palette;
    }

    /** Get all current assignments */
    getAll(): Map<string, TagColorAssignment> {
        return new Map(this.assignments);
    }

    /** Clear all assignments */
    clear(): void {
        this.assignments.clear();
        this.usedIndices.clear();
    }
}
