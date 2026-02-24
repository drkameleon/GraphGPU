// ============================================================
// graphGPU - Color Utilities
// ============================================================
// Handles all color parsing, conversion, and contrast logic.
// Inspired by graphGPU's idealForeground function.
// ============================================================

import type { ColorInput } from '../types';

/** Normalized RGBA tuple [0-1] */
export type RGBA = [number, number, number, number];

/**
 * Parse any ColorInput into a normalized [r, g, b, a] tuple (0-1 range).
 */
export function parseColor(input: ColorInput): RGBA {
    if (typeof input === 'string') {
        return parseColorString(input);
    }
    if (Array.isArray(input)) {
        return input.length === 3
            ? [input[0], input[1], input[2], 1.0]
            : [input[0], input[1], input[2], input[3]];
    }
    return [input.r, input.g, input.b, input.a ?? 1.0];
}

/**
 * Parse hex or rgb() string into RGBA.
 */
function parseColorString(str: string): RGBA {
    const s = str.trim();

    // #RGB
    if (s.length === 4 && s[0] === '#') {
        const r = parseInt(s[1] + s[1], 16) / 255;
        const g = parseInt(s[2] + s[2], 16) / 255;
        const b = parseInt(s[3] + s[3], 16) / 255;
        return [r, g, b, 1.0];
    }

    // #RRGGBB
    if (s.length === 7 && s[0] === '#') {
        const r = parseInt(s.slice(1, 3), 16) / 255;
        const g = parseInt(s.slice(3, 5), 16) / 255;
        const b = parseInt(s.slice(5, 7), 16) / 255;
        return [r, g, b, 1.0];
    }

    // #RRGGBBAA
    if (s.length === 9 && s[0] === '#') {
        const r = parseInt(s.slice(1, 3), 16) / 255;
        const g = parseInt(s.slice(3, 5), 16) / 255;
        const b = parseInt(s.slice(5, 7), 16) / 255;
        const a = parseInt(s.slice(7, 9), 16) / 255;
        return [r, g, b, a];
    }

    // rgb(r, g, b) or rgba(r, g, b, a)
    const match = s.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)/);
    if (match) {
        return [
            parseInt(match[1]) / 255,
            parseInt(match[2]) / 255,
            parseInt(match[3]) / 255,
            match[4] !== undefined ? parseFloat(match[4]) : 1.0,
        ];
    }

    // Fallback: white
    return [1.0, 1.0, 1.0, 1.0];
}

/**
 * Convert RGBA to hex string.
 */
export function rgbaToHex(rgba: RGBA): string {
    const [r, g, b] = rgba;
    const toHex = (v: number) => Math.round(v * 255).toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/**
 * Compute relative luminance (WCAG 2.0).
 * Port of graphGPU's idealForeground logic.
 */
export function luminance(rgba: RGBA): number {
    const linearize = (c: number) =>
        c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;

    const rLin = linearize(rgba[0]);
    const gLin = linearize(rgba[1]);
    const bLin = linearize(rgba[2]);

    return rLin * 0.2126 + gLin * 0.7152 + bLin * 0.0722;
}

/**
 * Get ideal foreground color (black or white) for given background.
 * Direct port of graphGPU's idealForeground function.
 */
export function idealForeground(bg: RGBA): RGBA {
    return luminance(bg) > 0.189
        ? [0, 0, 0, 1]   // black
        : [1, 1, 1, 1];  // white
}

/**
 * Darken a color by a factor (0-1).
 * Port of graphGPU's `darken` used in styleNode.
 */
export function darken(rgba: RGBA, amount: number): RGBA {
    const factor = 1 - amount;
    return [
        rgba[0] * factor,
        rgba[1] * factor,
        rgba[2] * factor,
        rgba[3],
    ];
}

/**
 * Lighten a color by a factor (0-1).
 */
export function lighten(rgba: RGBA, amount: number): RGBA {
    return [
        rgba[0] + (1 - rgba[0]) * amount,
        rgba[1] + (1 - rgba[1]) * amount,
        rgba[2] + (1 - rgba[2]) * amount,
        rgba[3],
    ];
}

/**
 * Write RGBA into a Float32Array at a given offset.
 * Hot path - no allocations.
 */
export function writeColorToBuffer(
    buffer: Float32Array,
    offset: number,
    rgba: RGBA,
): void {
    buffer[offset] = rgba[0];
    buffer[offset + 1] = rgba[1];
    buffer[offset + 2] = rgba[2];
    buffer[offset + 3] = rgba[3];
}
