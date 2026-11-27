// ============================================================
// graphGPU - Color & Palette tests
// ============================================================

import { describe, it, expect } from 'vitest';
import {
    parseColor, rgbaToHex, luminance,
    idealForeground, darken, lighten, writeColorToBuffer,
} from '../src/utils/color';
import { TagColorMap, PALETTES } from '../src/utils/palette';

describe('parseColor', () => {
    it('parses #rgb', () => {
        expect(parseColor('#f00')).toEqual([1, 0, 0, 1]);
    });

    it('parses #rrggbb', () => {
        const [r, g, b, a] = parseColor('#ff8000');
        expect(r).toBe(1);
        expect(g).toBeCloseTo(128 / 255);
        expect(b).toBe(0);
        expect(a).toBe(1);
    });

    it('parses #rrggbbaa', () => {
        expect(parseColor('#00ff0080')[3]).toBeCloseTo(128 / 255);
    });

    it('parses rgb() and rgba()', () => {
        expect(parseColor('rgb(255, 0, 0)')).toEqual([1, 0, 0, 1]);
        expect(parseColor('rgba(0,0,255,0.5)')).toEqual([0, 0, 1, 0.5]);
    });

    it('trims whitespace', () => {
        expect(parseColor('  #fff  ')).toEqual([1, 1, 1, 1]);
    });

    it('accepts arrays and objects', () => {
        expect(parseColor([0.1, 0.2, 0.3])).toEqual([0.1, 0.2, 0.3, 1]);
        expect(parseColor([0.1, 0.2, 0.3, 0.4])).toEqual([0.1, 0.2, 0.3, 0.4]);
        expect(parseColor({ r: 0.5, g: 0.5, b: 0.5 })).toEqual([0.5, 0.5, 0.5, 1]);
        expect(parseColor({ r: 0, g: 0, b: 0, a: 0 })).toEqual([0, 0, 0, 0]);
    });

    it('falls back to white for unknown strings', () => {
        expect(parseColor('not-a-color')).toEqual([1, 1, 1, 1]);
    });
});

describe('color helpers', () => {
    it('round-trips hex', () => {
        expect(rgbaToHex(parseColor('#3d0075'))).toBe('#3d0075');
    });

    it('computes luminance', () => {
        expect(luminance([0, 0, 0, 1])).toBe(0);
        expect(luminance([1, 1, 1, 1])).toBeCloseTo(1);
    });

    it('picks a readable foreground', () => {
        expect(idealForeground([1, 1, 1, 1])).toEqual([0, 0, 0, 1]);
        expect(idealForeground([0, 0, 0.2, 1])).toEqual([1, 1, 1, 1]);
    });

    it('darkens and lightens, keeping alpha', () => {
        expect(darken([1, 0.5, 0, 0.3], 0.5)).toEqual([0.5, 0.25, 0, 0.3]);
        expect(lighten([0, 0.5, 1, 0.3], 0.5)).toEqual([0.5, 0.75, 1, 0.3]);
    });

    it('writes into a buffer at an offset', () => {
        const buf = new Float32Array(8);
        writeColorToBuffer(buf, 4, [0.25, 0.5, 0.75, 1]);
        expect(Array.from(buf)).toEqual([0, 0, 0, 0, 0.25, 0.5, 0.75, 1]);
    });
});

describe('TagColorMap', () => {
    it('assigns a stable color per tag', () => {
        const map = new TagColorMap();
        expect(map.getColor('a')).toBe(map.getColor('a'));
    });

    it('assigns distinct colors until the palette is exhausted', () => {
        const map = new TagColorMap('pastel');
        const n = PALETTES.pastel.colors.length;
        const seen = new Set<string>();
        for (let i = 0; i < n; i++) seen.add(rgbaToHex(map.getColor(`t${i}`).bg));
        expect(seen.size).toBe(n);
    });

    it('cycles once the palette is exhausted', () => {
        const map = new TagColorMap('pastel');
        const n = PALETTES.pastel.colors.length;
        for (let i = 0; i < n; i++) map.getColor(`t${i}`);
        expect(() => map.getColor('overflow')).not.toThrow();
        expect(PALETTES.pastel.colors.map(c => c.toLowerCase()))
            .toContain(rgbaToHex(map.getColor('overflow').bg));
    });

    it('falls back to the default palette for unknown names', () => {
        expect(new TagColorMap('nope').getPalette()).toBe(PALETTES.default);
    });

    it('accepts custom palettes', () => {
        const map = new TagColorMap({ name: 'mine', colors: ['#123456'] });
        expect(rgbaToHex(map.getColor('x').bg)).toBe('#123456');
    });

    it('allows manual overrides', () => {
        const map = new TagColorMap();
        map.setColor('x', '#ff0000');
        expect(map.getColor('x').bg).toEqual([1, 0, 0, 1]);
    });

    it('resets assignments when switching palette', () => {
        const map = new TagColorMap('pastel');
        map.getColor('a');
        map.setPalette('vibrant');
        expect(map.getAll().size).toBe(0);
        expect(rgbaToHex(map.getColor('a').bg)).toBe(PALETTES.vibrant.colors[0].toLowerCase());
    });

    it('derives foreground and border colors', () => {
        const { bg, fg, border } = new TagColorMap().getColor('a');
        expect(fg).toEqual(idealForeground(bg));
        expect(border).toEqual(darken(bg, 0.2));
    });
});

describe('PALETTES', () => {
    it('contains only valid hex colors', () => {
        for (const [key, palette] of Object.entries(PALETTES)) {
            expect(palette.name).toBe(key);
            expect(palette.colors.length).toBeGreaterThan(0);
            for (const c of palette.colors) {
                expect(c).toMatch(/^#[0-9a-fA-F]{6}$/);
            }
        }
    });
});
