// ============================================================
// graphGPU - Layout option resolution
// ============================================================

/**
 * Merge layout options (later sources win), skipping unset
 * (undefined) values so they never override a default, and map
 * the legacy names (repulsion, attraction, gravity).
 */
export function resolveLayoutOptions<T extends object>(...sources: (object | undefined)[]): T {
    const out: Record<string, unknown> = {};
    for (const src of sources) {
        if (!src) continue;
        for (const [key, value] of Object.entries(src)) {
            if (value !== undefined) out[key] = value;
        }
    }

    const legacy: [string, string, (v: number) => number][] = [
        // Repulsion is a magnitude; the constant is negative = repulsive
        ['repulsion', 'gravitationalConstant', (v) => -Math.abs(v)],
        ['attraction', 'springConstant', (v) => v],
        ['gravity', 'centralGravity', (v) => v],
    ];
    for (const [old, name, map] of legacy) {
        if (typeof out[old] === 'number' && out[name] === undefined) {
            out[name] = map(out[old] as number);
        }
        delete out[old];
    }
    return out as T;
}
