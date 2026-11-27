// ============================================================
// graphGPU - Benchmark Utilities
// ============================================================

/** Human-readable duration from milliseconds */
export function formatMs(ms: number): string {
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (ms >= 1) return `${ms.toFixed(2)} ms`;
    return `${(ms * 1000).toFixed(1)} µs`;
}

/** Signed percentage change, e.g. "+12.3%" / "−4.0%" */
export function formatChange(ratio: number): string {
    const pct = (ratio - 1) * 100;
    const sign = pct > 0 ? '+' : pct < 0 ? '−' : '';
    return `${sign}${Math.abs(pct).toFixed(1)}%`;
}
