// ============================================================
// graphGPU - Benchmark Types
// ============================================================

export interface BenchCase {
    /** Case name, unique within its suite */
    name: string;
    /** The measured function */
    fn: () => void;
    /** Runs before every iteration (not measured) */
    beforeEach?: () => void;
}

export interface BenchSuite {
    name: string;
    /** Runs once before the suite (e.g. async initialization) */
    setup?: () => Promise<void> | void;
    cases: BenchCase[];
}

/** Per-case statistics, latency in milliseconds */
export interface BenchStats {
    mean: number;
    p50: number;
    p99: number;
    rme: number;
    samples: number;
}

export interface BenchReport {
    meta: {
        node: string;
        platform: string;
        arch: string;
        date: string;
    };
    results: Record<string, BenchStats>;
}
