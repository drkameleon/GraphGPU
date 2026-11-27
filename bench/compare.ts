// ============================================================
// graphGPU - Benchmark Comparison
// ============================================================
// Compares base vs head benchmark reports and prints a
// Markdown table. Each side may have several reports (rounds),
// ideally recorded alternately (base, head, base, head, ...)
// so that machine drift affects both sides equally.
//
// Usage:
//   tsx bench/compare.ts --base bench-base-*.json --head bench-head-*.json
//                        [--warn 1.1] [--fail 1.5]
//
// Per case, the median latency (p50) of every round is taken,
// and the median across rounds is compared. A difference only
// counts if it exceeds the noise: the margin of error of the
// runs plus the spread between rounds.
//
// Exits with code 1 if any case is slower than the --fail ratio.
// ============================================================

import { readFileSync } from 'node:fs';
import type { BenchReport } from './types';
import { formatChange, formatMs } from './utils';

// -----------------------------------------------------------
// CLI options
// -----------------------------------------------------------

function files(flag: string): string[] {
    const i = process.argv.indexOf(`--${flag}`);
    if (i < 0) return [];
    const out: string[] = [];
    for (let j = i + 1; j < process.argv.length && !process.argv[j].startsWith('--'); j++) {
        out.push(process.argv[j]);
    }
    return out;
}

function option(name: string, fallback: number): number {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 ? Number(process.argv[i + 1]) : fallback;
}

const baseFiles = files('base');
const headFiles = files('head');
if (baseFiles.length === 0 || headFiles.length === 0) {
    console.error('usage: tsx bench/compare.ts --base <json...> --head <json...> [--warn 1.1] [--fail 1.5]');
    process.exit(2);
}

const warn = option('warn', 1.1);
const fail = option('fail', 1.5);

const load = (paths: string[]): BenchReport[] =>
    paths.map(p => JSON.parse(readFileSync(p, 'utf8')) as BenchReport);

const baseReports = load(baseFiles);
const headReports = load(headFiles);

// -----------------------------------------------------------
// Aggregation across rounds
// -----------------------------------------------------------

interface Aggregate {
    /** Median across rounds of the per-round median latency */
    value: number;
    /** Relative noise: mean margin of error + half the round spread */
    noise: number;
}

function median(values: number[]): number {
    const s = [...values].sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function aggregate(reports: BenchReport[], name: string): Aggregate | undefined {
    const runs = reports.map(r => r.results[name]).filter(Boolean);
    if (runs.length === 0) return undefined;

    const p50s = runs.map(r => r.p50);
    const value = median(p50s);
    const rme = runs.reduce((sum, r) => sum + r.rme, 0) / runs.length / 100;
    const spread = (Math.max(...p50s) - Math.min(...p50s)) / 2 / value;
    return { value, noise: rme + spread };
}

// -----------------------------------------------------------
// Classification
// -----------------------------------------------------------

type Verdict = 'faster' | 'slower' | 'regression' | 'same' | 'new' | 'removed';

const LABELS: Record<Verdict, string> = {
    faster: '🟢 faster',
    slower: '🟠 slower',
    regression: '🔴 regression',
    same: '≈',
    new: 'new',
    removed: 'removed',
};

function classify(b: Aggregate | undefined, h: Aggregate | undefined): { verdict: Verdict; ratio: number } {
    if (!b) return { verdict: 'new', ratio: NaN };
    if (!h) return { verdict: 'removed', ratio: NaN };

    const ratio = h.value / b.value;
    const significant = Math.abs(ratio - 1) > b.noise + h.noise;

    if (!significant) return { verdict: 'same', ratio };
    if (ratio >= fail) return { verdict: 'regression', ratio };
    if (ratio >= warn) return { verdict: 'slower', ratio };
    if (ratio <= 1 / warn) return { verdict: 'faster', ratio };
    return { verdict: 'same', ratio };
}

// -----------------------------------------------------------
// Report
// -----------------------------------------------------------

const cell = (a: Aggregate | undefined): string =>
    a ? `${formatMs(a.value)} ±${(a.noise * 100).toFixed(1)}%` : '—';

const names = [...new Set([...baseReports, ...headReports].flatMap(r => Object.keys(r.results)))];
let regressions = 0;

const meta = headReports[0].meta;
console.log('### Benchmarks\n');
console.log(
    `Node ${meta.node} on ${meta.platform}/${meta.arch}, ` +
    `${baseReports.length} base / ${headReports.length} head round(s). ` +
    `Median latency, lower is better; changes within the noise are shown as ≈.\n`,
);
console.log('| Benchmark | Base | Head | Change | |');
console.log('|---|---:|---:|---:|---|');

for (const name of names) {
    const b = aggregate(baseReports, name);
    const h = aggregate(headReports, name);
    const { verdict, ratio } = classify(b, h);
    if (verdict === 'regression') regressions++;

    console.log(
        `| ${name} | ${cell(b)} | ${cell(h)} ` +
        `| ${Number.isNaN(ratio) ? '—' : formatChange(ratio)} | ${LABELS[verdict]} |`,
    );
}

if (regressions > 0) {
    console.log(`\n**${regressions} regression(s)**, at least ${fail}× slower than base.`);
    process.exit(1);
}
