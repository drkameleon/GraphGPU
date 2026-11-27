// ============================================================
// graphGPU - Benchmark Comparison
// ============================================================
// Compares two benchmark reports and prints a Markdown table.
//
// Usage:
//   tsx bench/compare.ts bench-base.json bench-head.json [--warn 1.1] [--fail 1.5]
//
// A case only counts as faster/slower when the difference is
// larger than the combined margin of error of both runs.
// Exits with code 1 if any case is slower than the --fail ratio.
// ============================================================

import { readFileSync } from 'node:fs';
import type { BenchReport, BenchStats } from './types';
import { formatChange, formatMs } from './utils';

function option(name: string, fallback: number): number {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 ? Number(process.argv[i + 1]) : fallback;
}

const [basePath, headPath] = process.argv.slice(2).filter(a => a.endsWith('.json'));
if (!basePath || !headPath) {
    console.error('usage: tsx bench/compare.ts <base.json> <head.json> [--warn 1.1] [--fail 1.5]');
    process.exit(2);
}

const warn = option('warn', 1.1);
const fail = option('fail', 1.5);

const base: BenchReport = JSON.parse(readFileSync(basePath, 'utf8'));
const head: BenchReport = JSON.parse(readFileSync(headPath, 'utf8'));

// -----------------------------------------------------------
// Classification
// -----------------------------------------------------------

type Verdict = 'faster' | 'slower' | 'regression' | 'same' | 'new' | 'removed';

function classify(b: BenchStats | undefined, h: BenchStats | undefined): { verdict: Verdict; ratio: number } {
    if (!b) return { verdict: 'new', ratio: NaN };
    if (!h) return { verdict: 'removed', ratio: NaN };

    const ratio = h.mean / b.mean;
    const noise = ((b.rme + h.rme) / 100) * b.mean;
    const significant = Math.abs(h.mean - b.mean) > noise;

    if (!significant) return { verdict: 'same', ratio };
    if (ratio >= fail) return { verdict: 'regression', ratio };
    if (ratio >= warn) return { verdict: 'slower', ratio };
    if (ratio <= 1 / warn) return { verdict: 'faster', ratio };
    return { verdict: 'same', ratio };
}

const LABELS: Record<Verdict, string> = {
    faster: '🟢 faster',
    slower: '🟠 slower',
    regression: '🔴 regression',
    same: '≈',
    new: 'new',
    removed: 'removed',
};

// -----------------------------------------------------------
// Report
// -----------------------------------------------------------

const names = [...new Set([...Object.keys(base.results), ...Object.keys(head.results)])];
const rows: string[] = [];
let regressions = 0;

for (const name of names) {
    const b = base.results[name];
    const h = head.results[name];
    const { verdict, ratio } = classify(b, h);
    if (verdict === 'regression') regressions++;

    rows.push(
        `| ${name} ` +
        `| ${b ? `${formatMs(b.mean)} ±${b.rme.toFixed(1)}%` : '—'} ` +
        `| ${h ? `${formatMs(h.mean)} ±${h.rme.toFixed(1)}%` : '—'} ` +
        `| ${Number.isNaN(ratio) ? '—' : formatChange(ratio)} ` +
        `| ${LABELS[verdict]} |`,
    );
}

console.log('### Benchmarks\n');
console.log(`Node ${head.meta.node} on ${head.meta.platform}/${head.meta.arch}. ` +
    `Lower is better; changes within the margin of error are shown as ≈.\n`);
console.log('| Benchmark | Base | Head | Change | |');
console.log('|---|---:|---:|---:|---|');
for (const row of rows) console.log(row);

if (regressions > 0) {
    console.log(`\n**${regressions} regression(s)** slower than ${fail}× the base.`);
    process.exit(1);
}
