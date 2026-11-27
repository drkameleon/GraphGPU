// ============================================================
// graphGPU - Benchmark Runner
// ============================================================
// Usage:
//   npm run bench
//   npm run bench -- --filter Layout --time 2000
//   npm run bench -- --json bench-head.json
// ============================================================

import { writeFileSync } from 'node:fs';
import { Bench } from 'tinybench';
import type { BenchReport, BenchSuite } from './types';
import { formatMs } from './utils';

import graph from './graph';
import quadtree from './quadtree';
import layout from './layout';
import renderer from './renderer';

const SUITES: BenchSuite[] = [graph, quadtree, layout, renderer];

// -----------------------------------------------------------
// CLI options
// -----------------------------------------------------------

function option(name: string): string | undefined {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 ? process.argv[i + 1] : undefined;
}

const jsonPath = option('json');
const filter = option('filter')?.toLowerCase();
const time = Number(option('time') ?? 1000);

// -----------------------------------------------------------
// Run
// -----------------------------------------------------------

async function main(): Promise<void> {
    const report: BenchReport = {
        meta: {
            node: process.version,
            platform: process.platform,
            arch: process.arch,
            date: new Date().toISOString(),
        },
        results: {},
    };

    let failed = false;

    for (const suite of SUITES) {
        const cases = suite.cases.filter(c =>
            !filter || `${suite.name} › ${c.name}`.toLowerCase().includes(filter),
        );
        if (cases.length === 0) continue;

        await suite.setup?.();

        const bench = new Bench({
            name: suite.name,
            time,
            iterations: 5,
            warmupTime: Math.min(250, time),
            warmupIterations: 2,
        });

        for (const c of cases) {
            bench.add(c.name, c.fn, { beforeEach: c.beforeEach });
        }

        await bench.run();

        console.log(`\n${suite.name}`);
        for (const task of bench.tasks) {
            const key = `${suite.name} › ${task.name}`;
            const result = task.result;

            if (result.state !== 'completed') {
                failed = true;
                const reason = result.state === 'errored' ? result.error.message : result.state;
                console.log(`  ✗ ${task.name}: ${reason}`);
                continue;
            }

            const { mean, p50, p99, rme, samplesCount } = result.latency;
            report.results[key] = { mean, p50, p99, rme, samples: samplesCount };

            console.log(
                `  ${task.name.padEnd(36)}` +
                `${formatMs(mean).padStart(11)}` +
                `  ±${rme.toFixed(1).padStart(4)}%` +
                `  p99 ${formatMs(p99).padStart(10)}` +
                `  (${samplesCount} samples)`,
            );
        }
    }

    if (jsonPath) {
        writeFileSync(jsonPath, JSON.stringify(report, null, 2) + '\n');
        console.log(`\nResults written to ${jsonPath}`);
    }

    if (failed) process.exit(1);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
