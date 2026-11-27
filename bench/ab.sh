#!/usr/bin/env bash
# ============================================================
# graphGPU - A/B Benchmark
# ============================================================
# Benchmarks src/ at <base-ref> against the current src/
# (uncommitted changes included), alternating rounds so that
# machine drift affects both sides equally. Both sides run
# the current benchmark suite.
#
# Usage:
#   bash bench/ab.sh <base-ref> [rounds]
#   npm run bench:compare -- main
#   BENCH_FILTER=QuadTree npm run bench:compare -- main
#
# Prints a Markdown comparison to stdout (benchmark logs go
# to stderr) and exits with 1 on a regression.
# ============================================================

set -euo pipefail

REF=${1:?usage: bash bench/ab.sh <base-ref> [rounds]}
ROUNDS=${2:-3}
FILTER=()
if [ -n "${BENCH_FILTER:-}" ]; then
    FILTER=(--filter "$BENCH_FILTER")
fi

cd "$(dirname "$0")/.."

HEAD_SRC=$(mktemp -d)
cp -R src "$HEAD_SRC/src"

use_base() {
    rm -rf src
    git archive "$REF" src | tar -x
}

use_head() {
    rm -rf src
    cp -R "$HEAD_SRC/src" src
}

cleanup() {
    use_head
    rm -rf "$HEAD_SRC"
}
trap cleanup EXIT

rm -f bench-base-*.json bench-head-*.json
BASE_OK=1

for ((i = 1; i <= ROUNDS; i++)); do
    echo "── round $i/$ROUNDS: base ($REF)" >&2
    use_base
    if ! npx tsx bench/run.ts ${FILTER[@]+"${FILTER[@]}"} --json "bench-base-$i.json" >&2; then
        BASE_OK=0
        break
    fi

    echo "── round $i/$ROUNDS: head" >&2
    use_head
    npx tsx bench/run.ts ${FILTER[@]+"${FILTER[@]}"} --json "bench-head-$i.json" >&2
done

use_head

if [ "$BASE_OK" -eq 0 ]; then
    echo "### Benchmarks"
    echo
    echo "\`$REF\` could not be benchmarked with the current suite, so there is no comparison."
    exit 0
fi

npx tsx bench/compare.ts --base bench-base-*.json --head bench-head-*.json --warn 1.1 --fail 1.5
