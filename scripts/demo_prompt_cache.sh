#!/bin/bash

# Demonstrate the TWZM persistent KV cache (--prompt-cache).
#
# Runs the same long prompt twice. The first run pays full prompt evaluation
# and writes its KV state to a Twizzler object; the second restores that state
# and only re-decodes the final token. The saving is proportional to prompt
# length, which is why this uses a deliberately long prompt - at ~15 tokens the
# difference is 232ms vs 55ms, at ~410 tokens it is 5539ms vs 54ms.
#
# Note that the generated text may differ slightly between the two runs. That
# is expected and is not caused by the cache: llama.cpp's matmul kernels are
# not batch-invariant, so decoding a prompt as one batch versus as a restore
# plus one token can flip a near-tie under greedy sampling. Decoding the same
# prompt at -b 256 versus -b 1 already differs with no cache involved. The
# invariant that matters - that the restored state is byte-identical to the
# state that was saved - is checked by --debug below.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="data/llama-3.2-1b-instruct-q8_0.twzm"
REPEATS=40
DEBUG=0
KEEP=0
COMPARE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)   MODEL="$2"; shift 2 ;;
        --repeats)    REPEATS="$2"; shift 2 ;;
        --debug)      DEBUG=1; shift ;;
        --keep-cache) KEEP=1; shift ;;
        --compare)    COMPARE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Show the effect of --prompt-cache by running one prompt twice."
            echo ""
            echo "Options:"
            echo "  -m, --model <path>  .twzm model (default: $MODEL)"
            echo "      --repeats <n>   Prompt length, in sentences (default: $REPEATS)"
            echo "      --debug         Show cache hit/miss detail and the"
            echo "                      restore round-trip integrity check"
            echo "      --keep-cache    Do not clear existing caches first, so"
            echo "                      run 1 is warm too"
            echo "      --compare       Also time the unmodified .gguf path and"
            echo "                      the uncached .twzm path, for reference"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# The loader resolves child objects (weights, vocab, KV cache) by id under this
# directory; it must match however gguf-to-twzm was run.
export TWZ_OBJECT_PATH="${TWZ_OBJECT_PATH:-./twzm_objects}"

LLAMAFILE="o//llamafile/llamafile"
[[ -x "$LLAMAFILE" ]] || LLAMAFILE="o/llamafile/llamafile"
if [[ ! -x "$LLAMAFILE" ]]; then
    echo "error: llamafile not built. Run: .cosmocc/4.0.2/bin/make -j\$(nproc)" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "error: model '$MODEL' not found." >&2
    echo "Convert one first: $LLAMAFILE-equivalent gguf-to-twzm <in.gguf> <out.twzm>" >&2
    exit 1
fi

# Clear existing KV cache objects so a run is genuinely cold. They are found by
# magic ("WKCM" = 574b434d little-endian) rather than by name, because a cache
# object's id is derived from the model rather than being stored anywhere.
clear_caches() {
    [[ "$KEEP" -eq 0 ]] || return 0
    shopt -s nullglob
    local f
    for f in "$TWZ_OBJECT_PATH"/*.twzm; do
        if [[ "$(head -c4 "$f" | xxd -p)" == "574b434d" ]]; then
            rm -f "$f"
        fi
    done
    shopt -u nullglob
}
clear_caches

PROMPT="$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.'] * $REPEATS))")"

FILTER='restore kv|evaluate prompt|save kv'
if [[ "$DEBUG" -eq 1 ]]; then
    export TWZM_DEBUG=2
    FILTER="$FILTER|\[kv\]"
fi

echo "model:  $MODEL"
echo "prompt: $REPEATS sentences"
echo

# Extract "<stage> <ms>" plus the running total from one run's timing output.
# Prints: "<load_ms> <eval_ms> <ready_ms>", where ready_ms is the cumulative
# time at the end of prompt evaluation - i.e. how long until the model can
# emit its first token.
time_one_run() {
    local out
    out="$("$@" -p "$PROMPT" -n 4 2>&1 >/dev/null || true)"
    local load eval_ms ready
    load=$(printf '%s\n' "$out"    | sed -n 's/^\[timing\] load model  *\([0-9.]*\) ms.*/\1/p')
    eval_ms=$(printf '%s\n' "$out" | sed -n 's/^\[timing\] evaluate prompt (text)  *\([0-9.]*\) ms.*/\1/p')
    ready=$(printf '%s\n' "$out"   | sed -n 's/^\[timing\] evaluate prompt (text).*total: \([0-9.]*\) ms.*/\1/p')
    printf '%s %s %s' "${load:-n/a}" "${eval_ms:-n/a}" "${ready:-n/a}"
}

if [[ "$COMPARE" -eq 1 ]]; then
    GGUF="${MODEL%.twzm}.gguf"
    printf '%-34s %10s %10s %10s\n' "" "load" "prompt" "ready"
    if [[ -f "$GGUF" ]]; then
        # shellcheck disable=SC2046
        printf '%-34s %10s %10s %10s\n' ".gguf, no cache (stock path)" \
            $(time_one_run "$LLAMAFILE" --cli -m "$GGUF")
    else
        printf '%-34s %s\n' ".gguf, no cache (stock path)" "skipped: $GGUF not found"
    fi
    # shellcheck disable=SC2046
    printf '%-34s %10s %10s %10s\n' ".twzm, no cache" \
        $(time_one_run "$LLAMAFILE" --cli -m "$MODEL")
    "$LLAMAFILE" --cli --prompt-cache -m "$MODEL" -p "$PROMPT" -n 4 >/dev/null 2>&1
    # shellcheck disable=SC2046
    printf '%-34s %10s %10s %10s\n' ".twzm + --prompt-cache (warm)" \
        $(time_one_run "$LLAMAFILE" --cli --prompt-cache -m "$MODEL")
    echo
    echo "(ms; 'ready' = cumulative time until the first token can be emitted)"
    echo
fi

# --compare leaves a warm cache behind, which would make "run 1" below a lie.
clear_caches

for i in 1 2; do
    if [[ $i -eq 1 ]]; then
        echo "--- run 1 (cold: computes the prompt, writes the cache)"
    else
        echo "--- run 2 (warm: restores the cache)"
    fi
    # 2>&1 >/dev/null keeps stderr (where the timings go) and drops the
    # model's generated text. Order matters: >/dev/null after the redirect.
    "$LLAMAFILE" --cli --prompt-cache -m "$MODEL" -p "$PROMPT" -n 4 \
        2>&1 >/dev/null | grep -E "$FILTER" || true
done

echo
echo "Run 2 has no 'save kv cache' line: the cached state already matches, and"
echo "rewriting it would cost a full state serialization for no benefit."
echo "For the multi-turn case, append to the prompt instead of repeating it -"
echo "the shared prefix is still reused. Try --compare for the .gguf baseline."
