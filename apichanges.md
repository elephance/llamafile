# API / Internals Change Tracker — Twizzler (TWZM) Work

Living document. Goal: keep a running account of every public-API break and
every deep internal change introduced while adding Twizzler-object model
loading, so we can weigh it against the load-time gains and actively look for
ways to shrink the footprint (fewer signature changes, less duplicated
struct-layout knowledge, smaller diffs against upstream llama.cpp).

**Baseline** (pre-Twizzler): llamafile `3693f69` ("Update llama.cpp to b10052"),
llama.cpp submodule `b2dd28a3b`.
**Current**: llamafile `HEAD` (`4b99cc3`) + uncommitted working tree
(`--prompt-cache` / KV-cache persistence), llama.cpp submodule `a752eb81e`.
Twizzler-era llamafile commits: `9ba43d8` → `4b99cc3` (`Adding file format
tests`, `Numerous fixes` ×3, `Multi-object`, `Major perf improvement` ×2,
`Improving loading speed` ×2), mirrored in the submodule by `5fbbe657c` →
`a752eb81e`.

Update this file as new commits land. Each section below should be revisited
when the corresponding files change again.

---

## 1. Public API changes — llama.cpp (`include/llama.h`)

This is the surface every downstream consumer of llama.cpp (not just
llamafile) depends on. Right now there is exactly **one** break:

### `llama_model_init_from_user()` gained three parameters

```c
// before
struct llama_model * llama_model_init_from_user(
    struct gguf_context * metadata,
    llama_model_set_tensor_data_t set_tensor_data,
    void * set_tensor_data_ud,
    struct llama_model_params params);

// after
struct llama_model * llama_model_init_from_user(
    struct gguf_context * metadata,
    llama_model_set_tensor_data_t set_tensor_data,
    void * set_tensor_data_ud,
    struct llama_model_params params,
    const void * twzm_vocab_section,   // optional TWZM flat vocab section
    const void * twzm_tensor_index,    // optional TWZM tensor index (sorted array)
    uint64_t     twzm_tensor_count);   // entries in twzm_tensor_index
```

- Source-breaking (not just ABI) for every existing caller of this API —
  anyone building a model from in-memory GGUF metadata + a tensor-data
  callback (llamafile is presently the only known caller upstream).
- The three new parameters are a leak of TWZM's on-disk format into the
  public C API: callers must know what a "TWZM flat vocab section" and
  "TWZM tensor index" are, even though only llamafile's `twizzler.cpp`
  produces them today.
- **Minimization angle**: this could become a single opaque
  `const void * twzm_loader_hints` (or a small forward-declared struct
  pointer) instead of three separate parameters, so future format
  additions (a 4th optional section, say) don't add a 4th parameter. Or,
  more aggressively, expose a lower-level "vocab loaded out of band" /
  "tensor table loaded out of band" pair of setter calls made on the
  `gguf_context`/loader *before* `llama_model_init_from_user()`, keeping
  the signature stable and pushing the TWZM-specific bits behind an
  already-internal boundary (`llama_model_loader`) rather than the public
  entry point.

No other function in `include/llama.h` changed.

---

## 2. Deep internal changes — llama.cpp (`src/`)

These don't cross `include/llama.h`, but they're substantial enough that any
future upstream merge/rebase will conflict with them, and they encode TWZM
knowledge (or format assumptions) deep in files that have nothing to do with
Twizzler on their face.

### 2.1 `llama_vocab::token_data` layout change (`src/llama-vocab.h`)

- `token_data.text` (owning `std::string`) removed; replaced with
  `uint32_t text_offset` into a new shared text pool
  (`llama_vocab::impl::text_pool`).
- New accessors: `token_get_text_view()` (returns `std::string_view`,
  no `strlen`), and `token_get_text()` still exists but now derives the
  pointer from the pool.
- Struct is now 12 bytes, "byte-identical to TWZM's on-disk `TwzmTokenData`"
  by design, so the TWZM path populates the whole array with one `memcpy`
  instead of constructing 128K `std::string`s.
- **Coupling risk**: `src/llama-vocab.h` (core llama.cpp internals) now
  has a comment explicitly citing TWZM's on-disk layout as a design
  constraint. If TWZM's `TwzmTokenData` layout ever changes, this comment
  (and the memcpy-compatibility invariant) silently rots — there's no
  compile-time assertion tying the two structs together (worth adding a
  `static_assert(sizeof(...) == ...)` / offset check, see §6).
- Any third-party code reading `.text` directly off `token_data` (outside
  llama.cpp, e.g. bindings or forks) breaks at compile time — this is the
  most invasive of the "internal" changes because it touches a
  vocab-wide data structure, not a single call path.

### 2.2 `llama_model_loader` gained a whole second tensor-resolution path (`src/llama-model-loader.h/.cpp`)

- New members: `twzm_vocab_section`, `twzm_tensor_index`,
  `twzm_tensor_count`, plus a locally-defined `twzm_tensor_entry` struct
  that **duplicates** `llamafile/twizzler.h`'s `TwzmTensorEntry` layout
  (deliberately kept as a local mirror rather than an `#include`, per the
  comment at `src/llama-model-loader.h:120`, "so llama.cpp stays
  independent of llamafile headers").
- `create_tensor()` now branches on whether a TWZM tensor index is present:
  if so, tensor type/shape come from the index (binary search by name)
  instead of `gguf_find_tensor()`/`gguf_get_tensor_type()`, and shapes are
  validated against `te->ne` — a validation that has no equivalent on the
  GGUF path structure-wise (GGUF's shape check happens elsewhere via
  `check_tensor_dims`).
- **This is the same "mirror the struct, add a comment, hope it stays in
  sync" pattern as §2.1.** Two independent copies of the on-disk
  `TwzmTensorEntry`/`TwzmTokenData` layouts now exist — one in
  `llamafile/twizzler.h` (source of truth, versioned via `TWZM_VERSION`)
  and one each in `llama-vocab.h`/`llama-model-loader.h`. A version bump
  in `llamafile/twizzler.h` must be manually propagated to both.
- `n_tensors` accounting, `create_tensor()`'s speculative
  over-allocation guess (previously always `+= n_layer*256` when there's
  no backing file), and duplicated-tensor handling (`TENSOR_DUPLICATED`)
  were all touched to support the no-file, index-only path.

### 2.3 `llama-mmap.cpp`/`.h`: file I/O now has a `COSMOCC` branch that defers to llamafile

- `llama_file::impl` now opens/reads/seeks/tells/closes through
  `llamafile_open_gguf()`/`llamafile_read()`/`llamafile_seek()`/
  `llamafile_tell()`/`llamafile_close()` when built under `COSMOCC`,
  instead of raw `fopen`/`fd` I/O.
- New public-ish methods on `llama_file`: `has_premapped_content()`,
  `premapped_content()`, `get_llamafile()` — used for llamafile's
  bundled-zip-asset path, not Twizzler directly, but part of the same
  "loading speed" initiative and the same risk category: `llama-mmap.h`
  (a core, previously platform-only file) now has an `#ifdef COSMOCC`
  dependency on a whole other project's (`llamafile/llamafile.h`) API
  surface.
- This is the deepest architectural change of the bunch: llama.cpp's file
  abstraction is no longer self-contained under cosmocc builds.

### 2.4 `llama_model_base::load_tensors()`: dummy-buffer optimization for caller-supplied tensor data (`src/llama-model.cpp`)

- When `set_tensor_data` is non-null (the TWZM path) *and* the target
  buffer type is host memory, `load_tensors()` now allocates a **dummy
  (zero-size) backend buffer** instead of a real one, since every
  `tensor->data` will be overwritten by the caller's callback anyway.
  Comment cites measured **17.3 GB of 41.9 GB VmPeak saved on a 30B
  model** — a real, load-bearing internal change, not cosmetic.
- Distinct code path from the pre-existing `ml.no_alloc` dummy-buffer
  case; the two are now `||`'d together at the allocation-branch level.

### 2.5 `llama-kv-cache.cpp` / `llama-kv-cache-dsv4.cpp`: skip zeroing host KV buffers

- `ggml_backend_buffer_clear(buf, 0)` calls now guarded by
  `!ggml_backend_buffer_is_host(buf)` — host buffers are always written
  before read, so the memset was pure waste. GPU buffers still cleared
  (padding can otherwise carry NaNs into kernels). Part of the general
  loading-speed push, not TWZM-specific, but bundled into the same
  commits.

### 2.6 `llama-impl.h`: shared stage-timing helper

- New `llama_stage_mark()` / `llama_stage_timing_enabled()` (gated on
  `TWZM_DEBUG` env var), called from `src/llama.cpp` (`llama_model_load`),
  `src/llama-model.cpp` (`load_tensors`), and mirrored in
  `llamafile/twizzler.cpp` and `llamafile/chatbot_cli.cpp`
  (`twzm_kv_debug_level()`). Two independent env-var-gated debug/timing
  systems (`TWZM_DEBUG` read in both `llama-impl.h` and
  `twizzler_kvcache.cpp`) with the same name and similar but not identical
  level semantics (`>0` traces, `>1` adds round-trip verification in one
  place vs. just "enabled" in the other) — worth unifying (see §6).

### 2.7 Unrelated-but-bundled fixes (same commits, not Twizzler-motivated)

Noting these so they aren't mistaken for TWZM footprint when auditing diff
size later:
- `src/models/dflash.cpp`, `eagle3.cpp`, `t5.cpp`: reordered explicit
  template specializations / added forward declarations — cosmocc/clang
  `-std=gnu++23` compatibility, unrelated to loading speed.
- `common/log.cpp`: blocks signals on the async logging thread and
  switches an untimed `condition_variable::wait()` to a 30s-polling
  `wait_for()` loop — works around a Cosmopolitan libc bug on macOS
  (untimed futex waits expiring after ~72 min).
- `src/llama-context.cpp`: `auto_fa` (auto Flash Attention) now forced off
  when there are no GPU devices — CPU FA path is slower than non-FA on x86
  (references `mozilla-ai/llamafile#975`).
- `common/common.cpp`, `common/arg.cpp`, `common/download.cpp`: `COSMOCC`
  branches for CPU-core detection, cache-directory resolution, and
  `PATH_MAX` — general cosmocc portability, not TWZM.
- `common/http.h`: `CPPHTTPLIB_OPENSSL_SUPPORT` → `CPPHTTPLIB_SSL_ENABLED`
  macro rename (upstream cpp-httplib rename), unrelated.
- `src/unicode.h/.cpp`: `unicode_cpt_from_utf8`/`unicode_cpts_from_utf8`
  now take `std::string_view` instead of `const std::string&` — enables
  the vocab text-pool work (§2.1) to pass views without allocating, so
  this one **is** TWZM-motivated despite touching a generic-looking file.

---

## 3. Public API changes — llamafile (`llamafile/`)

### 3.1 New entry point: `llamafile_model_load()` (`llamafile/llama.h`, `llamafile/llama.cpp`)

```c
struct llama_model * llamafile_model_load(const char * path,
                                          struct llama_model_params params);
```

Dispatches on file extension: `.twzm` → `llama_model_load_from_twzm_path()`,
else → `llama_model_load_from_file()`. Documented as "the preferred
model-loading entry point within llamafile." Additive, non-breaking.

- `llamafile/llama.h` also switched from forward-declaring
  `struct llama_model;` / `struct llama_context;` to
  `#include "../llama.cpp/include/llama.h"` directly, since it now needs
  the full `llama_model_params` type. Low risk, but worth knowing this
  small "helpers" header now pulls in the entire llama.cpp public API
  transitively for every one of its includers.

### 3.2 New CLI flag: `--prompt-cache` (uncommitted; `llamafile/args.cpp`, `llamafile/llamafile.h/.c`)

- New global `FLAG_prompt_cache` (`llamafile.h`/`llamafile.c`), parsed
  specially in `args.cpp` rather than via `common_params_parse()` because
  upstream's own `--prompt-cache` flag is registered only for
  `LLAMA_EXAMPLE_COMPLETION`, not `LLAMA_EXAMPLE_CLI`.
- **Name collision with upstream**: llama.cpp's `common/arg.cpp` already
  defines a `--prompt-cache <path>` option (file-based KV cache, takes a
  value) for the completion example. Llamafile's new flag is a bare
  boolean and is TWZM-specific (only works with `.twzm` models, derives
  its own cache location from the model's object id). Same flag name,
  different arity, different semantics, gated to different example
  types — this is a latent footgun if the CLI example type ever changes,
  or if someone reads `--help` output expecting upstream's behavior.
  **Worth renaming** (e.g. `--twzm-prompt-cache`) before this lands, to
  avoid the collision outright. See §6.

### 3.3 New binary: `gguf-to-twzm` (`llamafile/BUILD.mk`, `llamafile/gguf_to_twzm.cpp`)

- Offline converter, `Usage: gguf-to-twzm <input.gguf> <output.twzm>
  [--verify]`. Built as part of the default `llamafile` target
  (`o/$(MODE)/llamafile: o/$(MODE)/llamafile/llamafile
  o/$(MODE)/llamafile/gguf-to-twzm`), i.e. it now ships with every
  llamafile build. New user-facing surface (a whole new tool), even
  though it's off the hot path.

---

## 4. New modules (net-new files, no upstream equivalent)

| File | Role |
|---|---|
| `llamafile/twizzler.h` | On-disk TWZM format (root/vocab/KV-cache structs) + public loader API (`llama_model_load_from_twzm{_path,}`, `llama_model_load_from_twizzler_object`) |
| `llamafile/twizzler.cpp` | Loader implementation: maps root/vocab/tensor-data objects, builds `llama_model` via `llama_model_init_from_user()` |
| `llamafile/twizzler_platform.h` | Platform-abstraction API (`twz_object_map`, `twz_object_create*`, `twz_object_handle_*`) — the seam real Twizzler OS vs. the Linux dev shim implement |
| `llamafile/twizzler_linux.cpp` | Linux dev-shim implementation of the above (file-backed under `$TWZ_OBJECT_PATH`) |
| `llamafile/twizzler_kvcache.cpp` *(uncommitted)* | Persistent KV-cache-in-a-TWZM-object implementation backing `--prompt-cache` |
| `llamafile/gguf_to_twzm.cpp` | Offline `.gguf` → `.twzm` converter (864 lines) |

This is ~2,600+ lines of genuinely new surface area (loader + converter +
platform shim + kv-cache), on top of the llama.cpp changes in §1–2. All of
it is additive (no existing llamafile file was repurposed), so it doesn't
"break" anything by itself — but it's the bulk of what would need
maintaining/porting if TWZM's format changes again, and it's what a real
Twizzler-OS backend implementation (replacing `twizzler_linux.cpp`) will
need to match.

### Format versioning already in place

`TWZM_VERSION` is at 3 (v3 added `type`/`n_dims`/`ne` to `TwzmTensorEntry`,
explicitly documented as a breaking bump — "v2 objects are not readable by
a v3 loader"). `TWZM_KV_VERSION` is separately at 1. Two independent
version counters for two objects that are only ever produced/consumed
together by the same llamafile build — consider whether these need to
stay independent or whether a single "TWZM toolchain version" simplifies
compatibility reasoning (see §6).

---

## 5. Net gains being weighed against this footprint

(For context/justification of the above — captured from code comments,
not independently re-benchmarked here.)

- Vocab load: ~300ms+ → ~10ms for a 128K-token vocab (flat vocab section,
  §2.1, skips `gguf_init_from_buffer` + `load_vocab` entirely).
- Tensor-info parsing: `gguf_init_from_buffer()` tensor-info construction
  was ~95% of its cost, scaling with tensor count (0.52ms of 0.55ms on a
  579-tensor MoE) — eliminated by the TWZM tensor index (§2.2).
- Peak memory: 17.3 GB saved out of 41.9 GB VmPeak on a 30B model via the
  dummy-buffer path for caller-supplied tensor data (§2.4).
- Prompt evaluation (new, uncommitted): ~15ms/token with no fixed cost
  eliminated on a cache hit (measured ~40 tokens = 178ms, ~370 tokens =
  5542ms on a 1B model, replaced by a state-blob memcpy of ~33KB/token)
  via `--prompt-cache` (§3.2, §4).

---

## 6. Open items to reduce API / internal footprint

Running list — revisit as work continues:

1. **Collapse `llama_model_init_from_user()`'s 3 new params** into one
   opaque pointer (or move them below the public API into a
   loader-populated-before-call pattern) so future TWZM format additions
   don't grow the public signature again. (§1)
2. **Tie the duplicated on-disk structs together with a compile-time
   check.** `TwzmTokenData` (llamafile/twizzler.h) vs. `llama_vocab::token_data`
   (llama-vocab.h), and `TwzmTensorEntry` (llamafile/twizzler.h) vs.
   `llama_model_loader::twzm_tensor_entry` (llama-model-loader.h) are
   independently maintained mirrors with only comments enforcing
   agreement. Add `static_assert`s (size + offsetof for each field) at
   the point where the raw pointer is cast, so a future edit to one side
   fails to compile instead of silently corrupting reads.
3. **Rename `--prompt-cache`** to something TWZM-specific
   (`--twzm-prompt-cache`?) to avoid colliding with upstream's own
   `--prompt-cache <path>` (different arity/semantics, different example
   type today, but a latent conflict). (§3.2)
4. **Unify the two `TWZM_DEBUG`-gated timing/trace systems**
   (`llama-impl.h`'s `llama_stage_mark` vs. `twizzler_kvcache.cpp`'s
   `twzm_kv_debug()`/`KV_LOG`) into one, or at least document that they
   share an env var but have different level semantics. (§2.6)
5. **Reconsider `COSMOCC`-gated `llama-mmap.cpp` coupling to
   `llamafile.h`** (§2.3) — this is the deepest cross-project dependency
   introduced; worth a design pass on whether it can be pushed behind a
   narrower interface (e.g. a `llama_file_backend` vtable llamafile
   installs) rather than `#ifdef COSMOCC` branches inside llama.cpp core.
6. **Decide whether `TWZM_VERSION` and `TWZM_KV_VERSION` should be
   merged** into one version number now that there are two independently
   bumpable formats produced by the same converter. (§4)
7. Keep watching `llama-model-loader.cpp`'s `create_tensor()` — it now has
   three branches (GGUF-with-file, GGUF-no-file guess, TWZM-index-exact);
   confirm the GGUF-no-file guess path (`n_layer*256` over-allocation)
   isn't a candidate for the same exactness fix TWZM got, which would let
   some of this logic be shared instead of forked. (§2.2)
