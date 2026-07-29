# AGENTS.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Llamafile combines llama.cpp, whisper.cpp, stable-diffusion.cpp, and
transcribe.cpp with Cosmopolitan Libc to create single-file executables
(`llamafile`, `whisperfile`, `diffusionfile`, `transcribefile`) that run
locally across Windows, macOS, Linux, and BSD without installation.

For the class/API map and module relationships see
[`architecture.md`](architecture.md); for a file-by-file index see
[`repo_index.md`](repo_index.md).

## Twizzler / TWZM Model Format

[Twizzler](https://twizzler.io) is an OS research project built around
persistent, capability-addressed memory objects instead of files/processes:
any object (128-bit ObjID) can be mapped directly into a process's address
space via native syscalls. This repo's `TWZM` ("Twizzler Model") format and
the loader in `llamafile/twizzler.*` exist to prepare a GGUF model as one
such object, so that on Twizzler OS, "loading" a model is just mapping an
already-persistent object into memory - no parsing, no copying, as close to
instant as possible.

- `llamafile/gguf_to_twzm.cpp` - offline converter (`.gguf` -> `.twzm`) that
  precomputes everything expensive once (vocab hash tables, BPE merge-rank
  tables, detokenization piece cache) so the runtime loader never has to.
- `llamafile/twizzler.h`/`.cpp` - the loader: maps the object, parses the
  small embedded GGUF metadata blob, and builds a `llama_model` via
  `llama_model_init_from_user()` with a tensor-data callback pointing
  straight into the mapped region (no tensor-data copy).
- `llamafile/twizzler_platform.h` + `llamafile/twizzler_linux.cpp` - the
  platform boundary: real Twizzler OS gets native object-mapping syscalls;
  on Linux (used for all current development/testing) a shim mmaps a
  `.twzm` file found at `${TWZ_OBJECT_PATH}/<hi>_<lo>.twzm`.

Recent work on this repo drove TWZM vocab loading from ~300ms+ down to
~10ms by moving essentially all per-token computation out of the runtime
load path and into `gguf_to_twzm` (run once, offline, instead of on every
load). See `architecture.md` (TWZM fast-load format section) for the
on-disk struct layout and exactly how each optimization works.

## Quick Reference

```sh
# Initial setup (run once after clone, or after `make reset-repo`)
make setup

# Build (always use cosmocc make, not system make)
# Adapt `nproc` to the OS where you are building (e.g. `sysctl -n hw.physicalcpu` on mac)
.cosmocc/4.0.2/bin/make -j $(nproc)

# Run the full test suite
.cosmocc/4.0.2/bin/make check

# Run a single test (build+run one *.runs target - the pattern is
# o/$(MODE)/<path-to-test-binary>.runs)
.cosmocc/4.0.2/bin/make o/llamafile/highlight/highlight_test.runs
.cosmocc/4.0.2/bin/make o/tests/gpu_backend_test.runs

# Clean build outputs
.cosmocc/4.0.2/bin/make clean

# Reset all submodules (warning: removes local changes/patches)
make reset-repo
```

Manual smoke test of a built binary:

```sh
./o/llamafile/llamafile --cli -m model.gguf -p "hello"   # one-shot, scriptable
./o/llamafile/llamafile --model model.gguf                # TUI+server combined (default)
```

There is no repo-wide linter/formatter for `llamafile/` or the top-level
tree — code review is the only gate. (`transcribe.cpp/` is an exception: it
has its own pinned `clang-format` via `scripts/ci/clang-format.sh`, scoped
to that submodule only — see its local `AGENTS.md`.)

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `llamafile/` | Core library (edit directly) |
| `llama.cpp/` | LLM inference (submodule, edit directly then convert to patches) |
| `whisper.cpp/` | Speech-to-text (submodule, edit directly then convert to patches) |
| `stable-diffusion.cpp/` | Image generation (submodule, edit directly then convert to patches) |
| `transcribe.cpp/` | Multi-architecture speech-to-text (submodule, edit directly then convert to patches; has its own `AGENTS.md` and an 8-stage porting-skill pipeline) |
| `*.patches/` | Patch directories for submodules |
| `tests/`, `tools/`, `scripts/` | Repo-level tests, submodule-patch maintenance scripts, docs tooling |
| `o/` | Build outputs |

## Workflows

Two distinct workflows, per `CONTRIBUTING.md`:

- **Core code** (`llamafile/`, `whisperfile/`, `docs/`, `tests/`): edit,
  rebuild, test, commit normally.
- **Submodule code** (`llama.cpp/`, `whisper.cpp/`, `stable-diffusion.cpp/`,
  `transcribe.cpp/`): edit directly in the submodule, but changes must be
  converted to patches before committing:
  ```sh
  cd llama.cpp && ../tools/generate-patches.sh --output-dir ../llama.cpp.patches
  ```
  Then verify from a clean state: `make reset-repo && make setup &&
  .cosmocc/4.0.2/bin/make -j8 && .cosmocc/4.0.2/bin/make check`. Never
  hand-write or hand-edit `.patch` files - always regenerate via the
  script. Patch filenames mirror the file path with `/` replaced by `_`
  (e.g. `common_arg.cpp.patch` for `common/arg.cpp`).
- Each submodule may carry its **own** contribution policy in
  `<submodule>/AGENTS.md` - read it before editing inside that submodule.
  Notably: `llama.cpp/AGENTS.md` has a strict policy against
  fully-AI-generated PRs (it's an upstream-facing project);
  `transcribe.cpp/AGENTS.md` has submodule-local Python (`uv run`, never
  bare `python`/`pip`) and formatting conventions. `whisper.cpp/` and
  `stable-diffusion.cpp/` have no submodule-specific `AGENTS.md`.

Branch naming: `docs/`, `fix/`, `feature/`, `build/` prefixes. Open an issue
before starting large changes (new user-facing features, architectural
changes, new dependencies, build/packaging changes) - see `CONTRIBUTING.md`.

If you add a page under `docs/` that should appear in the hosted GitBook
site, also add it to `docs/SUMMARY.md` (CI catches dangling entries but not
missing ones). `docs/repo_index.md`, `docs/architecture.md`, and
`docs/skills/` are dev-reference docs, not part of that navigation.

## Important Notes

- Always use `.cosmocc/4.0.2/bin/make`, not system make.
- Run `make setup` after cloning, after `make reset-repo`, or after a
  submodule update.
- Submodule changes require patch files (see Workflows above).

## Detailed Documentation

For comprehensive build, architecture, development, and testing
documentation, ask Claude about "how to build llamafile" or "llamafile
development workflow" to load the llamafile skill (`docs/skills/llamafile/`).
