// Copyright 2026 Mozilla.ai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Twizzler Memory Object loader for llama.cpp models.
//
// Implements llama_model_load_from_twzm() and
// llama_model_load_from_twizzler_object() by:
//
//   1. Validating the TWZM header.
//   2. Parsing the embedded GGUF metadata blob with gguf_init_from_buffer()
//      (no tensor data allocation – just KV pairs + tensor type/shape info).
//   3. Building a name→(base,size) index from the tensor index section.
//   4. Calling llama_model_init_from_user() with a set_tensor_data callback
//      that sets tensor->data to the pre-mapped address (zero-copy).
//   5. Freeing the gguf_context (the model owns its own copy of all data).
//
// The caller that passes a pre-mapped region (llama_model_load_from_twzm)
// remains responsible for keeping the mapping alive for the lifetime of the
// returned model.  llama_model_load_from_twizzler_object() wraps the mapping
// in a deleter stored in a helper object so the mapping is freed when the
// model is freed.

#include "twizzler.h"
#include "twizzler_platform.h"

// twizzler.h already includes the real llama.h (via relative path).
// Include gguf.h and ggml.h the same way to avoid include-search shadowing.
#include "../llama.cpp/ggml/include/gguf.h"
#include "../llama.cpp/ggml/include/ggml.h"
#include "../llama.cpp/ggml/include/ggml-backend.h" // for ggml_backend_tensor_set

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

// Set TWZM_DEBUG=1 in the environment to enable verbose tracing.
static int twzm_debug_level() {
    static int level = -1;
    if (level < 0) {
        const char * v = getenv("TWZM_DEBUG");
        level = v ? atoi(v) : 0;
    }
    return level;
}
#define TWZM_LOG(fmt, ...) \
    do { if (twzm_debug_level() > 0) fprintf(stderr, "twzm: " fmt "\n", ##__VA_ARGS__); } while (0)

#define TWZM_LOG_V(fmt, ...) \
    do { if (twzm_debug_level() > 1) fprintf(stderr, "twzm: " fmt "\n", ##__VA_ARGS__); } while (0)
// ---------------------------------------------------------------------------
// Internal loader state — zero dynamic allocation
// ---------------------------------------------------------------------------

// The tensor index in the TWZM file is sorted by name at conversion time.
// We keep a direct pointer into the mmap'd object and use bsearch().
struct TwzmLoader {
    const uint8_t         * base    = nullptr;
    size_t                  size    = 0;
    const TwzmTensorEntry * entries = nullptr; // pointer directly into mmap'd object
    uint64_t                count   = 0;
    const gguf_context    * gguf_ctx = nullptr; // borrowed; used to classify missing tensors

    bool open(const void * base_ptr, size_t sz, const gguf_context * gguf);
};

static int twzm_entry_cmp(const void * key, const void * elem) {
    return strcmp(static_cast<const char *>(key),
                  static_cast<const TwzmTensorEntry *>(elem)->name);
}

static const TwzmTensorEntry * twzm_lookup(const TwzmLoader * loader, const char * name) {
    return static_cast<const TwzmTensorEntry *>(
        bsearch(name, loader->entries, loader->count,
                sizeof(TwzmTensorEntry), twzm_entry_cmp));
}

bool TwzmLoader::open(const void * base_ptr, size_t sz, const gguf_context * gguf) {
    base     = static_cast<const uint8_t *>(base_ptr);
    size     = sz;
    gguf_ctx = gguf;

    // -- Validate header --------------------------------------------------
    if (sz < sizeof(TwzmHeader)) {
        fprintf(stderr, "twzm: object too small to contain header (%zu bytes)\n", sz);
        return false;
    }

    TwzmHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));

    if (hdr.magic != TWZM_MAGIC) {
        fprintf(stderr, "twzm: bad magic 0x%08X (expected 0x%08X)\n",
                hdr.magic, TWZM_MAGIC);
        return false;
    }
    if (hdr.version != TWZM_VERSION) {
        fprintf(stderr, "twzm: unsupported version %" PRIu32 " (expected %" PRIu32 ")\n",
                hdr.version, TWZM_VERSION);
        return false;
    }

    // -- Validate metadata blob bounds ------------------------------------
    if (hdr.metadata_offset + hdr.metadata_size > sz) {
        fprintf(stderr, "twzm: metadata blob out of bounds\n");
        return false;
    }

    // -- Bind and validate tensor index -----------------------------------
    uint64_t index_bytes = hdr.tensor_count * sizeof(TwzmTensorEntry);
    if (hdr.tensor_index_offset + index_bytes > sz) {
        fprintf(stderr, "twzm: tensor index out of bounds\n");
        return false;
    }

    entries = reinterpret_cast<const TwzmTensorEntry *>(base + hdr.tensor_index_offset);
    count   = hdr.tensor_count;

    // Validate each entry and confirm the index is sorted (required for bsearch).
    for (uint64_t i = 0; i < count; ++i) {
        const TwzmTensorEntry & e = entries[i];
        if (strnlen(e.name, TWZM_TENSOR_NAME_MAX) == TWZM_TENSOR_NAME_MAX) {
            fprintf(stderr, "twzm: entry %" PRIu64 " name is not null-terminated\n", i);
            return false;
        }
        if (e.data_offset + e.data_size > sz) {
            fprintf(stderr, "twzm: tensor '%s' data out of bounds\n", e.name);
            return false;
        }
        if (i > 0 && strcmp(entries[i-1].name, e.name) >= 0) {
            fprintf(stderr, "twzm: index not sorted at entry %" PRIu64 "\n", i);
            return false;
        }
    }

    TWZM_LOG("loaded %" PRIu64 " tensor(s) from sorted index", count);
    if (twzm_debug_level() > 1) {
        for (uint64_t i = 0; i < count; ++i) {
            fprintf(stderr, "twzm:   [index] %-48s  off=%" PRIu64 "  len=%" PRIu64 "\n",
                    entries[i].name, entries[i].data_offset, entries[i].data_size);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Tensor data callback
// ---------------------------------------------------------------------------

// Called by llama_model_init_from_user() for every tensor that needs its
// data pointer set.  We resolve the tensor name to a pre-mapped region and
// either set tensor->data directly (zero-copy, CPU backend) or memcpy into
// an already-allocated buffer (GPU or non-CPU backend).
static void twzm_set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    TwzmLoader * loader = static_cast<TwzmLoader *>(userdata);
    const char * name = ggml_get_name(tensor);

    const TwzmTensorEntry * entry = twzm_lookup(loader, name);
    if (!entry) {
        if (loader->gguf_ctx &&
            gguf_find_tensor(loader->gguf_ctx, name) >= 0) {
            fprintf(stderr, "twzm: tensor '%s' is in GGUF metadata but missing "
                            "from TWZM index (conversion bug?)\n", name);
        } else {
            TWZM_LOG_V("set_tensor_data: %-48s  SKIPPED (not in GGUF)", name);
        }
        return;
    }

    size_t expected = ggml_nbytes(tensor);
    const uint8_t * src = loader->base + entry->data_offset;
    TWZM_LOG_V("set_tensor_data: %-48s  type=%-8s  expected=%zu  have=%" PRIu64,
             name, ggml_type_name(tensor->type), expected, entry->data_size);
    if (entry->data_size < expected) {
        fprintf(stderr, "twzm: tensor '%s': TWZM %" PRIu64 " B < expected %zu B\n",
                name, entry->data_size, expected);
        return;
    }

    // Zero-copy for CPU / no_alloc path: set tensor->data to point directly
    // into the TWZM mapping.  This covers two cases:
    //   (a) no_alloc=true: tensor->data is NULL (dummy zero-size buffer was used);
    //       direct assignment is the only option.
    //   (b) normal CPU allocation: tensor->data points into an anonymous mmap
    //       that we no longer need; redirect it to the TWZM mapping instead.
    // For device (GPU) tensors tensor->data is non-NULL device memory —
    // upload via the backend copy API.
    if (tensor->data == nullptr || (tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer))) {
        tensor->data = const_cast<uint8_t *>(src);
    } else {
        ggml_backend_tensor_set(tensor, src, 0, expected);
    }

    // TWZM_DEBUG >= 2: dump raw bytes from the TWZM mapping so we can
    // verify the file content without relying on backend readback.
    if (twzm_debug_level() >= 2) {
        constexpr size_t PROBE = 16;
        size_t probe = expected < PROBE ? expected : PROBE;
        fprintf(stderr, "twzm:   twzm_bytes[0..%zu]: ", probe);
        for (size_t b = 0; b < probe; ++b) fprintf(stderr, "%02x", src[b]);
        // Also print from tensor->data directly (only safe for CPU buffers).
        // On GPU builds this may crash — set TWZM_DEBUG=1 if it does.
        const uint8_t * dst = static_cast<const uint8_t *>(tensor->data);
        if (dst) {
            fprintf(stderr, "  tensor_data[0..%zu]: ", probe);
            for (size_t b = 0; b < probe; ++b) fprintf(stderr, "%02x", dst[b]);
        } else {
            fprintf(stderr, "  tensor->data=NULL");
        }
        fprintf(stderr, "\n");
    }
}

// ---------------------------------------------------------------------------
// llama_model_load_from_twzm
// ---------------------------------------------------------------------------

struct llama_model * llama_model_load_from_twzm(
        const void * base,
        size_t size,
        struct llama_model_params params) {

    auto twzm_ms = [](int64_t t0) -> double {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        int64_t t1 = (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
        return (double)(t1 - t0);
    };
    auto twzm_now = []() -> int64_t {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    };
    const bool do_time = twzm_debug_level() > 0;
#define TWZM_STAGE(label)  do { if (do_time) { \
    fprintf(stderr, "twzm: [time] %-40s %6.1f ms\n", (label), twzm_ms(t0_stage)); \
    t0_stage = twzm_now(); } } while(0)

    int64_t t0_stage = twzm_now();
    int64_t t0_total = t0_stage;

    // Fast path: the model was already built during a previous call in this
    // process and its pointer was cached in the header via a COW write.
    // On Twizzler this would be a persistent pointer surviving across processes.
    TwzmHeader * hdr_rw = const_cast<TwzmHeader *>(
        reinterpret_cast<const TwzmHeader *>(base));
    if (hdr_rw->cached_model_ptr) {
        TWZM_LOG("returning cached model from header pointer");
        return reinterpret_cast<struct llama_model *>(
            (uintptr_t)hdr_rw->cached_model_ptr);
    }

    // 1. Open and validate the TWZM object.
    //    Pass gguf_ctx=nullptr for now; we'll set it after parsing.
    TwzmLoader loader;
    if (!loader.open(base, size, nullptr)) {
        return nullptr;
    }
    TWZM_STAGE("open + validate index");

    // 2. Parse the embedded GGUF metadata blob.
    //    The blob contains all KV pairs and tensor info (name/type/shape) but
    //    no tensor data.  no_alloc=true means gguf will not try to allocate
    //    any ggml_context for tensor storage.
    const TwzmHeader * hdr = reinterpret_cast<const TwzmHeader *>(base);
    const void * meta_blob = static_cast<const uint8_t *>(base) + hdr->metadata_offset;

    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    struct gguf_context * gguf_ctx =
        gguf_init_from_buffer(meta_blob, static_cast<size_t>(hdr->metadata_size), gguf_params);

    if (!gguf_ctx) {
        fprintf(stderr, "twzm: failed to parse embedded GGUF metadata blob\n");
        return nullptr;
    }
    TWZM_STAGE("gguf_init_from_buffer");

    // Give the loader the context so it can classify missing tensors correctly.
    loader.gguf_ctx = gguf_ctx;

    // 3. Handle weight tying: if output.weight is absent from the GGUF metadata
    //    but IS present in the TWZM index (added by gguf_to_twzm), add a
    //    synthetic gguf_context entry so create_tensor picks up the correct
    //    quantised type rather than defaulting to F32.
    if (gguf_find_tensor(gguf_ctx, "output.weight") < 0) {
        const TwzmTensorEntry * out_entry = twzm_lookup(&loader, "output.weight");
        int64_t tok_id = gguf_find_tensor(gguf_ctx, "token_embd.weight");
        if (out_entry && tok_id >= 0) {
            TWZM_LOG("weight tying: output.weight aliased to token_embd.weight in gguf_ctx");

            enum ggml_type type   = gguf_get_tensor_type(gguf_ctx, tok_id);
            const int64_t * ne    = gguf_get_tensor_ne(gguf_ctx, tok_id);
            int64_t blk           = (int64_t)ggml_blck_size(type);

            struct ggml_tensor t  = {};
            t.type    = type;
            t.ne[0]   = ne[0];
            t.ne[1]   = ne[1];
            t.ne[2]   = 1;
            t.ne[3]   = 1;
            t.nb[0]   = ggml_type_size(type);
            t.nb[1]   = (blk > 0) ? (t.nb[0] * t.ne[0] / blk) : (t.nb[0] * t.ne[0]);
            t.nb[2]   = t.nb[1] * t.ne[1];
            t.nb[3]   = t.nb[2] * t.ne[2];
            strncpy(t.name, "output.weight", GGML_MAX_NAME - 1);

            gguf_add_tensor(gguf_ctx, &t);
        }
    }
    TWZM_STAGE("weight tying");

    // 4. Build the llama_model via the existing user-init path.
    params.use_mmap        = false;
    params.use_extra_bufts = false;

    struct llama_model * model =
        llama_model_init_from_user(gguf_ctx, twzm_set_tensor_data, &loader, params);
    TWZM_STAGE("llama_model_init_from_user");

    // 4b. Replace token_to_id with the pre-built hash table if available.
    //     load_vocab already built it the slow way (unordered_map inserts);
    //     we swap in the hash-table-populated version for ~zero-cost lookups.
    if (hdr->vocab_offset != 0 && hdr->vocab_size >= sizeof(TwzmVocabHeader)) {
        const auto * vhdr = reinterpret_cast<const TwzmVocabHeader *>(
            static_cast<const uint8_t *>(base) + hdr->vocab_offset);
        if (vhdr->magic == TWZM_VOCAB_MAGIC &&
            vhdr->token_hash_offset > 0 &&
            vhdr->token_hash_capacity > 0) {
            const char * text_pool = static_cast<const char *>(base) + hdr->vocab_offset +
                sizeof(TwzmVocabHeader) +
                vhdr->n_vocab * sizeof(TwzmTokenData) +
                vhdr->n_merges * sizeof(uint32_t);
            const void * hash_tbl = text_pool + vhdr->text_pool_size;

            const struct llama_vocab * vocab = llama_model_get_vocab(model);
            if (vocab) {
                TWZM_LOG("vocab n_tokens before hash table: %d", llama_vocab_n_tokens(vocab));
                llama_vocab_load_token_to_id_from_hash_table(
                    vocab, hash_tbl, vhdr->token_hash_capacity, text_pool);
                TWZM_LOG("replaced token_to_id from pre-built hash table "
                         "(%" PRIu32 " entries, %" PRIu32 " capacity)",
                         vhdr->token_hash_count, vhdr->token_hash_capacity);
                TWZM_LOG("vocab n_tokens after hash table: %d", llama_vocab_n_tokens(vocab));
                // Sanity check: tokenize a simple string and log result.
                if (twzm_debug_level() > 0) {
                    const char * test = "Hello";
                    std::vector<llama_token> toks;
                    toks.resize(16);
                    int32_t n = llama_tokenize(vocab, test, strlen(test), toks.data(), toks.size(), false, false);
                    fprintf(stderr, "twzm:   test tokenize '%s' -> %d tokens:", test, n);
                    for (int32_t i = 0; i < n && i < 8; ++i) fprintf(stderr, " %d", toks[i]);
                    fprintf(stderr, "\n");
                }
            }
        }
    }

    // 5. Free the gguf_context (model has its own copy of all metadata).
    // NOTE: set gguf_ctx pointer in loader to null before freeing so the
    // callback (which may still run during model cleanup) doesn't dereference it.
    loader.gguf_ctx = nullptr;
    gguf_free(gguf_ctx);

    if (do_time)
        fprintf(stderr, "twzm: [time] %-40s %6.1f ms  (total)\n", "", twzm_ms(t0_total));

    // Cache the model pointer in the header (MAP_PRIVATE COW write — file
    // on disk is unchanged, pointer lives only in this process's page copy).
    hdr_rw->cached_model_ptr = (uint64_t)(uintptr_t)model;

#undef TWZM_STAGE
    return model;
}

// ---------------------------------------------------------------------------
// llama_model_load_from_twizzler_object
// ---------------------------------------------------------------------------

// Helper that pairs a mapping with a model so the mapping can be freed when
// the model is freed.  We use a side-channel via a global registry rather
// than modifying llama_model internals.
//
// NOTE: llama.cpp does not expose a model-destruction hook, so we cannot
// automatically unmap when llama_model_free() is called.  Instead we
// document that callers must call twz_object_unmap() with the values
// returned from twz_object_map() after freeing the model, or use the
// lower-level llama_model_load_from_twzm() with an externally managed
// mapping lifetime.
//
// For convenience, llama_model_load_from_twizzler_object() still maps the
// object and returns both the model and—via the out parameters when non-
// NULL—the mapping handle so the caller can unmap it.

struct llama_model * llama_model_load_from_twzm_path(
        const char * path,
        struct llama_model_params params) {
    // Reuse the same mapping across calls for the same path so that the
    // cached_model_ptr written into the header is visible on the next call.
    static std::unordered_map<std::string, std::pair<void*, size_t>> s_mappings;
    auto it = s_mappings.find(path);
    void * base;
    size_t sz;
    if (it != s_mappings.end()) {
        base = it->second.first;
        sz   = it->second.second;
    } else {
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            fprintf(stderr, "twzm: cannot open '%s': %s\n", path, strerror(errno));
            return nullptr;
        }
        struct stat st;
        if (fstat(fd, &st) < 0) {
            fprintf(stderr, "twzm: fstat '%s' failed: %s\n", path, strerror(errno));
            close(fd);
            return nullptr;
        }
        sz = (size_t)st.st_size;
        // MAP_PRIVATE + PROT_WRITE: writes are copy-on-write (file never modified).
        // Required so the COW cache write into the header doesn't segfault.
        base = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        close(fd);
        if (base == MAP_FAILED) {
            fprintf(stderr, "twzm: mmap '%s' failed: %s\n", path, strerror(errno));
            return nullptr;
        }
        s_mappings[path] = {base, sz};
    }
    struct llama_model * model = llama_model_load_from_twzm(base, sz, params);
    if (!model) {
        s_mappings.erase(path);
        munmap(base, sz);
    }
    return model;
}

struct llama_model * llama_model_load_from_twizzler_object(
        twz_objid id,
        struct llama_model_params params) {
    size_t map_size = 0;
    void * map_base = twz_object_map(id, &map_size);
    if (!map_base) {
        return nullptr;
    }

    struct llama_model * model =
        llama_model_load_from_twzm(map_base, map_size, params);

    if (!model) {
        twz_object_unmap(map_base, map_size);
        return nullptr;
    }

    // The mapping must remain live for the lifetime of `model`.
    // See NOTE above: caller must eventually call
    //   twz_object_unmap(map_base, map_size)
    // after llama_model_free(model).
    //
    // TODO: once llama.cpp exposes a model-destruction callback, register
    // twz_object_unmap here to make lifetime management automatic.

    return model;
}
