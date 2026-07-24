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
#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// ---------------------------------------------------------------------------
// Internal loader state
// ---------------------------------------------------------------------------

struct TwzmLoader {
    const uint8_t      * base = nullptr;
    size_t               size = 0;
    const gguf_context * gguf_ctx = nullptr; // borrowed; used to classify missing tensors

    // Map from tensor name to its data region within the object.
    struct Region {
        const uint8_t * ptr;
        size_t          len;
    };
    std::unordered_map<std::string, Region> tensor_map;

    // Open and validate a mapped TWZM object.
    // gguf is the already-parsed metadata context; we borrow it to validate
    // "not found" lookups at set_tensor_data time.
    // Returns true on success; logs an error and returns false on failure.
    bool open(const void * base_ptr, size_t sz, const gguf_context * gguf);
};

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
        fprintf(stderr, "twzm: metadata blob [%" PRIu64 ", %" PRIu64 ") out of bounds\n",
                hdr.metadata_offset, hdr.metadata_offset + hdr.metadata_size);
        return false;
    }

    // -- Validate and parse tensor index ----------------------------------
    uint64_t index_bytes = hdr.tensor_count * sizeof(TwzmTensorEntry);
    if (hdr.tensor_index_offset + index_bytes > sz) {
        fprintf(stderr, "twzm: tensor index [%" PRIu64 ", %" PRIu64 ") out of bounds\n",
                hdr.tensor_index_offset, hdr.tensor_index_offset + index_bytes);
        return false;
    }

    const TwzmTensorEntry * entries =
        reinterpret_cast<const TwzmTensorEntry *>(base + hdr.tensor_index_offset);

    tensor_map.reserve(static_cast<size_t>(hdr.tensor_count));
    for (uint64_t i = 0; i < hdr.tensor_count; ++i) {
        const TwzmTensorEntry & e = entries[i];

        // Validate name is null-terminated within bounds.
        if (strnlen(e.name, TWZM_TENSOR_NAME_MAX) == TWZM_TENSOR_NAME_MAX) {
            fprintf(stderr, "twzm: tensor index entry %" PRIu64 " name is not null-terminated\n", i);
            return false;
        }
        // Validate data region bounds.
        if (e.data_offset + e.data_size > sz) {
            fprintf(stderr, "twzm: tensor '%s' data [%" PRIu64 ", %" PRIu64 ") out of bounds\n",
                    e.name, e.data_offset, e.data_offset + e.data_size);
            return false;
        }

        tensor_map[std::string(e.name)] = Region{
            base + e.data_offset,
            static_cast<size_t>(e.data_size)
        };
    }

    TWZM_LOG("loaded %" PRIu64 " tensor(s) from index", hdr.tensor_count);
    if (twzm_debug_level() > 1) {
        for (const auto & kv : tensor_map) {
            fprintf(stderr, "twzm:   [index] %-48s  off=%" PRIu64 "  len=%zu\n",
                    kv.first.c_str(),
                    (uint64_t)(kv.second.ptr - base),
                    kv.second.len);
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

    auto it = loader->tensor_map.find(name);
    if (it == loader->tensor_map.end()) {
        // Distinguish two cases:
        //   (a) Tensor IS in the GGUF but missing from the TWZM index – genuine bug.
        //   (b) Tensor is NOT in the GGUF – it is either optional (correct zeros)
        //       or weight-tied and handled by the architecture; silently skip.
        if (loader->gguf_ctx &&
            gguf_find_tensor(loader->gguf_ctx, name) >= 0) {
            fprintf(stderr, "twzm: tensor '%s' is in GGUF metadata but missing "
                            "from TWZM index (conversion bug?)\n", name);
        } else {
            TWZM_LOG("set_tensor_data: %-48s  SKIPPED (not in GGUF)", name);
        }
        return;
    }

    const TwzmLoader::Region & region = it->second;

    size_t expected = ggml_nbytes(tensor);
    TWZM_LOG("set_tensor_data: %-48s  type=%-8s  expected=%zu  have=%zu",
             name, ggml_type_name(tensor->type), expected, region.len);
    if (region.len < expected) {
        fprintf(stderr, "twzm: tensor '%s': TWZM data %zu B < expected %zu B\n",
                name, region.len, expected);
        return;
    }
    // Use the backend API so the upload works for CPU, CUDA, Metal, etc.
    ggml_backend_tensor_set(tensor, region.ptr, 0, expected);

    // TWZM_DEBUG >= 2: dump raw bytes from the TWZM mapping so we can
    // verify the file content without relying on backend readback.
    if (twzm_debug_level() >= 2) {
        constexpr size_t PROBE = 16;
        size_t probe = expected < PROBE ? expected : PROBE;
        const uint8_t * src = region.ptr;
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

    // 1. Open and validate the TWZM object.
    //    Pass gguf_ctx=nullptr for now; we'll set it after parsing.
    TwzmLoader loader;
    if (!loader.open(base, size, nullptr)) {
        return nullptr;
    }

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

    // Give the loader the context so it can classify missing tensors correctly.
    loader.gguf_ctx = gguf_ctx;

    // 3. Handle weight tying: models like Llama 3.2 1B tie output.weight to
    //    token_embd.weight.  When output.weight is absent from the GGUF the
    //    llama.cpp files.empty() code path still creates an F32 tensor for it
    //    (it doesn't return nullptr for TENSOR_NOT_REQUIRED the same way the
    //    file-based path does), so weight tying is never triggered.  We fix
    //    this by:
    //      a) Adding output.weight to the TWZM tensor_map pointing at
    //         token_embd.weight's data region (same bytes, same type).
    //      b) Adding output.weight to the gguf_context so create_tensor picks
    //         up the correct quantised type rather than defaulting to F32.
    if (gguf_find_tensor(gguf_ctx, "output.weight") < 0) {
        auto tok_it = loader.tensor_map.find("token_embd.weight");
        int64_t tok_id = gguf_find_tensor(gguf_ctx, "token_embd.weight");
        if (tok_it != loader.tensor_map.end() && tok_id >= 0) {
            // (a) TWZM alias
            loader.tensor_map["output.weight"] = tok_it->second;
            TWZM_LOG("weight tying: output.weight aliased to token_embd.weight");

            // (b) gguf_context synthetic entry so create_tensor sees the right type
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

    // 4. Build the llama_model via the existing user-init path.
    //    llama_model_init_from_user() will call twzm_set_tensor_data for each
    //    tensor, then free the gguf_context it received.
    //    We disable mmap (we handle the mapping ourselves) and extra buffers.
    params.use_mmap       = false;
    params.use_extra_bufts = false;

    struct llama_model * model =
        llama_model_init_from_user(gguf_ctx, twzm_set_tensor_data, &loader, params);

    // 5. Free the gguf_context (model has its own copy of all metadata).
    // NOTE: set gguf_ctx pointer in loader to null before freeing so the
    // callback (which may still run during model cleanup) doesn't dereference it.
    loader.gguf_ctx = nullptr;
    gguf_free(gguf_ctx);

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
    size_t sz = (size_t)st.st_size;
    void * base = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        fprintf(stderr, "twzm: mmap '%s' failed: %s\n", path, strerror(errno));
        return nullptr;
    }
    struct llama_model * model = llama_model_load_from_twzm(base, sz, params);
    if (!model) {
        munmap(base, sz);
        return nullptr;
    }
    // The mapping must remain live for the model's lifetime.
    // It is intentionally not unmapped here; it will be released on process exit.
    // TODO: track {base,sz} keyed by model* and munmap from a llama_model_free wrapper.
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
