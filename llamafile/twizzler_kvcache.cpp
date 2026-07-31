// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
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

// Persistent KV cache stored in a Twizzler object.
//
// Prompt evaluation is the dominant startup cost once TWZM has reduced model
// loading to a couple of milliseconds, and it is pure recomputation - every
// run re-decodes the same prompt. This file stores the KV state produced by
// llama_state_seq_get_data() in its own object so a later run with the same
// prompt prefix can restore it instead.
//
// Two properties drive the design:
//
//   1. The cache object is found by a DERIVED id, not by a pointer in the
//      root object. The root is mapped MAP_PRIVATE, so anything written into
//      its header is copy-on-write and dies with the process. Deriving the id
//      from the model identity needs no root mutation and no version bump.
//
//   2. Acquiring write access is split from sizing. llamafile's CLI pledges
//      "stdio rpath tty" before the model is loaded, so open() is gone by the
//      time the state's size is known. twzm_kv_cache_open() must therefore run
//      before the sandbox; twz_object_handle_map()/twz_object_resize() do the
//      rest afterwards, which the pledge still permits on an open descriptor.
//
// Nothing here is ever fatal. A missing, stale, corrupt or unwritable cache
// degrades to a normal cold run - the cache is an optimisation, and a failure
// to use it must never stop a model from running.

#include "twizzler.h"
#include "twizzler_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <new>

// Set TWZM_DEBUG=1 to trace cache hits/misses.
static int twzm_kv_debug() {
    static int level = -1;
    if (level < 0) {
        const char * v = getenv("TWZM_DEBUG");
        level = v ? atoi(v) : 0;
    }
    return level;
}
#define KV_LOG(fmt, ...) \
    do { if (twzm_kv_debug() > 0) fprintf(stderr, "twzm: [kv] " fmt "\n", ##__VA_ARGS__); } while (0)

// ---------------------------------------------------------------------------
// Cache id derivation
// ---------------------------------------------------------------------------

// SplitMix64 finalizer - a strong 64-bit avalanche mix. Used rather than
// FNV-1a (which the vocab tables use for short strings) because the inputs
// here are a handful of integers, where FNV's per-byte mixing is both slower
// and weaker.
static uint64_t twzm_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

int twzm_kv_debug_level(void) {
    return twzm_kv_debug();
}

twz_objid twzm_kv_cache_id(twz_objid model_id) {
    // Domain separator: keeps a derived cache id from ever colliding with a
    // converter-minted object id, which is drawn from getrandom().
    const uint64_t kDomain = 0x54575A4D4B565F31ull; // "TWZMKV_1"

    uint64_t h = kDomain;
    h = twzm_mix64(h ^ model_id.hi);
    h = twzm_mix64(h ^ model_id.lo);
    h = twzm_mix64(h ^ TWZM_KV_VERSION);

    twz_objid id;
    id.hi = h;
    id.lo = twzm_mix64(h ^ kDomain);
    // {0,0} is the format's "absent" sentinel; astronomically unlikely, but
    // the check costs nothing and keeps the invariant total.
    if (id.hi == 0 && id.lo == 0) {
        id.lo = 1;
    }
    return id;
}

twz_objid twzm_file_model_id(const char * path) {
    twz_objid none = {0, 0};
    if (!path || !*path) {
        return none;
    }

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return none;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return none;
    }

    // 64 KiB comfortably covers a GGUF's header plus its KV metadata for the
    // models this is used with; a short read (tiny file) is fine, we just mix
    // whatever there was.
    unsigned char buf[64 * 1024];
    ssize_t got = read(fd, buf, sizeof(buf));
    close(fd);
    if (got < 0) {
        return none;
    }

    const uint64_t kDomain = 0x54575A4D46494C45ull; // "TWZMFILE"
    uint64_t h = twzm_mix64(kDomain ^ (uint64_t) st.st_size);

    // Fold the prefix eight bytes at a time. memcpy rather than a cast: `buf`
    // has no alignment guarantee relative to uint64_t.
    ssize_t i = 0;
    for (; i + 8 <= got; i += 8) {
        uint64_t word;
        memcpy(&word, buf + i, sizeof(word));
        h = twzm_mix64(h ^ word);
    }
    uint64_t tail = 0;
    for (ssize_t j = 0; i + j < got; ++j) {
        tail |= (uint64_t) buf[i + j] << (8 * j);
    }
    h = twzm_mix64(h ^ tail);

    twz_objid id;
    id.hi = h;
    id.lo = twzm_mix64(h ^ kDomain);
    if (id.hi == 0 && id.lo == 0) {
        id.lo = 1;
    }
    return id;
}

twz_objid twzm_root_model_id(const void * base, size_t size) {
    twz_objid none = {0, 0};
    if (!base || size < sizeof(TwzmHeader)) {
        return none;
    }
    TwzmHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (hdr.magic != TWZM_MAGIC || hdr.version != TWZM_VERSION) {
        return none;
    }
    return hdr.tensor_data.id;
}

// ---------------------------------------------------------------------------
// Cache handle
// ---------------------------------------------------------------------------

struct TwzmKvCache {
    twz_objid   cache_id  = {0, 0};
    twz_handle  handle    = TWZ_HANDLE_INVALID; // write access, if acquired
    void      * base      = nullptr;            // live MAP_SHARED mapping
    size_t      size      = 0;                  // bytes mapped at `base`
    bool        read_only = false;
    bool        validated = false;              // header checked and usable
};

// Validate the mapped object against the caller's model/geometry. Returns the
// header on success, nullptr on any mismatch (which is a miss, not an error).
static const TwzmKvHeader * kv_valid_header(const TwzmKvCache * kv,
                                            twz_objid model_id, uint32_t n_ctx,
                                            uint32_t type_k, uint32_t type_v) {
    if (!kv || !kv->base || kv->size < sizeof(TwzmKvHeader)) {
        return nullptr;
    }
    const TwzmKvHeader * h = static_cast<const TwzmKvHeader *>(kv->base);

    if (h->magic != TWZM_KV_MAGIC) {
        KV_LOG("miss: bad magic 0x%08" PRIx32, h->magic);
        return nullptr;
    }
    if (h->version != TWZM_KV_VERSION) {
        KV_LOG("miss: version %" PRIu32 " != %" PRIu32, h->version, TWZM_KV_VERSION);
        return nullptr;
    }
    if (h->model_id.hi != model_id.hi || h->model_id.lo != model_id.lo) {
        // A different model's cache. Can only happen if a derived id collided
        // or an object was hand-copied, but check anyway: restoring another
        // model's KV would produce silent garbage rather than a crash.
        KV_LOG("miss: model id mismatch");
        return nullptr;
    }
    if (h->n_ctx != n_ctx || h->type_k != type_k || h->type_v != type_v) {
        KV_LOG("miss: geometry changed (n_ctx %" PRIu32 "->%" PRIu32 ")", h->n_ctx, n_ctx);
        return nullptr;
    }
    if (h->state_version != (uint32_t) LLAMA_STATE_SEQ_VERSION) {
        KV_LOG("miss: llama state version %" PRIu32 " != %d",
               h->state_version, LLAMA_STATE_SEQ_VERSION);
        return nullptr;
    }
    if (h->n_tokens == 0 || h->blob_size == 0) {
        return nullptr;
    }

    // Bounds: every offset/length pair must lie inside the mapping. Checked
    // with subtraction rather than addition so a hostile size can't overflow.
    const uint64_t tok_bytes = (uint64_t) h->n_tokens * sizeof(int32_t);
    if (h->tokens_offset > kv->size || kv->size - h->tokens_offset < tok_bytes) {
        KV_LOG("miss: token array out of bounds");
        return nullptr;
    }
    if (h->blob_offset > kv->size || kv->size - h->blob_offset < h->blob_size) {
        KV_LOG("miss: state blob out of bounds");
        return nullptr;
    }
    return h;
}

struct TwzmKvCache * twzm_kv_cache_open(twz_objid cache_id, bool read_only) {
    TwzmKvCache * kv = new (std::nothrow) TwzmKvCache();
    if (!kv) {
        return nullptr;
    }
    kv->cache_id  = cache_id;
    kv->read_only = read_only;

    if (read_only) {
        size_t sz = 0;
        void * base = twz_object_map(cache_id, &sz);
        if (base) {
            kv->base = base;
            kv->size = sz;
        }
        return kv;
    }

    // Acquire write access now - this is the last moment open() is available.
    kv->handle = twz_object_open_rw(cache_id);
    if (kv->handle == TWZ_HANDLE_INVALID) {
        KV_LOG("could not acquire cache object; running without a cache");
        return kv;
    }

    const long existing = twz_object_handle_size(kv->handle);
    if (existing > 0) {
        // Map the previous contents so they can be read after the sandbox.
        // MAP_SHARED, so a same-size overwrite later needs no remap.
        void * base = twz_object_handle_map(kv->handle, (size_t) existing);
        kv->handle = TWZ_HANDLE_INVALID; // consumed by the map call
        if (base) {
            kv->base = base;
            kv->size = (size_t) existing;
        }
    }
    return kv;
}

const int32_t * twzm_kv_cache_tokens(struct TwzmKvCache * kv,
                                     twz_objid model_id, uint32_t n_ctx,
                                     uint32_t type_k, uint32_t type_v,
                                     uint32_t * out_n_tokens) {
    if (out_n_tokens) {
        *out_n_tokens = 0;
    }
    const TwzmKvHeader * h = kv_valid_header(kv, model_id, n_ctx, type_k, type_v);
    if (!h) {
        return nullptr;
    }
    kv->validated = true;
    if (out_n_tokens) {
        *out_n_tokens = h->n_tokens;
    }
    return reinterpret_cast<const int32_t *>(
        static_cast<const uint8_t *>(kv->base) + h->tokens_offset);
}

const void * twzm_kv_cache_blob(struct TwzmKvCache * kv, uint64_t * out_size) {
    if (out_size) {
        *out_size = 0;
    }
    // Only reachable after twzm_kv_cache_tokens() validated the header.
    if (!kv || !kv->validated || !kv->base) {
        return nullptr;
    }
    const TwzmKvHeader * h = static_cast<const TwzmKvHeader *>(kv->base);
    if (out_size) {
        *out_size = h->blob_size;
    }
    return static_cast<const uint8_t *>(kv->base) + h->blob_offset;
}

bool twzm_kv_cache_save(struct TwzmKvCache * kv, twz_objid model_id,
                        uint32_t n_ctx, uint32_t type_k, uint32_t type_v,
                        const int32_t * tokens, uint32_t n_tokens,
                        const void * blob, uint64_t blob_size) {
    if (!kv || kv->read_only) {
        return false;
    }
    if (!tokens || n_tokens == 0 || !blob || blob_size == 0) {
        return false;
    }
    if (kv->handle == TWZ_HANDLE_INVALID && !kv->base) {
        // Write access was never acquired (open failed before the sandbox).
        return false;
    }

    const uint64_t tokens_offset = sizeof(TwzmKvHeader);
    const uint64_t tok_bytes     = (uint64_t) n_tokens * sizeof(int32_t);
    const uint64_t blob_offset   = tokens_offset + tok_bytes;
    const uint64_t total         = blob_offset + blob_size;

    // Grow (or first-map) the object to fit. Both paths are permitted after
    // the privilege drop because the descriptor already exists.
    if (!kv->base) {
        kv->base = twz_object_handle_map(kv->handle, (size_t) total);
        kv->handle = TWZ_HANDLE_INVALID; // consumed either way
        if (!kv->base) {
            return false;
        }
        kv->size = (size_t) total;
    } else if (kv->size != (size_t) total) {
        void * nb = twz_object_resize(kv->base, kv->size, (size_t) total);
        if (!nb) {
            return false;
        }
        kv->base = nb;
        kv->size = (size_t) total;
    }

    uint8_t * out = static_cast<uint8_t *>(kv->base);

    // Header last: until it carries the right magic the object is not
    // considered a cache, so a crash mid-write leaves a miss, not a bad hit.
    memcpy(out + tokens_offset, tokens, (size_t) tok_bytes);
    memcpy(out + blob_offset, blob, (size_t) blob_size);

    TwzmKvHeader h;
    memset(&h, 0, sizeof(h));
    h.magic         = TWZM_KV_MAGIC;
    h.version       = TWZM_KV_VERSION;
    h.model_id      = model_id;
    h.n_ctx         = n_ctx;
    h.type_k        = type_k;
    h.type_v        = type_v;
    h.state_version = (uint32_t) LLAMA_STATE_SEQ_VERSION;
    h.n_tokens      = n_tokens;
    h.tokens_offset = tokens_offset;
    h.blob_offset   = blob_offset;
    h.blob_size     = blob_size;
    memcpy(out, &h, sizeof(h));

    KV_LOG("saved %" PRIu32 " tokens, %" PRIu64 " byte state (%zu byte object)",
           n_tokens, blob_size, kv->size);
    return true;
}

void twzm_kv_cache_close(struct TwzmKvCache * kv) {
    if (!kv) {
        return;
    }
    if (kv->base) {
        if (kv->read_only) {
            twz_object_unmap(kv->base, kv->size);
        } else {
            twz_object_finalize(kv->base, kv->size);
        }
    } else if (kv->handle != TWZ_HANDLE_INVALID) {
        twz_object_close(kv->handle);
    }
    delete kv;
}
