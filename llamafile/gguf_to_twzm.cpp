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

// gguf-to-twzm: Convert a GGUF model file to the TWZM (Twizzler Model) format.
//
// Usage: gguf-to-twzm <input.gguf> <output.twzm>
//
// Writes THREE Twizzler objects:
//   - Root/metadata object, at <output.twzm> (header + tensor index + GGUF
//     metadata blob).
//   - Tensor-data object, at a freshly generated 128-bit id under
//     $TWZ_OBJECT_PATH (all raw tensor bytes, page-aligned).
//   - Vocab object, at another freshly generated id under $TWZ_OBJECT_PATH
//     (flat vocab section - tokens, hash tables, merges, piece cache).
// The root object refers to the other two via a TwzmGlobalPtr (id + offset)
// embedded in its header. Loading the root (by path or by its own id) later
// requires $TWZ_OBJECT_PATH to be set to the same directory used here, so
// the embedded references can be resolved.
//
// Root object layout:
//
//   [  0 ]  TwzmHeader (96 bytes)
//   [ 96 ]  Tensor index: tensor_count × TwzmTensorEntry (112 bytes each)
//   [  A ]  GGUF metadata blob (bytes [0, data_offset) of the original GGUF)
//
// where A = 96 + tensor_count×112.

#include "twizzler.h"
#include "twizzler_platform.h"

#include "../llama.cpp/include/llama.h"

// Use relative path to avoid llamafile/llama.h shadowing the real gguf header.
#include "../llama.cpp/ggml/include/gguf.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// FNV-1a 32-bit hash (must match llama.cpp/src/llama-vocab.cpp).
static uint32_t fnv1a(const uint8_t * data, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t next_pow2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

// Align `v` up to the next multiple of `align` (must be a power of two).
static uint64_t align_up(uint64_t v, uint64_t align) {
    return (v + align - 1u) & ~(align - 1u);
}

// Copy `len` bytes starting at `src_offset` in `src` to `dst + dst_offset`,
// where `dst` points into a mapped, writable TWZM object.
static bool copy_bytes(uint8_t * dst, uint64_t dst_offset, FILE * src, uint64_t src_offset, uint64_t len) {
    if (fseeko(src, (off_t)src_offset, SEEK_SET) != 0) {
        fprintf(stderr, "gguf-to-twzm: fseeko failed: %s\n", strerror(errno));
        return false;
    }

    const size_t buf_size = 1 << 20; // 1 MiB
    std::vector<uint8_t> buf(buf_size);
    uint64_t remaining = len;
    uint64_t off = dst_offset;

    while (remaining > 0) {
        size_t to_read = remaining < buf_size ? (size_t)remaining : buf_size;
        size_t got = fread(buf.data(), 1, to_read, src);
        if (got == 0) {
            fprintf(stderr, "gguf-to-twzm: unexpected EOF while copying %" PRIu64 " bytes\n",
                    remaining);
            return false;
        }
        memcpy(dst + off, buf.data(), got);
        off += got;
        remaining -= got;
    }
    return true;
}

int main(int argc, char ** argv) {
    bool verify = false;
    if (argc == 4 && strcmp(argv[3], "--verify") == 0) {
        verify = true;
        argc = 3; // treat as if --verify wasn't passed for path args
    }
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.twzm> [--verify]\n", argv[0]);
        return 1;
    }
    const char * in_path  = argv[1];
    const char * out_path = argv[2];

    // Needed later for a vocab_only model load, used to precompute the
    // token-to-piece cache. Cheap in this build (CPU-only, statically
    // linked backend - no dlopen/directory scan).
    llama_backend_init();

    // -----------------------------------------------------------------------
    // 1. Parse the GGUF metadata (no tensor data allocation).
    // -----------------------------------------------------------------------
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(in_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "gguf-to-twzm: failed to parse GGUF metadata from '%s'\n", in_path);
        return 1;
    }

    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    if (n_tensors < 0) {
        fprintf(stderr, "gguf-to-twzm: gguf_get_n_tensors returned %" PRId64 "\n", n_tensors);
        gguf_free(gguf_ctx);
        return 1;
    }

    // Byte offset in the GGUF file where tensor data begins.
    // Everything before this offset is the metadata blob we will embed.
    uint64_t gguf_data_offset = (uint64_t)gguf_get_data_offset(gguf_ctx);

    // -----------------------------------------------------------------------
    // 1b. Detect weight tying (output.weight absent, aliased to token_embd).
    //     We add a synthetic sorted entry so the loader can find it via bsearch
    //     without any dynamic allocation.
    // -----------------------------------------------------------------------
    int64_t weight_tying_src = -1; // GGUF index of token_embd.weight
    {
        bool has_output = false;
        int64_t tok_embd_i = -1;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * nm = gguf_get_tensor_name(gguf_ctx, i);
            if (strcmp(nm, "output.weight") == 0)     has_output = true;
            if (strcmp(nm, "token_embd.weight") == 0) tok_embd_i = i;
        }
        if (!has_output && tok_embd_i >= 0) {
            weight_tying_src = tok_embd_i;
            fprintf(stderr, "  weight tying:    output.weight aliased to token_embd.weight\n");
        }
    }

    // Total entries = real tensors + synthetic weight-tying aliases.
    int64_t n_entries = n_tensors + (weight_tying_src >= 0 ? 1 : 0);

    fprintf(stderr, "gguf-to-twzm: %s\n", in_path);
    fprintf(stderr, "  tensors:         %" PRId64 "\n", n_tensors);
    fprintf(stderr, "  index entries:   %" PRId64 "\n", n_entries);
    fprintf(stderr, "  metadata blob:   %" PRIu64 " bytes (source)\n", gguf_data_offset);

    // -----------------------------------------------------------------------
    // 2. Compute layouts. The root object holds header + tensor index +
    //    metadata blob only; tensor data and vocab each live in their own
    //    dedicated object (see top-of-file comment), so their layouts are
    //    computed independently, each starting at offset 0.
    //
    //    metadata_size / root_total_size are NOT known yet: once a vocab
    //    object exists, the tokenizer arrays are stripped from the root's
    //    metadata blob (see step 6), which changes its size. They are
    //    resolved after the vocab step.
    // -----------------------------------------------------------------------
    const uint64_t header_size       = (uint64_t)sizeof(TwzmHeader);
    const uint64_t entry_size        = (uint64_t)sizeof(TwzmTensorEntry);
    const uint64_t tensor_index_off  = header_size;
    const uint64_t tensor_index_size = (uint64_t)n_entries * entry_size;
    const uint64_t metadata_off      = tensor_index_off + tensor_index_size;

    // Tensor-data object layout: page-aligned regions starting at offset 0.
    uint64_t next_data_off = 0;

    // Build the tensor index in GGUF tensor order first, then sort by name.
    // We also save per-tensor data offsets before sorting so the write phase
    // can use them (sort reorders index[] but doesn't change what offset each
    // GGUF tensor should be written to).
    std::vector<TwzmTensorEntry> index((size_t)n_entries);
    std::vector<uint64_t>        gguf_tensor_data_off((size_t)n_tensors);
    uint64_t total_tensor_bytes = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);

        if (strlen(name) >= TWZM_TENSOR_NAME_MAX) {
            fprintf(stderr, "gguf-to-twzm: tensor name '%s' too long (max %d)\n",
                    name, TWZM_TENSOR_NAME_MAX - 1);
            gguf_free(gguf_ctx);
            return 1;
        }

        memset(&index[i], 0, sizeof(TwzmTensorEntry));
        strncpy(index[i].name, name, TWZM_TENSOR_NAME_MAX - 1);
        index[i].data_offset = next_data_off;
        index[i].data_size   = (uint64_t)tensor_size;

        // Type and shape make the index self-sufficient, so the loader never
        // needs a gguf_tensor_info and the blob can drop them (step 6).
        index[i].type = (uint32_t)gguf_get_tensor_type(gguf_ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(gguf_ctx, i);
        uint32_t n_dims = 1;
        for (int d = 0; d < 4; ++d) {
            index[i].ne[d] = ne[d];
            if (ne[d] > 1) {
                n_dims = (uint32_t)d + 1;
            }
        }
        index[i].n_dims = n_dims;

        gguf_tensor_data_off[i] = next_data_off; // save before sorting

        next_data_off  = align_up(next_data_off + (uint64_t)tensor_size, TWZM_DATA_ALIGNMENT);
        total_tensor_bytes += (uint64_t)tensor_size;
    }

    // Add synthetic weight-tying alias (points at token_embd.weight's data).
    if (weight_tying_src >= 0) {
        TwzmTensorEntry & e = index[n_tensors];
        memset(&e, 0, sizeof(TwzmTensorEntry));
        strncpy(e.name, "output.weight", TWZM_TENSOR_NAME_MAX - 1);
        e.data_offset = index[weight_tying_src].data_offset;
        e.data_size   = index[weight_tying_src].data_size;
        e.type        = index[weight_tying_src].type;
        e.n_dims      = index[weight_tying_src].n_dims;
        memcpy(e.ne, index[weight_tying_src].ne, sizeof(e.ne));
    }

    // Sort all entries by name so the loader can use bsearch (zero allocation).
    std::sort(index.begin(), index.end(),
              [](const TwzmTensorEntry & a, const TwzmTensorEntry & b) {
                  return strcmp(a.name, b.name) < 0;
              });

    const uint64_t tensor_data_total_size = next_data_off;

    fprintf(stderr, "  tensor data:     %" PRIu64 " bytes\n", total_tensor_bytes);
    fprintf(stderr, "  tensor-data obj: %" PRIu64 " bytes\n", tensor_data_total_size);

    // -----------------------------------------------------------------------
    // 3. Open input GGUF as raw bytes.
    // -----------------------------------------------------------------------
    FILE * fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "gguf-to-twzm: cannot open '%s': %s\n", in_path, strerror(errno));
        gguf_free(gguf_ctx);
        return 1;
    }

    // -----------------------------------------------------------------------
    // 4. Create the tensor-data object and write all tensor bytes into it.
    //    Created before the root, since the root header embeds this
    //    object's freshly generated id.
    // -----------------------------------------------------------------------
    twz_objid tensor_data_id{};
    uint8_t * td_obj = static_cast<uint8_t *>(
        twz_object_create_fresh(&tensor_data_id, tensor_data_total_size));
    if (!td_obj) {
        fprintf(stderr, "gguf-to-twzm: cannot create tensor-data object\n");
        fclose(fin);
        gguf_free(gguf_ctx);
        return 1;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        // gguf_get_tensor_offset() returns the offset of tensor i within the
        // GGUF data section (i.e. relative to gguf_data_offset, not file start).
        uint64_t in_offset = gguf_data_offset + (uint64_t)gguf_get_tensor_offset(gguf_ctx, i);
        uint64_t in_size   = (uint64_t)gguf_get_tensor_size(gguf_ctx, i);

        if (!copy_bytes(td_obj, gguf_tensor_data_off[i], fin, in_offset, in_size)) {
            fprintf(stderr, "gguf-to-twzm: failed to copy tensor '%s'\n",
                    gguf_get_tensor_name(gguf_ctx, i));
            twz_object_finalize(td_obj, tensor_data_total_size);
            twz_object_destroy(tensor_data_id);
            fclose(fin);
            gguf_free(gguf_ctx);
            return 1;
        }

        if ((i + 1) % 100 == 0 || i == n_tensors - 1) {
            fprintf(stderr, "\r  copying tensors: %" PRId64 "/%" PRId64, i + 1, n_tensors);
        }
    }
    fprintf(stderr, "\n");

    twz_object_finalize(td_obj, tensor_data_total_size);
    fprintf(stderr, "gguf-to-twzm: tensor-data object: %016" PRIx64 "_%016" PRIx64 "\n",
            tensor_data_id.hi, tensor_data_id.lo);

    // -----------------------------------------------------------------------
    // 5. Build the flat vocab section and write it to its own object.
    //    Unlike the old single-object format, the object is created at its
    //    final size directly (vsize is fully known before any bytes are
    //    written) - no grow-after-the-fact step needed.
    // -----------------------------------------------------------------------
    twz_objid vocab_id{}; // stays {0,0} (absent) if no vocab object is created
    {
        // Re-open gguf_ctx to read token data.
        struct gguf_context * vctx = gguf_init_from_file(in_path, gguf_params);
        if (vctx) {
            // Find tokenizer KV indices.
            const int tok_idx   = gguf_find_key(vctx, "tokenizer.ggml.tokens");
            const int score_idx = gguf_find_key(vctx, "tokenizer.ggml.scores");
            const int type_idx  = gguf_find_key(vctx, "tokenizer.ggml.token_type");
            const int merge_idx = gguf_find_key(vctx, "tokenizer.ggml.merges");
            const int vtype_idx = gguf_find_key(vctx, "tokenizer.ggml.model");
            const uint32_t nv = tok_idx >= 0 ? (uint32_t)gguf_get_arr_n(vctx, tok_idx) : 0;
            const uint32_t nm = merge_idx >= 0 ? (uint32_t)gguf_get_arr_n(vctx, merge_idx) : 0;

            // Determine vocab_type integer from model string.
            uint32_t vtype = 2; // LLAMA_VOCAB_TYPE_BPE default
            if (vtype_idx >= 0) {
                const char * vm = gguf_get_val_str(vctx, vtype_idx);
                if (strcmp(vm, "llama") == 0)   vtype = 2;
                else if (strcmp(vm, "gpt2") == 0)  vtype = 2;
                else if (strcmp(vm, "bert") == 0)  vtype = 3;
                else if (strcmp(vm, "rwkv") == 0)  vtype = 5;
            }

            // Build text pool: tokens then merges.
            std::vector<uint8_t> pool;
            std::vector<uint32_t> tok_offsets(nv), merge_offsets(nm);
            uint32_t max_token_len = 0;
            for (uint32_t i = 0; i < nv; ++i) {
                tok_offsets[i] = (uint32_t)pool.size();
                const char * s = (tok_idx >= 0) ? gguf_get_arr_str(vctx, tok_idx, i) : "";
                size_t slen = strlen(s);
                if (slen > max_token_len) max_token_len = (uint32_t)slen;
                size_t len = slen + 1;
                pool.insert(pool.end(), s, s + len);
            }
            for (uint32_t i = 0; i < nm; ++i) {
                merge_offsets[i] = (uint32_t)pool.size();
                const char * s = gguf_get_arr_str(vctx, merge_idx, i);
                size_t len = strlen(s) + 1;
                pool.insert(pool.end(), s, s + len);
            }

            // Per-token data (score, attr, text_offset).
            struct TokEntry { float score; int32_t attr; uint32_t text_offset; };
            std::vector<TokEntry> td(nv);
            const float   * scores = (score_idx >= 0) ?
                (const float *)gguf_get_arr_data(vctx, score_idx) : nullptr;
            const int32_t * types  = (type_idx  >= 0) ?
                (const int32_t *)gguf_get_arr_data(vctx, type_idx) : nullptr;
            for (uint32_t i = 0; i < nv; ++i) {
                td[i].score = scores ? scores[i] : 0.f;
                int32_t attr = LLAMA_TOKEN_ATTR_NORMAL;
                if (types) {
                    switch (types[i]) {
                        case LLAMA_TOKEN_TYPE_UNKNOWN:      attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
                        case LLAMA_TOKEN_TYPE_UNUSED:       attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
                        case LLAMA_TOKEN_TYPE_NORMAL:       attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
                        case LLAMA_TOKEN_TYPE_CONTROL:      attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
                        case LLAMA_TOKEN_TYPE_USER_DEFINED: attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
                        case LLAMA_TOKEN_TYPE_BYTE:         attr = LLAMA_TOKEN_ATTR_BYTE;         break;
                        case LLAMA_TOKEN_TYPE_UNDEFINED:    attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                        default:                            attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                    }
                }
                td[i].attr        = attr;
                td[i].text_offset = tok_offsets[i];
            }

            // Header layout declared early so its size can be used below
            // without a forward-reference chicken-and-egg problem.
            struct FlatHdr {
                uint32_t magic, n_vocab, vocab_type, n_merges, text_pool_size;
                uint32_t token_hash_offset, token_hash_capacity, token_hash_count;
                uint32_t merge_hash_offset, merge_hash_capacity, merge_hash_count;
                uint32_t piece_data_offset, piece_pool_offset, piece_pool_size;
                uint32_t max_token_len;
            };

            // Build open-addressing hash tables (text -> id) so the loader can
            // search them directly against the mmap'd file instead of building
            // in-memory unordered_maps at load time. Must mirror
            // llama_vocab::impl::hash_lookup()/merge_hash_lookup()'s probe
            // sequence exactly (same fnv1a, same linear probing).
            struct HashEntry { uint32_t hash_value; int32_t token_id; uint32_t text_offset; };
            const char * pool_base = reinterpret_cast<const char *>(pool.data());

            std::vector<HashEntry> hash_tbl;
            uint32_t hash_cap = 0;
            if (nv > 0) {
                hash_cap = next_pow2((uint32_t)((nv * 4 + 2) / 3)); // load factor < 75%
                if (hash_cap < 16) hash_cap = 16;
                hash_tbl.assign(hash_cap, HashEntry{0, -1, 0});
                for (uint32_t i = 0; i < nv; ++i) {
                    const char * s = pool_base + tok_offsets[i];
                    uint32_t h = fnv1a(reinterpret_cast<const uint8_t *>(s), strlen(s));
                    uint32_t idx = h & (hash_cap - 1);
                    while (hash_tbl[idx].token_id >= 0) {
                        idx = (idx + 1) & (hash_cap - 1);
                    }
                    hash_tbl[idx] = HashEntry{h, (int32_t)i, tok_offsets[i]};
                }
            }

            // Merge-rank hash table: keyed by the exact "p1 p2" merge text
            // (as already stored in the pool at merge_offsets[i]); value is
            // the merge rank (i). Lets find_bpe_rank() probe this directly
            // instead of rebuilding an unordered_map<pair<string,string>,int>
            // (bpe_ranks) at load time.
            std::vector<HashEntry> merge_hash_tbl;
            uint32_t merge_hash_cap = 0;
            if (nm > 0) {
                merge_hash_cap = next_pow2((uint32_t)((nm * 4 + 2) / 3));
                if (merge_hash_cap < 16) merge_hash_cap = 16;
                merge_hash_tbl.assign(merge_hash_cap, HashEntry{0, -1, 0});
                for (uint32_t i = 0; i < nm; ++i) {
                    const char * s = pool_base + merge_offsets[i];
                    uint32_t h = fnv1a(reinterpret_cast<const uint8_t *>(s), strlen(s));
                    uint32_t idx = h & (merge_hash_cap - 1);
                    while (merge_hash_tbl[idx].token_id >= 0) {
                        idx = (idx + 1) & (merge_hash_cap - 1);
                    }
                    merge_hash_tbl[idx] = HashEntry{h, (int32_t)i, merge_offsets[i]};
                }
            }

            // Precompute the token-to-piece cache (llama_token_to_piece with
            // special=true, lstrip=0 - matches token_to_piece_for_cache()'s
            // call exactly) once, offline, instead of paying for it on every
            // load. Uses a real vocab_only model load rather than
            // reimplementing the decode logic, so this is guaranteed to
            // match runtime output byte-for-byte (including the load-time
            // special-token-by-text auto-detection, which raw GGUF metadata
            // alone doesn't capture).
            std::vector<uint8_t> piece_pool;
            std::vector<TwzmPieceEntry> piece_entries;
            {
                llama_model_params vp = llama_model_default_params();
                vp.vocab_only = true;
                llama_model * vmodel = llama_model_load_from_file(in_path, vp);
                if (vmodel) {
                    const llama_vocab * vvocab = llama_model_get_vocab(vmodel);
                    piece_entries.resize(nv);
                    std::vector<char> buf(256);
                    for (uint32_t id = 0; id < nv; ++id) {
                        int32_t n = llama_token_to_piece(vvocab, (llama_token)id,
                            buf.data(), (int32_t)buf.size(), 0, true);
                        if (n < 0) {
                            buf.resize((size_t)(-n));
                            n = llama_token_to_piece(vvocab, (llama_token)id,
                                buf.data(), (int32_t)buf.size(), 0, true);
                        }
                        piece_entries[id] = TwzmPieceEntry{(uint32_t)piece_pool.size(), (uint32_t)n};
                        piece_pool.insert(piece_pool.end(), buf.begin(), buf.begin() + n);
                    }
                    llama_model_free(vmodel);
                } else {
                    fprintf(stderr, "gguf-to-twzm: warning: vocab_only load failed, "
                            "piece cache will be rebuilt at load time instead\n");
                    piece_entries.clear();
                }
            }

            const uint32_t token_hash_offset = hash_tbl.empty() ? 0 : (uint32_t)(
                sizeof(FlatHdr) +
                sizeof(TokEntry) * nv +
                sizeof(uint32_t) * nm +
                pool.size());
            const uint32_t merge_hash_offset = merge_hash_tbl.empty() ? 0 : (uint32_t)(
                sizeof(FlatHdr) +
                sizeof(TokEntry) * nv +
                sizeof(uint32_t) * nm +
                pool.size() +
                hash_tbl.size() * sizeof(HashEntry));
            const uint32_t piece_data_offset = piece_entries.empty() ? 0 : (uint32_t)(
                sizeof(FlatHdr) +
                sizeof(TokEntry) * nv +
                sizeof(uint32_t) * nm +
                pool.size() +
                hash_tbl.size() * sizeof(HashEntry) +
                merge_hash_tbl.size() * sizeof(HashEntry));
            const uint32_t piece_pool_offset = piece_entries.empty() ? 0 :
                piece_data_offset + (uint32_t)(piece_entries.size() * sizeof(TwzmPieceEntry));

            FlatHdr fh = {0x4D435657u, nv, vtype, nm,
                    (uint32_t)pool.size(), token_hash_offset, hash_cap, nv,
                    merge_hash_offset, merge_hash_cap, nm,
                    piece_data_offset, piece_pool_offset, (uint32_t)piece_pool.size(),
                    max_token_len};

            const uint64_t vsize =
                sizeof(fh) +
                sizeof(TokEntry) * nv +
                sizeof(uint32_t) * nm +
                pool.size() +
                hash_tbl.size() * sizeof(HashEntry) +
                merge_hash_tbl.size() * sizeof(HashEntry) +
                piece_entries.size() * sizeof(TwzmPieceEntry) +
                piece_pool.size();

            uint8_t * vocab_obj = static_cast<uint8_t *>(twz_object_create_fresh(&vocab_id, vsize));
            if (!vocab_obj) {
                fprintf(stderr, "gguf-to-twzm: failed to create vocab object\n");
                twz_object_destroy(tensor_data_id);
                gguf_free(vctx);
                fclose(fin);
                gguf_free(gguf_ctx);
                return 1;
            }

            // Write: header, token data, merge offsets, text pool, token
            // hash table, merge hash table, piece data, piece pool.
            uint64_t w = 0;
            memcpy(vocab_obj + w, &fh, sizeof(fh));                          w += sizeof(fh);
            memcpy(vocab_obj + w, td.data(), sizeof(TokEntry) * nv);         w += sizeof(TokEntry) * nv;
            memcpy(vocab_obj + w, merge_offsets.data(), sizeof(uint32_t) * nm); w += sizeof(uint32_t) * nm;
            memcpy(vocab_obj + w, pool.data(), pool.size());                 w += pool.size();
            if (!hash_tbl.empty()) {
                memcpy(vocab_obj + w, hash_tbl.data(), hash_tbl.size() * sizeof(HashEntry));
                w += hash_tbl.size() * sizeof(HashEntry);
            }
            if (!merge_hash_tbl.empty()) {
                memcpy(vocab_obj + w, merge_hash_tbl.data(), merge_hash_tbl.size() * sizeof(HashEntry));
                w += merge_hash_tbl.size() * sizeof(HashEntry);
            }
            if (!piece_entries.empty()) {
                memcpy(vocab_obj + w, piece_entries.data(), piece_entries.size() * sizeof(TwzmPieceEntry));
                w += piece_entries.size() * sizeof(TwzmPieceEntry);
                memcpy(vocab_obj + w, piece_pool.data(), piece_pool.size());
                w += piece_pool.size();
            }

            twz_object_finalize(vocab_obj, vsize);
            fprintf(stderr, "gguf-to-twzm: vocab object: %016" PRIx64 "_%016" PRIx64 "\n",
                    vocab_id.hi, vocab_id.lo);
            fprintf(stderr, "gguf-to-twzm: vocab section: %u tokens, %u merges, "
                    "token hash %u/%u slots, merge hash %u/%u slots, "
                    "piece cache %s, %" PRIu64 " bytes\n",
                    nv, nm, nv, hash_cap, nm, merge_hash_cap,
                    piece_entries.empty() ? "absent" : "precomputed", vsize);
        } else {
            fprintf(stderr, "gguf-to-twzm: warning: could not re-parse GGUF for vocab section, "
                    "vocab object will be absent (slower load-time vocab construction)\n");
        }
        if (vctx) gguf_free(vctx);
    }

    // -----------------------------------------------------------------------
    // 6. Build the root object's GGUF metadata blob.
    //
    //    When a vocab object exists, every token/merge the tokenizer needs is
    //    already in it, precomputed - so the corresponding GGUF arrays in the
    //    metadata blob are dead weight that gguf_init_from_buffer() would
    //    still pay to parse into a std::vector<std::string> (one heap
    //    allocation per element: ~128K tokens + ~280K merges for a Llama-3
    //    class model, ~20ms and ~7.8MB of the root object). Strip them and
    //    re-emit a compact blob instead of copying the source bytes verbatim.
    //
    //    NOT stripped: tokenizer.ggml.precompiled_charsmap, which UGM vocabs
    //    read unconditionally and which the vocab object does not carry.
    //
    //    Consequence: the vocab object becomes REQUIRED for such a root - the
    //    GGUF fallback vocab path can no longer construct a vocab. If no
    //    vocab object was created, the blob is copied verbatim as before so
    //    that fallback keeps working.
    // -----------------------------------------------------------------------
    //    Tensor infos are ALWAYS stripped: since v3 the tensor index carries
    //    type and shape, so it - not the blob - is the tensor table the loader
    //    reads. Constructing one gguf_tensor_info per tensor was ~95% of
    //    gguf_init_from_buffer()'s cost (0.52ms of 0.55ms at 579 tensors, and
    //    growing with tensor count); the 35 KV pairs that remain parse in
    //    0.03ms flat. This is why the blob is always re-emitted now rather than
    //    copied verbatim.
    std::vector<uint8_t> meta_blob;
    uint64_t metadata_size = 0;

    {
        struct gguf_context * mctx = gguf_init_from_file(in_path, gguf_params);
        if (!mctx) {
            fprintf(stderr, "gguf-to-twzm: cannot re-parse GGUF to build metadata blob\n");
            twz_object_destroy(tensor_data_id);
            twz_object_destroy(vocab_id);
            fclose(fin);
            gguf_free(gguf_ctx);
            return 1;
        }

        // Emit KV pairs only - gguf_set_kv copies every key across but no
        // tensor infos, so the result has n_tensors = 0.
        struct gguf_context * kv_only = gguf_init_empty();
        gguf_set_kv(kv_only, mctx);
        gguf_free(mctx);
        mctx = kv_only;

        // tokenizer.ggml.tokens doubles as the fallback source for n_vocab in
        // some architectures (ml.get_arr_n(LLM_KV_TOKENIZER_LIST, ...) when
        // <arch>.vocab_size is absent). Synthesize the scalar first so
        // removing the array can't lose that information.
        const int arch_idx = gguf_find_key(mctx, "general.architecture");
        const int toks_idx = gguf_find_key(mctx, "tokenizer.ggml.tokens");
        if (arch_idx >= 0 && toks_idx >= 0) {
            char vs_key[256];
            snprintf(vs_key, sizeof(vs_key), "%s.vocab_size", gguf_get_val_str(mctx, arch_idx));
            if (gguf_find_key(mctx, vs_key) < 0) {
                gguf_set_val_u32(mctx, vs_key, (uint32_t)gguf_get_arr_n(mctx, toks_idx));
                fprintf(stderr, "gguf-to-twzm: synthesized '%s' = %zu "
                        "(was only derivable from the stripped tokens array)\n",
                        vs_key, gguf_get_arr_n(mctx, toks_idx));
            }
        }

        if (vocab_id.hi || vocab_id.lo) {
            static const char * const strip_keys[] = {
                "tokenizer.ggml.tokens",
                "tokenizer.ggml.scores",
                "tokenizer.ggml.token_type",
                "tokenizer.ggml.merges",
            };
            for (const char * k : strip_keys) {
                gguf_remove_key(mctx, k);
            }
        }

        const size_t sz = gguf_get_meta_size(mctx);
        meta_blob.resize(sz);
        gguf_get_meta_data(mctx, meta_blob.data());
        gguf_free(mctx);

        metadata_size = (uint64_t)sz;
        fprintf(stderr, "gguf-to-twzm: metadata blob: %" PRIu64 " -> %" PRIu64 " bytes "
                "(%.2f%%; tensor infos stripped (tensor index supplies them)%s)\n",
                gguf_data_offset, metadata_size,
                gguf_data_offset ? 100.0 * (double)metadata_size / (double)gguf_data_offset : 0.0,
                (vocab_id.hi || vocab_id.lo)
                    ? ", tokenizer arrays stripped (vocab object supplies them)" : "");
    }

    const uint64_t root_total_size = metadata_off + metadata_size;
    fprintf(stderr, "  root object:     %" PRIu64 " bytes\n", root_total_size);

    // -----------------------------------------------------------------------
    // 7. Create the root object and write header + tensor index + metadata.
    //    Written last, since the header embeds the tensor-data/vocab ids
    //    generated above.
    // -----------------------------------------------------------------------
    uint8_t * obj = static_cast<uint8_t *>(twz_object_create_at_path(out_path, root_total_size));
    if (!obj) {
        fprintf(stderr, "gguf-to-twzm: cannot create '%s'\n", out_path);
        twz_object_destroy(tensor_data_id);
        if (vocab_id.hi || vocab_id.lo) twz_object_destroy(vocab_id);
        fclose(fin);
        gguf_free(gguf_ctx);
        return 1;
    }

    {
        TwzmHeader hdr = {};
        hdr.magic               = TWZM_MAGIC;
        hdr.version             = TWZM_VERSION;
        hdr.metadata_offset     = metadata_off;
        hdr.metadata_size       = metadata_size;
        hdr.tensor_index_offset = tensor_index_off;
        hdr.tensor_count        = (uint64_t)n_entries; // sorted; includes aliases
        hdr.tensor_data         = TwzmGlobalPtr{tensor_data_id, 0}; // required
        hdr.vocab               = TwzmGlobalPtr{vocab_id, 0};       // {0,0} id = absent
        memcpy(obj, &hdr, sizeof(hdr));
    }

    memcpy(obj + tensor_index_off, index.data(), (size_t)tensor_index_size);

    if (!meta_blob.empty()) {
        // Stripped/re-emitted blob (step 6) - already in memory.
        memcpy(obj + metadata_off, meta_blob.data(), meta_blob.size());
    } else if (!copy_bytes(obj, metadata_off, fin, 0, metadata_size)) {
        // No vocab object: copy the source metadata bytes verbatim so the
        // GGUF fallback vocab path still has its tokenizer arrays.
        fprintf(stderr, "gguf-to-twzm: failed to copy GGUF metadata blob\n");
        twz_object_finalize(obj, root_total_size);
        remove(out_path);
        twz_object_destroy(tensor_data_id);
        if (vocab_id.hi || vocab_id.lo) twz_object_destroy(vocab_id);
        fclose(fin);
        gguf_free(gguf_ctx);
        return 1;
    }

    twz_object_finalize(obj, root_total_size);
    fclose(fin);
    gguf_free(gguf_ctx);
    fprintf(stderr, "gguf-to-twzm: wrote root object '%s'\n", out_path);
    fprintf(stderr, "gguf-to-twzm: note: $TWZ_OBJECT_PATH must be set consistently "
            "(same value used here) when loading this model later\n");

    // -----------------------------------------------------------------------
    // 8. Optional verification: re-map all objects and spot-check each tensor.
    // -----------------------------------------------------------------------
    if (verify) {
        fprintf(stderr, "gguf-to-twzm: verifying ...\n");
        int mismatches = 0;

        size_t dst_size = 0;
        const uint8_t * dst_map = static_cast<const uint8_t *>(
            twz_object_map_at_path(out_path, &dst_size));
        if (!dst_map) {
            fprintf(stderr, "gguf-to-twzm: verify: cannot map output object\n");
            return 1;
        }

        TwzmHeader whdr;
        memcpy(&whdr, dst_map, sizeof(whdr));

        // 1. Header sanity
        {
            if (whdr.magic != TWZM_MAGIC)
                fprintf(stderr, "gguf-to-twzm: verify: wrong magic\n"), mismatches++;
            if (whdr.version != TWZM_VERSION)
                fprintf(stderr, "gguf-to-twzm: verify: wrong version\n"), mismatches++;
            if (whdr.cached_model_ptr != 0) {
                fprintf(stderr, "gguf-to-twzm: verify: cached_model_ptr is non-zero "
                        "(%" PRIu64 ") in freshly written file\n", whdr.cached_model_ptr);
                mismatches++;
            }
            if (whdr.tensor_count != (uint64_t)n_entries) {
                fprintf(stderr, "gguf-to-twzm: verify: tensor_count %" PRIu64
                        " != expected %" PRId64 "\n", whdr.tensor_count, n_entries);
                mismatches++;
            }
            if (mismatches == 0)
                fprintf(stderr, "gguf-to-twzm: verify: header OK "
                        "(%" PRIu64 " tensors)\n", whdr.tensor_count);
        }

        // 2. Map the tensor-data object and spot-check each tensor.
        size_t td_size = 0;
        const uint8_t * td_map = static_cast<const uint8_t *>(
            twz_object_map(whdr.tensor_data.id, &td_size));
        if (!td_map) {
            fprintf(stderr, "gguf-to-twzm: verify: cannot map tensor-data object\n");
            twz_object_unmap(const_cast<uint8_t *>(dst_map), dst_size);
            return 1;
        }

        // Re-open the GGUF to get tensor data offsets.
        struct gguf_context * vctx = gguf_init_from_file(in_path, gguf_params);
        if (!vctx) {
            fprintf(stderr, "gguf-to-twzm: verify: cannot re-parse GGUF\n");
            twz_object_unmap(const_cast<uint8_t *>(td_map), td_size);
            twz_object_unmap(const_cast<uint8_t *>(dst_map), dst_size);
            return 1;
        }
        uint64_t vgguf_data_off = (uint64_t)gguf_get_data_offset(vctx);

        FILE * fsrc = fopen(in_path, "rb");
        if (!fsrc) {
            fprintf(stderr, "gguf-to-twzm: verify: cannot re-open '%s'\n", in_path);
            gguf_free(vctx);
            twz_object_unmap(const_cast<uint8_t *>(td_map), td_size);
            twz_object_unmap(const_cast<uint8_t *>(dst_map), dst_size);
            return 1;
        }

        const size_t PROBE = 64; // bytes to compare per tensor
        std::vector<uint8_t> src_buf(PROBE);

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name(vctx, i);
            uint64_t src_off    = vgguf_data_off + (uint64_t)gguf_get_tensor_offset(vctx, i);
            uint64_t tsize      = (uint64_t)gguf_get_tensor_size(vctx, i);
            size_t   probe      = (size_t)(tsize < PROBE ? tsize : PROBE);

            // Find this tensor in the sorted index by name.
            TwzmTensorEntry key = {};
            strncpy(key.name, name, TWZM_TENSOR_NAME_MAX - 1);
            auto it = std::lower_bound(index.begin(), index.end(), key,
                [](const TwzmTensorEntry & a, const TwzmTensorEntry & b) {
                    return strcmp(a.name, b.name) < 0;
                });
            if (it == index.end() || strcmp(it->name, name) != 0) {
                fprintf(stderr, "gguf-to-twzm: verify: tensor '%s' not found in index\n", name);
                mismatches++;
                continue;
            }
            uint64_t dst_off = it->data_offset;

            if (fseeko(fsrc, (off_t)src_off, SEEK_SET) != 0 ||
                fread(src_buf.data(), 1, probe, fsrc) != probe) {
                fprintf(stderr, "gguf-to-twzm: verify: cannot read GGUF for '%s'\n", name);
                mismatches++;
                continue;
            }
            if (dst_off + probe > td_size) {
                fprintf(stderr, "gguf-to-twzm: verify: tensor-data offset out of bounds for '%s'\n", name);
                mismatches++;
                continue;
            }
            if (memcmp(src_buf.data(), td_map + dst_off, probe) != 0) {
                fprintf(stderr, "gguf-to-twzm: verify: MISMATCH for '%s' at GGUF off=%" PRIu64 " TWZM off=%" PRIu64 "\n",
                        name, src_off, dst_off);
                fprintf(stderr, "  gguf: ");
                for (size_t b = 0; b < probe; ++b) fprintf(stderr, "%02x", src_buf[b]);
                fprintf(stderr, "\n  twzm: ");
                for (size_t b = 0; b < probe; ++b) fprintf(stderr, "%02x", td_map[dst_off + b]);
                fprintf(stderr, "\n");
                mismatches++;
            }
        }

        fclose(fsrc);
        gguf_free(vctx);
        twz_object_unmap(const_cast<uint8_t *>(td_map), td_size);

        // 3. Map the vocab object (if present) and sanity-check it.
        if (whdr.vocab.id.hi != 0 || whdr.vocab.id.lo != 0) {
            size_t v_size = 0;
            const uint8_t * v_map = static_cast<const uint8_t *>(
                twz_object_map(whdr.vocab.id, &v_size));
            if (!v_map) {
                fprintf(stderr, "gguf-to-twzm: verify: cannot map vocab object\n");
                mismatches++;
            } else {
                uint32_t vmagic = 0;
                if (whdr.vocab.offset + sizeof(vmagic) <= v_size) {
                    memcpy(&vmagic, v_map + whdr.vocab.offset, sizeof(vmagic));
                }
                if (vmagic != TWZM_VOCAB_MAGIC) {
                    fprintf(stderr, "gguf-to-twzm: verify: vocab object bad magic 0x%08X\n", vmagic);
                    mismatches++;
                } else {
                    fprintf(stderr, "gguf-to-twzm: verify: vocab object OK\n");
                }
                twz_object_unmap(const_cast<uint8_t *>(v_map), v_size);
            }
        }

        twz_object_unmap(const_cast<uint8_t *>(dst_map), dst_size);

        if (mismatches == 0) {
            fprintf(stderr, "gguf-to-twzm: verify: all %" PRId64 " tensors OK\n", n_tensors);
        } else {
            fprintf(stderr, "gguf-to-twzm: verify: %d issue(s) FAILED\n", mismatches);
            return 1;
        }
    }

    return 0;
}
