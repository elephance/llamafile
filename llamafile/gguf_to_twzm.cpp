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
// The TWZM file can then be loaded with llama_model_load_from_twizzler_object()
// (after placing it in $TWZ_OBJECT_PATH under the name <hi>_<lo>.twzm) or
// directly with llama_model_load_from_twzm().
//
// Output file layout:
//
//   [  0 ]  TwzmHeader (64 bytes)
//   [ 64 ]  Tensor index: tensor_count × TwzmTensorEntry (128 bytes each)
//   [  A ]  GGUF metadata blob (bytes [0, data_offset) of the original GGUF)
//   [  B ]  Tensor data regions (each page-aligned to TWZM_DATA_ALIGNMENT)
//
// where A = 64 + tensor_count×128 and B = ALIGN(A + metadata_size, PAGE).

#include "twizzler.h"

// Use relative path to avoid llamafile/llama.h shadowing the real gguf header.
#include "../llama.cpp/ggml/include/gguf.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Align `v` up to the next multiple of `align` (must be a power of two).
static uint64_t align_up(uint64_t v, uint64_t align) {
    return (v + align - 1u) & ~(align - 1u);
}

// Write `size` zero bytes to `fp`.
static bool write_zeros(FILE * fp, size_t size) {
    const size_t chunk = 65536;
    static const uint8_t zeros[65536] = {};
    while (size > 0) {
        size_t n = size < chunk ? size : chunk;
        if (fwrite(zeros, 1, n, fp) != n) return false;
        size -= n;
    }
    return true;
}

// Copy `len` bytes starting at `src_offset` from `src` to `dst`.
static bool copy_bytes(FILE * dst, FILE * src, uint64_t src_offset, uint64_t len) {
    if (fseeko(src, (off_t)src_offset, SEEK_SET) != 0) {
        fprintf(stderr, "gguf-to-twzm: fseeko failed: %s\n", strerror(errno));
        return false;
    }

    const size_t buf_size = 1 << 20; // 1 MiB
    std::vector<uint8_t> buf(buf_size);
    uint64_t remaining = len;

    while (remaining > 0) {
        size_t to_read = remaining < buf_size ? (size_t)remaining : buf_size;
        size_t got = fread(buf.data(), 1, to_read, src);
        if (got == 0) {
            fprintf(stderr, "gguf-to-twzm: unexpected EOF while copying %" PRIu64 " bytes\n",
                    remaining);
            return false;
        }
        if (fwrite(buf.data(), 1, got, dst) != got) {
            fprintf(stderr, "gguf-to-twzm: fwrite failed: %s\n", strerror(errno));
            return false;
        }
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
    fprintf(stderr, "  metadata blob:   %" PRIu64 " bytes\n", gguf_data_offset);

    // -----------------------------------------------------------------------
    // 2. Compute the TWZM file layout.
    // -----------------------------------------------------------------------
    const uint64_t header_size       = (uint64_t)sizeof(TwzmHeader);
    const uint64_t entry_size        = (uint64_t)sizeof(TwzmTensorEntry);
    const uint64_t tensor_index_off  = header_size;
    const uint64_t tensor_index_size = (uint64_t)n_entries * entry_size;
    const uint64_t metadata_off      = tensor_index_off + tensor_index_size;
    const uint64_t metadata_size     = gguf_data_offset;

    // First tensor data starts at the next page boundary after the metadata.
    uint64_t next_data_off = align_up(metadata_off + metadata_size, TWZM_DATA_ALIGNMENT);

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
    }

    // Sort all entries by name so the loader can use bsearch (zero allocation).
    std::sort(index.begin(), index.end(),
              [](const TwzmTensorEntry & a, const TwzmTensorEntry & b) {
                  return strcmp(a.name, b.name) < 0;
              });

    uint64_t total_file_size = next_data_off;

    fprintf(stderr, "  tensor data:     %" PRIu64 " bytes\n", total_tensor_bytes);
    fprintf(stderr, "  output size:     %" PRIu64 " bytes\n", total_file_size);

    // -----------------------------------------------------------------------
    // 3. Open input GGUF as raw bytes and output file for writing.
    // -----------------------------------------------------------------------
    FILE * fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "gguf-to-twzm: cannot open '%s': %s\n", in_path, strerror(errno));
        gguf_free(gguf_ctx);
        return 1;
    }

    FILE * fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "gguf-to-twzm: cannot create '%s': %s\n", out_path, strerror(errno));
        fclose(fin);
        gguf_free(gguf_ctx);
        return 1;
    }

    // -----------------------------------------------------------------------
    // 4. Write TwzmHeader.
    // -----------------------------------------------------------------------
    TwzmHeader hdr = {};
    hdr.magic               = TWZM_MAGIC;
    hdr.version             = TWZM_VERSION;
    hdr.metadata_offset     = metadata_off;
    hdr.metadata_size       = metadata_size;
    hdr.tensor_index_offset = tensor_index_off;
    hdr.tensor_count        = (uint64_t)n_entries; // sorted; includes aliases

    if (fwrite(&hdr, sizeof(hdr), 1, fout) != 1) {
        fprintf(stderr, "gguf-to-twzm: failed to write header\n");
        goto fail;
    }

    // -----------------------------------------------------------------------
    // 5. Write sorted tensor index.
    // -----------------------------------------------------------------------
    for (int64_t i = 0; i < n_entries; ++i) {
        if (fwrite(&index[i], sizeof(TwzmTensorEntry), 1, fout) != 1) {
            fprintf(stderr, "gguf-to-twzm: failed to write tensor index entry %" PRId64 "\n", i);
            goto fail;
        }
    }

    // -----------------------------------------------------------------------
    // 6. Write GGUF metadata blob (bytes [0, gguf_data_offset) of input file).
    // -----------------------------------------------------------------------
    if (!copy_bytes(fout, fin, 0, metadata_size)) {
        fprintf(stderr, "gguf-to-twzm: failed to copy GGUF metadata blob\n");
        goto fail;
    }

    // -----------------------------------------------------------------------
    // 7. Write tensor data regions at page-aligned offsets.
    //    Use the saved per-tensor offsets (pre-sort) so we know where each
    //    GGUF tensor was assigned regardless of the sorted index order.
    // -----------------------------------------------------------------------
    for (int64_t i = 0; i < n_tensors; ++i) {
        if (fseeko(fout, (off_t)gguf_tensor_data_off[i], SEEK_SET) != 0) {
            fprintf(stderr, "gguf-to-twzm: fseeko output failed: %s\n", strerror(errno));
            goto fail;
        }

        // gguf_get_tensor_offset() returns the offset of tensor i within the
        // GGUF data section (i.e. relative to gguf_data_offset, not file start).
        uint64_t in_offset = gguf_data_offset + (uint64_t)gguf_get_tensor_offset(gguf_ctx, i);
        uint64_t in_size   = (uint64_t)gguf_get_tensor_size(gguf_ctx, i);

        if (!copy_bytes(fout, fin, in_offset, in_size)) {
            fprintf(stderr, "gguf-to-twzm: failed to copy tensor '%s'\n",
                    gguf_get_tensor_name(gguf_ctx, i));
            goto fail;
        }

        if ((i + 1) % 100 == 0 || i == n_tensors - 1) {
            fprintf(stderr, "\r  copying tensors: %" PRId64 "/%" PRId64, i + 1, n_tensors);
        }
    }
    fprintf(stderr, "\n");

    // Ensure output file extends to total_file_size (last page boundary).
    // This matters if the last tensor data doesn't fill its page.
    {
        long cur = ftello(fout);
        if (cur >= 0 && (uint64_t)cur < total_file_size) {
            if (fseeko(fout, (off_t)(total_file_size - 1), SEEK_SET) != 0 ||
                fputc(0, fout) == EOF) {
                fprintf(stderr, "gguf-to-twzm: failed to extend output file\n");
                goto fail;
            }
        }
    }

    fclose(fin);
    fclose(fout);
    gguf_free(gguf_ctx);
    fprintf(stderr, "gguf-to-twzm: wrote '%s'\n", out_path);

    // -----------------------------------------------------------------------
    // 8. Append flat vocab section and update the header.
    // -----------------------------------------------------------------------
    {
        // Re-open gguf_ctx to read token data.
        struct gguf_context * vctx = gguf_init_from_file(in_path, gguf_params);
        FILE * fv = fopen(out_path, "r+b");
        if (vctx && fv) {
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
                else if (strcmp(vm, "rwkv") == 0)  vtype = 4;
            }

            // Build text pool: tokens then merges.
            std::vector<uint8_t> pool;
            std::vector<uint32_t> tok_offsets(nv), merge_offsets(nm);
            for (uint32_t i = 0; i < nv; ++i) {
                tok_offsets[i] = (uint32_t)pool.size();
                const char * s = (tok_idx >= 0) ? gguf_get_arr_str(vctx, tok_idx, i) : "";
                size_t len = strlen(s) + 1;
                pool.insert(pool.end(), s, s + len);
            }
            for (uint32_t i = 0; i < nm; ++i) {
                merge_offsets[i] = (uint32_t)pool.size();
                const char * s = gguf_get_arr_str(vctx, merge_idx, i);
                size_t len = strlen(s) + 1;
                pool.insert(pool.end(), s, s + len);
            }

            struct FlatHdr {
                uint32_t magic, n_vocab, vocab_type, n_merges,
                         text_pool_size, res[3];
            } fh = {0x4D435657u, nv, vtype, nm,
                    (uint32_t)pool.size(), {0,0,0}};

            // Per-token data (score, attr, text_offset).
            struct TokEntry { float score; int32_t attr; uint32_t text_offset; };
            std::vector<TokEntry> td(nv);
            const float   * scores = (score_idx >= 0) ?
                (const float *)gguf_get_arr_data(vctx, score_idx) : nullptr;
            const int32_t * types  = (type_idx  >= 0) ?
                (const int32_t *)gguf_get_arr_data(vctx, type_idx) : nullptr;
            for (uint32_t i = 0; i < nv; ++i) {
                td[i].score       = scores ? scores[i] : 0.f;
                td[i].attr        = types  ? types[i]  : 1; // NORMAL
                td[i].text_offset = tok_offsets[i];
            }

            // Seek to end of file and record vocab_offset.
            fseeko(fv, 0, SEEK_END);
            uint64_t voff = (uint64_t)ftello(fv);

            // Write: header, token data, merge offsets, text pool.
            fwrite(&fh,               sizeof(fh),    1,    fv);
            fwrite(td.data(),         sizeof(TokEntry), nv, fv);
            fwrite(merge_offsets.data(), sizeof(uint32_t), nm, fv);
            fwrite(pool.data(),       1, pool.size(),      fv);

            uint64_t vsize = (uint64_t)ftello(fv) - voff;

            // Patch vocab_offset + vocab_size into the header at the fixed offsets.
            // TwzmHeader layout: magic(4)+version(4)+meta_off(8)+meta_sz(8)+
            //   idx_off(8)+n(8)+cached(8)+vocab_off(8)+vocab_sz(8) = 64 bytes
            fseeko(fv, 48, SEEK_SET);  // offset of vocab_offset in header
            fwrite(&voff,  sizeof(voff),  1, fv);
            fwrite(&vsize, sizeof(vsize), 1, fv);

            fprintf(stderr, "gguf-to-twzm: vocab section: %u tokens, %u merges, "
                    "%" PRIu64 " bytes\n", nv, nm, vsize);
        }
        if (fv) fclose(fv);
        if (vctx) gguf_free(vctx);
    }

    // -----------------------------------------------------------------------
    // 8. Optional verification: re-open both files and spot-check each tensor.
    // -----------------------------------------------------------------------
    if (verify) {
        fprintf(stderr, "gguf-to-twzm: verifying ...\n");
        int mismatches = 0;

        // 1. Header sanity
        {
            FILE * fhdr = fopen(out_path, "rb");
            TwzmHeader whdr = {};
            bool ok = fhdr && fread(&whdr, sizeof(whdr), 1, fhdr) == 1;
            if (fhdr) fclose(fhdr);
            if (!ok) {
                fprintf(stderr, "gguf-to-twzm: verify: cannot read output header\n");
                mismatches++;
            } else {
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
        }

        // Re-open the GGUF to get tensor data offsets.
        struct gguf_context * vctx = gguf_init_from_file(in_path, gguf_params);
        if (!vctx) {
            fprintf(stderr, "gguf-to-twzm: verify: cannot re-parse GGUF\n");
            return 1;
        }
        uint64_t vgguf_data_off = (uint64_t)gguf_get_data_offset(vctx);

        FILE * fsrc = fopen(in_path,  "rb");
        FILE * fdst = fopen(out_path, "rb");
        if (!fsrc || !fdst) {
            fprintf(stderr, "gguf-to-twzm: verify: cannot re-open files\n");
            if (fsrc) fclose(fsrc);
            if (fdst) fclose(fdst);
            gguf_free(vctx);
            return 1;
        }

        const size_t PROBE = 64; // bytes to compare per tensor
        std::vector<uint8_t> src_buf(PROBE), dst_buf(PROBE);

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
            if (fseeko(fdst, (off_t)dst_off, SEEK_SET) != 0 ||
                fread(dst_buf.data(), 1, probe, fdst) != probe) {
                fprintf(stderr, "gguf-to-twzm: verify: cannot read TWZM for '%s'\n", name);
                mismatches++;
                continue;
            }
            if (memcmp(src_buf.data(), dst_buf.data(), probe) != 0) {
                fprintf(stderr, "gguf-to-twzm: verify: MISMATCH for '%s' at GGUF off=%" PRIu64 " TWZM off=%" PRIu64 "\n",
                        name, src_off, dst_off);
                fprintf(stderr, "  gguf: ");
                for (size_t b = 0; b < probe; ++b) fprintf(stderr, "%02x", src_buf[b]);
                fprintf(stderr, "\n  twzm: ");
                for (size_t b = 0; b < probe; ++b) fprintf(stderr, "%02x", dst_buf[b]);
                fprintf(stderr, "\n");
                mismatches++;
            }
        }

        fclose(fsrc);
        fclose(fdst);
        gguf_free(vctx);

        if (mismatches == 0) {
            fprintf(stderr, "gguf-to-twzm: verify: all %" PRId64 " tensors OK\n", n_tensors);
        } else {
            fprintf(stderr, "gguf-to-twzm: verify: %d tensor(s) FAILED\n", mismatches);
            return 1;
        }
    }

    return 0;

fail:
    fclose(fin);
    fclose(fout);
    remove(out_path);
    gguf_free(gguf_ctx);
    return 1;
}
