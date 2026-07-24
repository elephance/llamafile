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

    fprintf(stderr, "gguf-to-twzm: %s\n", in_path);
    fprintf(stderr, "  tensors:         %" PRId64 "\n", n_tensors);
    fprintf(stderr, "  metadata blob:   %" PRIu64 " bytes\n", gguf_data_offset);

    // -----------------------------------------------------------------------
    // 2. Compute the TWZM file layout.
    // -----------------------------------------------------------------------
    const uint64_t header_size       = (uint64_t)sizeof(TwzmHeader);
    const uint64_t entry_size        = (uint64_t)sizeof(TwzmTensorEntry);
    const uint64_t tensor_index_off  = header_size;
    const uint64_t tensor_index_size = (uint64_t)n_tensors * entry_size;
    const uint64_t metadata_off      = tensor_index_off + tensor_index_size;
    const uint64_t metadata_size     = gguf_data_offset;

    // First tensor data starts at the next page boundary after the metadata.
    uint64_t next_data_off = align_up(metadata_off + metadata_size, TWZM_DATA_ALIGNMENT);

    // Build the tensor index, computing output offsets.
    std::vector<TwzmTensorEntry> index((size_t)n_tensors);
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

        next_data_off  = align_up(next_data_off + (uint64_t)tensor_size, TWZM_DATA_ALIGNMENT);
        total_tensor_bytes += (uint64_t)tensor_size;
    }

    uint64_t total_file_size = next_data_off; // last used page-aligned boundary

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
    hdr.tensor_count        = (uint64_t)n_tensors;

    if (fwrite(&hdr, sizeof(hdr), 1, fout) != 1) {
        fprintf(stderr, "gguf-to-twzm: failed to write header\n");
        goto fail;
    }

    // -----------------------------------------------------------------------
    // 5. Write tensor index.
    // -----------------------------------------------------------------------
    for (int64_t i = 0; i < n_tensors; ++i) {
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
    // -----------------------------------------------------------------------
    for (int64_t i = 0; i < n_tensors; ++i) {
        // Seek output to the computed page-aligned offset.
        if (fseeko(fout, (off_t)index[i].data_offset, SEEK_SET) != 0) {
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
    // 8. Optional verification: re-open both files and spot-check each tensor.
    // -----------------------------------------------------------------------
    if (verify) {
        fprintf(stderr, "gguf-to-twzm: verifying ...\n");

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
        int mismatches = 0;

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name(vctx, i);
            uint64_t src_off    = vgguf_data_off + (uint64_t)gguf_get_tensor_offset(vctx, i);
            uint64_t dst_off    = index[i].data_offset;
            uint64_t tsize      = (uint64_t)gguf_get_tensor_size(vctx, i);
            size_t   probe      = (size_t)(tsize < PROBE ? tsize : PROBE);

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
