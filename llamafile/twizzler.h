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
// The TWZM (Twizzler Model) format is a self-contained binary object that
// stores a llama.cpp model and can be directly memory-mapped.  The format
// embeds:
//   - A fixed 64-byte header (magic, version, field offsets).
//   - A tensor index (name → in-object offset lookup table).
//   - A GGUF metadata blob (all KV pairs + tensor info headers, no data).
//   - Page-aligned tensor data regions (zero-copy mappable).
//
// On Twizzler OS the object is identified by a 128-bit ObjID and mapped
// via native Twizzler syscalls.  On Linux a file-backed shim is provided
// for development and testing (see TWZ_OBJECT_PATH).
//
// Loading flow:
//   1. twz_object_map()         – map the object into address space.
//   2. gguf_init_from_buffer()  – parse embedded GGUF metadata blob.
//   3. llama_model_init_from_user() – build llama_model from metadata +
//        set_tensor_data_cb that sets tensor->data pointers into the map.
//   4. twz_object_unmap()       – called by TwzmModel destructor.

#pragma once

// Use the relative path to reach the real llama API header.
// Cannot use #include "llama.h" here because llamafile/llama.h (a small
// helpers-only file) shadows llama.cpp/include/llama.h in the include search.
#include "../llama.cpp/include/llama.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Twizzler object identity
// ---------------------------------------------------------------------------

// A Twizzler Object ID is a 128-bit value.  On Linux the hi/lo pair encodes
// an opaque handle; twz_object_map() resolves it to a file path under the
// directory named by the TWZ_OBJECT_PATH environment variable (default: ".").
typedef struct {
    uint64_t hi;
    uint64_t lo;
} twz_objid;

// ---------------------------------------------------------------------------
// On-disk / in-object format structs
//
// All multi-byte integers are stored little-endian (native on x86/ARM).
// ---------------------------------------------------------------------------

#define TWZM_MAGIC   0x4D5A5754u  // "TWZM" (little-endian)
#define TWZM_VERSION 1u

#define TWZM_TENSOR_NAME_MAX 96   // maximum tensor name length incl. NUL

// Fixed-size object header at byte 0.
typedef struct __attribute__((packed)) {
    uint32_t magic;               // Must equal TWZM_MAGIC.
    uint32_t version;             // Must equal TWZM_VERSION.
    uint64_t metadata_offset;     // Byte offset to the GGUF metadata blob.
    uint64_t metadata_size;       // Byte length of the GGUF metadata blob.
    uint64_t tensor_index_offset; // Byte offset to the tensor index array.
    uint64_t tensor_count;        // Number of entries in the tensor index.
    // Process-local model pointer cache.  Written at runtime via a COW page
    // (MAP_PRIVATE, so the on-disk file is never modified).  On Twizzler this
    // would be a persistent pointer fixed up on remap.  Zero means uncached.
    uint64_t cached_model_ptr;
    uint64_t vocab_offset;        // Byte offset to flat vocab section (0 = absent).
    uint64_t vocab_size;          // Byte length of flat vocab section.
} TwzmHeader;

// One entry in the tensor index.
typedef struct __attribute__((packed)) {
    char     name[TWZM_TENSOR_NAME_MAX]; // Tensor name (null-terminated).
    uint64_t data_offset;                // Byte offset within the object.
    uint64_t data_size;                  // Byte length of the tensor data.
} TwzmTensorEntry;

// Alignment for tensor data regions (must be a power of two).
#define TWZM_DATA_ALIGNMENT 4096u

// ---------------------------------------------------------------------------
// Flat vocab section format (at vocab_offset in the TWZM object)
//
// The vocab section lets the loader skip gguf_init_from_buffer and load_vocab
// entirely on first load, avoiding 150ms+ of hash-table construction.
//
// Layout:
//   TwzmVocabHeader  (60 bytes, fixed)
//   id_data[]        (n_vocab × 12 bytes)  score+attr+text_offset per token
//   merges[]         (n_merges × 4 bytes)  text_offset of "p1 p2\0" merge string
//   text_pool        (text_pool_size bytes) all strings, null-terminated
//   token_hash[]     (token_hash_capacity × 12 bytes) open-addressing hash table
//                     for O(1) text→token_id lookup.  Empty slots have
//                     token_id = -1.  Sized to keep load factor < 75%.
//   merge_hash[]     (merge_hash_capacity × 12 bytes) open-addressing hash table
//                     for O(1) "left right"→rank lookup (find_bpe_rank), same
//                     entry layout as token_hash[] with token_id repurposed to
//                     hold the merge rank.  Keyed by the exact "p1 p2" merge
//                     text (as stored in merges[]/text_pool), hashed the same
//                     way as token text.  Empty slots have token_id = -1.
//   piece_data[]     (n_vocab × 8 bytes) TwzmPieceEntry per token: the
//                     precomputed llama_token_to_piece(id, special=true)
//                     result, computed once at conversion time (see
//                     gguf_to_twzm.cpp) instead of at every load.
//   piece_pool       (piece_pool_size bytes) concatenated piece bytes,
//                     addressed by piece_data[]. NOT null-terminated -
//                     byte-fallback tokens can decode to any byte incl. 0x00.
// ---------------------------------------------------------------------------

#define TWZM_VOCAB_MAGIC 0x4D435657u  // "WVCM"

typedef struct __attribute__((packed)) {
    uint32_t magic;                // TWZM_VOCAB_MAGIC
    uint32_t n_vocab;
    uint32_t vocab_type;           // llama_vocab_type enum
    uint32_t n_merges;
    uint32_t text_pool_size;
    uint32_t token_hash_offset;    // byte offset from start of vocab section to hash table (0 = absent)
    uint32_t token_hash_capacity;  // number of slots in the hash table (power of two)
    uint32_t token_hash_count;     // number of occupied slots
    uint32_t merge_hash_offset;    // byte offset from start of vocab section to merge-rank hash table (0 = absent)
    uint32_t merge_hash_capacity;  // number of slots in the merge-rank hash table (power of two)
    uint32_t merge_hash_count;     // number of occupied slots (== n_merges when present)
    uint32_t piece_data_offset;    // byte offset from start of vocab section to piece_data[] (0 = absent)
    uint32_t piece_pool_offset;    // byte offset from start of vocab section to piece_pool
    uint32_t piece_pool_size;      // byte size of piece_pool
    uint32_t max_token_len;        // longest token text, in bytes (precomputed, avoids a strlen() scan at load)
} TwzmVocabHeader;

typedef struct __attribute__((packed)) {
    float    score;
    int32_t  attr;            // llama_token_attr flags
    uint32_t text_offset;     // byte offset into text_pool
} TwzmTokenData;

// One entry in the open-addressing token hash table (also reused, as-is, for
// the merge-rank hash table - see TwzmVocabHeader.merge_hash_offset).
// Empty slots have token_id = -1 (LLAMA_TOKEN_NULL).
typedef struct __attribute__((packed)) {
    uint32_t hash_value;      // FNV-1a hash of the token (or "left right" merge) text
    int32_t  token_id;        // llama_token, or merge rank in merge_hash[] (or -1 for empty)
    uint32_t text_offset;     // byte offset into text_pool
} TwzmTokenHashEntry;

// One entry in piece_data[] - see TwzmVocabHeader.piece_data_offset.
typedef struct __attribute__((packed)) {
    uint32_t piece_offset;    // byte offset into piece_pool
    uint32_t piece_length;    // length in bytes (NOT null-terminated)
} TwzmPieceEntry;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Load a llama_model from a Twizzler memory object identified by `id`.
//
// On Twizzler OS, `id` is mapped via the native object API.
// On Linux, `id` is resolved to a file path and mmap'd (see twizzler_platform.h).
//
// The returned model must be freed with llama_model_free().  The caller must
// NOT unmap the underlying object while the model is live; ownership of the
// mapping is transferred to the returned model (via a custom deleter).
//
// Returns NULL on failure (error is logged to stderr).
struct llama_model * llama_model_load_from_twizzler_object(
    twz_objid id,
    struct llama_model_params params);

// Load directly from a .twzm file path on the local filesystem.
// The file is mmap'd read-only; the mapping is retained for the model's
// lifetime (released on process exit).  Callers needing explicit lifetime
// control should use llama_model_load_from_twzm() with a caller-managed map.
// Returns NULL on failure (error is logged to stderr).
struct llama_model * llama_model_load_from_twzm_path(
    const char * path,
    struct llama_model_params params);

// Lower-level variant that operates on an already-mapped region.
// `base` must point to the start of a mapped TWZM object of `size` bytes.
// The caller retains ownership of the mapping; it must remain valid for the
// lifetime of the returned model.
struct llama_model * llama_model_load_from_twzm(
    const void * base,
    size_t size,
    struct llama_model_params params);

#ifdef __cplusplus
} // extern "C"
#endif
