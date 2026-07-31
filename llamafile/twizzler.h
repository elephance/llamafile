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
// The TWZM (Twizzler Model) format stores a llama.cpp model across THREE
// Twizzler objects, each directly memory-mappable:
//   - Root/metadata object: a fixed header (magic, version, field offsets),
//     a tensor index (name → in-tensor-data-object offset lookup table),
//     and a GGUF metadata blob (all KV pairs + tensor info headers, no data).
//   - Vocab object (optional): the flat vocab section (tokens, hash tables,
//     merges, detokenize piece cache) - see TwzmVocabHeader below.
//   - Tensor-data object: all raw tensor bytes, at page-aligned offsets
//     (zero-copy mappable).
//
// The root object refers to the other two via a TwzmGlobalPtr - a 128-bit
// Twizzler object id plus a byte offset within that object - embedded in
// TwzmHeader. Callers address the root directly (by id or, on the Linux dev
// shim, by file path); everything else is reached by following the
// pointers embedded in it, same as a real Twizzler-OS deployment would
// resolve them.
//
// On Twizzler OS each object is identified by a 128-bit ObjID and mapped
// via native Twizzler syscalls.  On Linux a file-backed shim is provided
// for development and testing - see TWZ_OBJECT_PATH (twizzler_platform.h).
// Note that loading the root by direct file path (llama_model_load_from_twzm_path)
// still requires TWZ_OBJECT_PATH to be set consistently with conversion
// time, since the root's embedded pointers are always resolved by id.
//
// Loading flow:
//   1. twz_object_map()/twz_object_map_at_path() – map the root object.
//   2. gguf_init_from_buffer()  – parse embedded GGUF metadata blob.
//   3. twz_object_map() the tensor-data object (required) and, if present,
//      the vocab object - both referenced from the root header.
//   4. llama_model_init_from_user() – build llama_model from metadata +
//        set_tensor_data_cb that sets tensor->data pointers into the
//        tensor-data mapping, and (if present) the vocab mapping for
//        fast vocab loading.
//   5. All three mappings are kept alive for the model's lifetime (no
//      model-destruction hook exists to unmap them automatically).

#pragma once

// Use the relative path to reach the real llama API header.
// Cannot use #include "llama.h" here because llamafile/llama.h (a small
// helpers-only file) shadows llama.cpp/include/llama.h in the include search.
#include "../llama.cpp/include/llama.h"

#include <stdbool.h>
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
// v3: TwzmTensorEntry gained type/n_dims/ne, which lets the metadata blob drop
// its tensor infos. v2 objects are not readable by a v3 loader (their index
// entries are a different size and their blob is laid out differently) - they
// must be regenerated with gguf-to-twzm.
#define TWZM_VERSION 3u

#define TWZM_TENSOR_NAME_MAX 96   // maximum tensor name length incl. NUL

// A reference to a byte range inside another Twizzler object: an id plus a
// byte offset within that object. {0,0} id means "absent/unset". offset is
// always 0 today (each referenced object below is dedicated to exactly one
// section) but is a real field - not assumed zero - so a future phase that
// co-locates sections in one object doesn't need another format bump.
typedef struct __attribute__((packed)) {
    twz_objid id;
    uint64_t  offset;
} TwzmGlobalPtr;

// Fixed-size root/metadata object header at byte 0.
typedef struct __attribute__((packed)) {
    uint32_t magic;               // Must equal TWZM_MAGIC.
    uint32_t version;             // Must equal TWZM_VERSION.
    uint64_t metadata_offset;     // Byte offset (within THIS object) to the GGUF metadata blob.
    uint64_t metadata_size;       // Byte length of the GGUF metadata blob.
    uint64_t tensor_index_offset; // Byte offset (within THIS object) to the tensor index array.
    uint64_t tensor_count;        // Number of entries in the tensor index.
    // Process-local model pointer cache.  Written at runtime via a COW page
    // (MAP_PRIVATE, so the on-disk file is never modified).  On Twizzler this
    // would be a persistent pointer fixed up on remap.  Zero means uncached.
    uint64_t cached_model_ptr;
    TwzmGlobalPtr vocab;          // Flat vocab section object (absent = no fast-load vocab).
    TwzmGlobalPtr tensor_data;    // Tensor data object (required).
} TwzmHeader;

// One entry in the tensor index. Sorted by name, so twzm_lookup() can bsearch.
//
// This is the authoritative tensor table: since v3 it carries `type` and `ne`
// as well as the data location, which makes it self-sufficient and lets the
// converter strip tensor infos out of the embedded GGUF metadata blob (the
// same trick already used for the tokenizer arrays). That matters because
// gguf_init_from_buffer() spent ~95% of its time constructing one
// gguf_tensor_info per tensor - 0.52ms of 0.55ms on a 579-tensor MoE - while
// the 35 KV pairs it actually still needs parse in 0.03ms. With the infos
// gone, that stage stops scaling with tensor count entirely.
typedef struct __attribute__((packed)) {
    char     name[TWZM_TENSOR_NAME_MAX]; // Tensor name (null-terminated).
    uint64_t data_offset;                // Byte offset within the tensor-data object.
    uint64_t data_size;                  // Byte length of the tensor data.
    uint32_t type;                       // ggml_type of the tensor data.
    uint32_t n_dims;                     // Number of significant entries in ne[].
    int64_t  ne[4];                      // Dimensions; unused trailing entries are 1.
} TwzmTensorEntry;

// Alignment for tensor data regions (must be a power of two).
#define TWZM_DATA_ALIGNMENT 4096u

// ---------------------------------------------------------------------------
// Flat vocab section format (the entire content of the vocab object,
// referenced from TwzmHeader.vocab)
//
// The vocab section lets the loader skip gguf_init_from_buffer and load_vocab
// entirely on first load, avoiding 150ms+ of hash-table construction. All
// offsets below are relative to the start of the vocab section/object
// itself (i.e. relative to TwzmHeader.vocab.offset, not the root object).
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
// KV cache object format (the entire content of a KV cache object)
//
// Prompt evaluation is pure recomputation: without a cache every run decodes
// the whole prompt from n_past = 0, at roughly 15ms per token with no fixed
// cost (measured: ~40 tokens = 178ms, ~370 tokens = 5542ms on a 1B model).
// Persisting the KV state turns that into a memcpy of about 33KB per token.
//
// Unlike the vocab and tensor-data objects, a KV cache object is NOT
// referenced from TwzmHeader: the root object is mapped MAP_PRIVATE, so a
// TwzmGlobalPtr written into it at runtime would be copy-on-write and would
// not survive the process. Instead the id is *derived* (see
// twzm_kv_cache_id()) and resolved through the ordinary TWZ_OBJECT_PATH
// convention, which needs no root mutation and no format version bump.
//
// Layout:
//   TwzmKvHeader   (fixed)
//   tokens[]       (n_tokens x 4 bytes)  llama_token, the prompt this covers
//   blob           (blob_size bytes)     llama_state_seq_get_data() output
// ---------------------------------------------------------------------------

#define TWZM_KV_MAGIC   0x4D434B57u  // "WKCM"
#define TWZM_KV_VERSION 1u

typedef struct __attribute__((packed)) {
    uint32_t  magic;          // TWZM_KV_MAGIC
    uint32_t  version;        // TWZM_KV_VERSION
    twz_objid model_id;       // must equal the root's TwzmHeader.tensor_data.id
    uint32_t  n_ctx;          // context size the state was captured at
    uint32_t  type_k;         // ggml_type of the K cache
    uint32_t  type_v;         // ggml_type of the V cache
    uint32_t  state_version;  // LLAMA_STATE_SEQ_VERSION at capture time
    uint32_t  n_tokens;       // number of tokens in tokens[]
    uint32_t  reserved;       // pad to 8-byte alignment; must be 0
    uint64_t  tokens_offset;  // byte offset to tokens[]
    uint64_t  blob_offset;    // byte offset to the state blob
    uint64_t  blob_size;      // used bytes of the state blob
} TwzmKvHeader;

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
// The root file is mmap'd read-only by path; its tensor-data object (and
// vocab object, if present) are then resolved and mapped by id via
// TWZ_OBJECT_PATH (see twizzler_platform.h) - it must be set consistently
// with however `gguf-to-twzm` was run to produce this file. All mappings
// are retained for the model's lifetime (released on process exit).
// Callers needing explicit lifetime control should use
// llama_model_load_from_twzm() with a caller-managed map.
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

// ---------------------------------------------------------------------------
// KV cache (see TwzmKvHeader above)
// ---------------------------------------------------------------------------

// Derive the KV cache object id for a model. Purely a function of its input,
// so any process can find the same cache without a registry or a pointer
// stored in the root object.
//
// `model_id` should be the root's TwzmHeader.tensor_data.id, which is a random
// 128-bit id minted per conversion and therefore a precise model identity: a
// reconverted model gets a different id and so cannot collide with a stale
// cache.
//
// Deliberately NOT keyed on n_ctx/type_k/type_v, even though a state blob is
// only valid for the geometry it was captured at: the id must be computable
// before the sandbox, and n_ctx is not final until the model has loaded (it is
// clamped to the training context afterwards). Geometry is recorded in
// TwzmKvHeader and checked on read instead, so a changed context size is a
// clean miss that the next save overwrites. One cache per model, therefore;
// alternating between two context sizes will thrash it.
twz_objid twzm_kv_cache_id(twz_objid model_id);

// Read the tensor_data object id out of a mapped root object, which is what
// twzm_kv_cache_id() wants for `model_id`. Returns {0,0} if `base`/`size` is
// not a valid TWZM root.
twz_objid twzm_root_model_id(const void * base, size_t size);

// Derive a model identity for a plain file (a .gguf), so the KV cache works
// without the TWZM format - prompt caching and TWZM are independent wins and
// should not be coupled.
//
// Fingerprints the file's size together with its leading bytes rather than
// hashing it whole: a GGUF's header and KV metadata sit at the front and
// already encode the architecture, every tensor's name, shape and type, so two
// different models effectively always differ there, while a 1.3GB full hash
// would cost more than the prompt evaluation being avoided. The trade-off is
// that a model edited only in its weight bytes, keeping the same size and
// metadata, would not be detected - which is why the geometry and llama state
// version are still re-validated from TwzmKvHeader on every read.
//
// Opens the file, so it must be called before any privilege drop that removes
// open(). Returns {0,0} if the file cannot be read.
twz_objid twzm_file_model_id(const char * path);

// An open KV cache. Acquired before a privilege drop, used after it.
struct TwzmKvCache;

// Acquire write access to the KV cache object for `cache_id` and map any
// existing contents. MUST be called before llamafile_sandbox_enter(), since
// it is the only step that needs open(). Never fails in a way that should
// abort a run: on any error it returns a handle that simply reports no cached
// state, and saving is skipped. `read_only` skips acquiring write access.
struct TwzmKvCache * twzm_kv_cache_open(twz_objid cache_id, bool read_only);

// Tokens covered by the cached state, or NULL when there is no usable cache.
// Validates magic/version/model_id/geometry against the arguments; a mismatch
// reports "no cache" rather than an error, so a stale object degrades to a
// cold run. *out_n_tokens is set to 0 when NULL is returned.
const int32_t * twzm_kv_cache_tokens(struct TwzmKvCache * kv,
                                     twz_objid model_id, uint32_t n_ctx,
                                     uint32_t type_k, uint32_t type_v,
                                     uint32_t * out_n_tokens);

// The cached state blob matching twzm_kv_cache_tokens(). NULL if absent.
const void * twzm_kv_cache_blob(struct TwzmKvCache * kv, uint64_t * out_size);

// Replace the cache contents. Grows the object as needed - safe after a
// privilege drop, since the descriptor was acquired by twzm_kv_cache_open().
// Returns false (and logs) on failure; callers should treat that as
// non-fatal. No-op on a read-only or failed handle.
bool twzm_kv_cache_save(struct TwzmKvCache * kv, twz_objid model_id,
                        uint32_t n_ctx, uint32_t type_k, uint32_t type_v,
                        const int32_t * tokens, uint32_t n_tokens,
                        const void * blob, uint64_t blob_size);

// Release the handle and its mappings.
void twzm_kv_cache_close(struct TwzmKvCache * kv);

// TWZM_DEBUG level, for callers that want to trace cache decisions alongside
// this file's own logging. >=1 traces hits/misses; >=2 additionally enables
// the caller-side restore round-trip check in chatbot_cli.cpp.
int twzm_kv_debug_level(void);

#ifdef __cplusplus
} // extern "C"
#endif
