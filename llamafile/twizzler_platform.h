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

// Platform abstraction for Twizzler memory-object mapping.
//
// Include this header exactly once (from twizzler_linux.cpp or the
// Twizzler-native implementation).  Other translation units should use the
// functions declared here via an extern declaration or through twizzler.h.

#pragma once

#include "twizzler.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Map a Twizzler object identified by `id` into the process address space.
//
// On success, *out_size is set to the total byte length of the mapped region
// and a pointer to the start of the region is returned.
// On failure, NULL is returned and *out_size is set to 0.
//
// The mapping is read-only / copy-on-write: local writes (e.g. the
// cached_model_ptr trick in twizzler.cpp) never reach backing storage. Use
// twz_object_create()/twz_object_resize() below for a mapping whose writes
// are meant to persist.
//
// Platform behaviour:
//   TWIZZLER build: calls the native Twizzler object API.
//   Linux build   : opens a .twzm file under $TWZ_OBJECT_PATH (default: ".")
//                   with the name "<hi>_<lo>.twzm" and mmap's it read-only.
void * twz_object_map(twz_objid id, size_t * out_size);

// Same as twz_object_map(), but for a direct filesystem path rather than an
// objid - used by callers that bypass $TWZ_OBJECT_PATH resolution (e.g.
// llama_model_load_from_twzm_path()). twz_object_map() is implemented in
// terms of this on the Linux build.
void * twz_object_map_at_path(const char * path, size_t * out_size);

// Unmap a previously mapped Twizzler object.
// `base` and `size` must match the values returned / set by twz_object_map()
// or twz_object_map_at_path().
void twz_object_unmap(void * base, size_t size);

// ---------------------------------------------------------------------------
// Writable object creation (for the format converter / any writer).
// ---------------------------------------------------------------------------

// Create a new object of `size` bytes (zero-initialized) and map it
// read-write, such that writes ARE persisted to backing storage (unlike the
// copy-on-write mapping from twz_object_map() above). Returns NULL on
// failure.
void * twz_object_create(twz_objid id, size_t size);

// Same as twz_object_create(), but for a direct filesystem path rather than
// an objid - used by gguf-to-twzm's <output.twzm> CLI argument.
void * twz_object_create_at_path(const char * path, size_t size);

// Grow or shrink a writable mapping obtained from twz_object_create()/
// twz_object_create_at_path() to `new_size` bytes, returning the (possibly
// relocated) base pointer. `base` must not be used again after a successful
// call. Returns NULL on failure, in which case the original mapping at
// `base` is left intact and still valid.
void * twz_object_resize(void * base, size_t old_size, size_t new_size);

// Flush a writable mapping's contents to backing storage and unmap it. Must
// be used (instead of twz_object_unmap()) for mappings obtained from
// twz_object_create()/twz_object_create_at_path()/twz_object_resize().
void twz_object_finalize(void * base, size_t size);

// Generate a fresh, currently-unused 128-bit object id and atomically
// create+map it read-write of `size` bytes (same semantics as
// twz_object_create() otherwise - use twz_object_finalize() when done).
// On success *out_id is set to the generated id. On failure NULL is
// returned and *out_id is zeroed.
//
// Used by writers (e.g. gguf-to-twzm) that need to create a new object
// without already knowing what id to give it - unlike twz_object_create(),
// which requires the caller to supply one (e.g. a root object addressed by
// a fixed CLI output path has no id of its own to pick).
void * twz_object_create_fresh(twz_objid * out_id, size_t size);

// Best-effort delete of an object by id. Ignores "does not exist". Not for
// use on a still-mapped object - twz_object_finalize()/twz_object_unmap()
// it first. Intended for failure-path cleanup of objects created via
// twz_object_create_fresh()/twz_object_create().
void twz_object_destroy(twz_objid id);

// ---------------------------------------------------------------------------
// Deferred-mapping writable handles (for writers that must acquire the right
// to write BEFORE a privilege drop, but only learn the size afterwards).
// ---------------------------------------------------------------------------
//
// The KV cache writer needs this: llamafile's CLI pledges "stdio rpath tty"
// before the model is even loaded, so open() is unavailable by the time the
// cache's size is known (it depends on how many tokens were decoded). Every
// other writable entry point here opens and sizes in one shot, which is too
// early.
//
// Splitting acquire from size lets the caller open the object while it still
// may, then map it later at whatever size it turns out to need. ftruncate(),
// mmap(MAP_SHARED), msync() and mremap() on an already-open descriptor all
// remain permitted under that pledge (verified on Linux/SECCOMP).
//
// An opaque handle to a writable object. On Linux this is a file descriptor;
// on Twizzler it is an object capability handle. TWZ_HANDLE_INVALID is the
// failure value. Do not do arithmetic on it or assume it is an fd.
typedef long twz_handle;
#define TWZ_HANDLE_INVALID ((twz_handle)-1)

// Acquire write access to an object without mapping or resizing it, creating
// an empty one if it does not exist. Existing contents are preserved (unlike
// twz_object_create*, which always truncate). Call before dropping
// privileges. Returns TWZ_HANDLE_INVALID on failure.
twz_handle twz_object_open_rw(twz_objid id);

// Current size in bytes of a handle's object, or -1 on failure. Zero means
// the object was just created (or is empty) and has nothing to read.
long twz_object_handle_size(twz_handle h);

// Resize the handle's object to `size` and map it read-write MAP_SHARED, as
// twz_object_create() would. Usable after a privilege drop that forbids
// open(). The result participates in twz_object_resize()/_finalize() exactly
// like a created mapping; release it with twz_object_finalize().
//
// Consumes the handle: on success ownership passes to the mapping (which
// closes it on finalize), and on failure the handle is closed. Either way the
// caller must not reuse it, nor call twz_object_close() on it.
void * twz_object_handle_map(twz_handle h, size_t size);

// Release a handle that was never passed to twz_object_handle_map().
void twz_object_close(twz_handle h);

#ifdef __cplusplus
} // extern "C"
#endif

// Pointer to `offset` bytes into a mapped object. Trivial today; exists as a
// named seam so future bounds-checking/capability work doesn't have to touch
// call sites. Not extern "C" - a plain inline helper, not part of the ABI.
static inline void * twz_object_ptr(void * base, size_t offset) {
    return (unsigned char *)base + offset;
}

static inline const void * twz_object_cptr(const void * base, size_t offset) {
    return (const unsigned char *)base + offset;
}
