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
