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
// Platform behaviour:
//   TWIZZLER build: calls the native Twizzler object API.
//   Linux build   : opens a .twzm file under $TWZ_OBJECT_PATH (default: ".")
//                   with the name "<hi>_<lo>.twzm" and mmap's it read-only.
void * twz_object_map(twz_objid id, size_t * out_size);

// Unmap a previously mapped Twizzler object.
// `base` and `size` must match the values returned / set by twz_object_map().
void twz_object_unmap(void * base, size_t size);

#ifdef __cplusplus
} // extern "C"
#endif
