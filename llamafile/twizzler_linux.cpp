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

// Linux shim for Twizzler memory-object mapping.
//
// Implements twz_object_map() / twz_object_unmap() for development and
// testing on Linux without Twizzler OS.
//
// Object resolution:
//   The object file is looked up as:
//       ${TWZ_OBJECT_PATH:-./twzm_objects}/<hi_hex>_<lo_hex>.twzm
//   where <hi_hex> and <lo_hex> are the zero-padded 16-character hex
//   representations of twz_objid.hi and twz_objid.lo respectively.
//
// The file is opened read-only and its entire contents are mmap'd with
// MAP_PRIVATE.  The mapping size is the file size.

#ifndef TWIZZLER // only compile on Linux; real Twizzler gets its own impl

#include "twizzler_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <unordered_map>

// Build the file path for an object ID.
// Returns 1 on success, 0 if the path would overflow buf_size.
static int twz_object_path(twz_objid id, char * buf, size_t buf_size) {
    const char * base = getenv("TWZ_OBJECT_PATH");
    if (!base || base[0] == '\0') {
        base = "./twzm_objects";
    }
    int n = snprintf(buf, buf_size, "%s/%016" PRIx64 "_%016" PRIx64 ".twzm",
                     base, id.hi, id.lo);
    return n > 0 && (size_t)n < buf_size;
}

// Per-process table of open TWZM mappings keyed by file path.
// Keeping the same mapping alive lets llama_model_load_from_twzm() find the
// cached_model_ptr it wrote into the header on first load.
struct TwzmMapping { void * base; size_t size; };
static std::unordered_map<std::string, TwzmMapping> g_twzm_mappings;

void * twz_object_map(twz_objid id, size_t * out_size) {
    *out_size = 0;

    char path[4096];
    if (!twz_object_path(id, path, sizeof(path))) {
        fprintf(stderr, "twz_object_map: path too long for object %016" PRIx64
                "_%016" PRIx64 "\n", id.hi, id.lo);
        return NULL;
    }

    // Re-use an existing mapping for this path if we have one.
    auto it = g_twzm_mappings.find(path);
    if (it != g_twzm_mappings.end()) {
        *out_size = it->second.size;
        return it->second.base;
    }

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "twz_object_map: cannot open '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "twz_object_map: fstat '%s' failed: %s\n",
                path, strerror(errno));
        close(fd);
        return NULL;
    }

    if (st.st_size <= 0) {
        fprintf(stderr, "twz_object_map: '%s' is empty\n", path);
        close(fd);
        return NULL;
    }

    size_t sz = (size_t)st.st_size;
    // MAP_PRIVATE + PROT_WRITE: writes are copy-on-write (never touch the file).
    // This lets us cache llama_model* in the header's reserved field at runtime.
    void * ptr = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) {
        fprintf(stderr, "twz_object_map: mmap '%s' (%zu bytes) failed: %s\n",
                path, sz, strerror(errno));
        return NULL;
    }

    g_twzm_mappings[path] = {ptr, sz};
    *out_size = sz;
    return ptr;
}

void twz_object_unmap(void * base, size_t size) {
    if (base && size) {
        munmap(base, size);
    }
}

#endif // !TWIZZLER
