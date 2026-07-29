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
// Implements the twz_object_* API (declared in twizzler_platform.h) for
// development and testing on Linux without Twizzler OS: both read-only
// (copy-on-write) mapping of existing objects, and creation/growth of
// writable objects for the format converter.
//
// Object resolution:
//   The object file is looked up as:
//       ${TWZ_OBJECT_PATH:-./twzm_objects}/<hi_hex>_<lo_hex>.twzm
//   where <hi_hex> and <lo_hex> are the zero-padded 16-character hex
//   representations of twz_objid.hi and twz_objid.lo respectively.

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

// ---------------------------------------------------------------------------
// Read-only (copy-on-write) mapping cache, keyed by resolved path.
//
// Keeping the same mapping alive lets llama_model_load_from_twzm() find the
// cached_model_ptr it wrote into the header on first load. Shared by both
// the objid-based (twz_object_map) and direct-path (twz_object_map_at_path)
// entry points, so there is exactly one cache instead of one per caller.
// ---------------------------------------------------------------------------

struct TwzmMapping { void * base; size_t size; };
static std::unordered_map<std::string, TwzmMapping> g_twzm_mappings;

void * twz_object_map_at_path(const char * path, size_t * out_size) {
    *out_size = 0;

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

void * twz_object_map(twz_objid id, size_t * out_size) {
    *out_size = 0;

    char path[4096];
    if (!twz_object_path(id, path, sizeof(path))) {
        fprintf(stderr, "twz_object_map: path too long for object %016" PRIx64
                "_%016" PRIx64 "\n", id.hi, id.lo);
        return NULL;
    }

    return twz_object_map_at_path(path, out_size);
}

void twz_object_unmap(void * base, size_t size) {
    if (!base || !size) {
        return;
    }

    // Evict from the read-only mapping cache, if present, so a subsequent
    // twz_object_map()/twz_object_map_at_path() call for the same path
    // doesn't hand back a now-dangling pointer.
    for (auto it = g_twzm_mappings.begin(); it != g_twzm_mappings.end(); ++it) {
        if (it->second.base == base) {
            g_twzm_mappings.erase(it);
            break;
        }
    }

    munmap(base, size);
}

// ---------------------------------------------------------------------------
// Writable object creation (for the format converter / any writer).
//
// Separate from the read-only cache above: writable mappings are one-shot
// (created, written, finalized) and are never looked up by path. But
// twz_object_resize()/twz_object_finalize() only receive a base pointer, so
// we track which fd/size backs each currently-active writable mapping.
// ---------------------------------------------------------------------------

struct TwzmWritableMapping { int fd; size_t size; };
static std::unordered_map<void *, TwzmWritableMapping> g_writable_mappings;

void * twz_object_create_at_path(const char * path, size_t size) {
    if (size == 0) {
        fprintf(stderr, "twz_object_create: cannot create a zero-size object ('%s')\n", path);
        return NULL;
    }

    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        fprintf(stderr, "twz_object_create: cannot create '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }

    // Zero-fills the file up to `size` (sparse on most filesystems).
    if (ftruncate(fd, (off_t)size) != 0) {
        fprintf(stderr, "twz_object_create: ftruncate '%s' to %zu bytes failed: %s\n",
                path, size, strerror(errno));
        close(fd);
        return NULL;
    }

    // MAP_SHARED + PROT_WRITE: writes ARE persisted to backing storage,
    // unlike the copy-on-write mapping from twz_object_map()/_at_path().
    void * ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "twz_object_create: mmap '%s' (%zu bytes) failed: %s\n",
                path, size, strerror(errno));
        close(fd);
        return NULL;
    }

    g_writable_mappings[ptr] = TwzmWritableMapping{fd, size};
    return ptr;
}

void * twz_object_create(twz_objid id, size_t size) {
    char path[4096];
    if (!twz_object_path(id, path, sizeof(path))) {
        fprintf(stderr, "twz_object_create: path too long for object %016" PRIx64
                "_%016" PRIx64 "\n", id.hi, id.lo);
        return NULL;
    }

    return twz_object_create_at_path(path, size);
}

void * twz_object_resize(void * base, size_t old_size, size_t new_size) {
    auto it = g_writable_mappings.find(base);
    if (it == g_writable_mappings.end()) {
        fprintf(stderr, "twz_object_resize: %p is not a live writable mapping\n", base);
        return NULL;
    }
    if (it->second.size != old_size) {
        fprintf(stderr, "twz_object_resize: old_size %zu does not match tracked size %zu\n",
                old_size, it->second.size);
        return NULL;
    }

    int fd = it->second.fd;
    if (ftruncate(fd, (off_t)new_size) != 0) {
        fprintf(stderr, "twz_object_resize: ftruncate to %zu bytes failed: %s\n",
                new_size, strerror(errno));
        return NULL;
    }

    void * new_base = mremap(base, old_size, new_size, MREMAP_MAYMOVE);
    if (new_base == MAP_FAILED) {
        fprintf(stderr, "twz_object_resize: mremap to %zu bytes failed: %s\n",
                new_size, strerror(errno));
        return NULL;
    }

    g_writable_mappings.erase(it);
    g_writable_mappings[new_base] = TwzmWritableMapping{fd, new_size};
    return new_base;
}

void twz_object_finalize(void * base, size_t size) {
    auto it = g_writable_mappings.find(base);
    if (it == g_writable_mappings.end()) {
        fprintf(stderr, "twz_object_finalize: %p is not a live writable mapping\n", base);
        return;
    }
    if (it->second.size != size) {
        fprintf(stderr, "twz_object_finalize: size %zu does not match tracked size %zu\n",
                size, it->second.size);
    }

    int fd = it->second.fd;
    msync(base, size, MS_SYNC);
    munmap(base, size);
    close(fd);
    g_writable_mappings.erase(it);
}

#endif // !TWIZZLER
