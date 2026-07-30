// Offline generator for the expanded unicode codepoint-flags table.
// Replicates unicode.cpp's unicode_cpt_flags_array() exactly, then emits the
// resulting bytes as a C++ source file.
#include "unicode.h"
#include "unicode-data.h"

#include <cassert>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    // --- verbatim copy of unicode.cpp:unicode_cpt_flags_array() ---
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];
        const auto range_end = unicode_ranges_flags.begin()[i];
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }
    for (auto cpt : unicode_set_whitespace)   { cpt_flags[cpt].is_whitespace = true; }
    for (auto p : unicode_map_lowercase)      { cpt_flags[p.second].is_lowercase = true; }
    for (auto p : unicode_map_uppercase)      { cpt_flags[p.second].is_uppercase = true; }
    for (auto & range : unicode_ranges_nfd)   { cpt_flags[range.nfd].is_nfd = true; }
    // --- end verbatim copy ---

    static_assert(sizeof(unicode_cpt_flags) == 2, "expected a 2-byte flags word");

    FILE * f = fopen(argv[1], "wb");
    if (!f) { perror("fopen"); return 1; }

    fprintf(f,
        "// GENERATED FILE - DO NOT EDIT.\n"
        "//\n"
        "// The expanded per-codepoint Unicode flags table, precomputed from the\n"
        "// range/set/map tables in unicode-data.cpp. unicode.cpp used to build this\n"
        "// at first use, which cost ~3ms and 2.2MB of heap on every process that\n"
        "// tokenized anything (1,114,112 scalar stores plus four sparse passes).\n"
        "// Emitting it as .rodata makes that cost zero - only the handful of pages\n"
        "// actually indexed get demand-paged in.\n"
        "//\n"
        "// Regenerate with llamafile/gen_unicode_flags.cpp whenever unicode-data.cpp\n"
        "// changes; llamafile/unicode_flags_test.cpp asserts the two stay in sync.\n"
        "\n"
        "#include <cstdint>\n"
        "#include <cstddef>\n"
        "\n"
        "// %u codepoints x 2 bytes, little-endian. Stored as a byte string rather\n"
        "// than a uint16_t initializer list purely for compile speed/memory.\n"
        "extern const unsigned char llamafile_unicode_cpt_flags_data[];\n"
        "extern const size_t llamafile_unicode_cpt_flags_size;\n"
        "\n"
        "const unsigned char llamafile_unicode_cpt_flags_data[] =\n",
        (unsigned) MAX_CODEPOINTS);

    const unsigned char * raw = reinterpret_cast<const unsigned char *>(cpt_flags.data());
    const size_t nbytes = cpt_flags.size() * sizeof(unicode_cpt_flags);

    // 32 bytes per source line keeps lines ~128 chars.
    for (size_t i = 0; i < nbytes; i += 32) {
        fputc('"', f);
        for (size_t j = i; j < i + 32 && j < nbytes; ++j) {
            fprintf(f, "\\x%02x", raw[j]);
        }
        fputs("\"\n", f);
    }
    fprintf(f,
        ";\n"
        "\n"
        "const size_t llamafile_unicode_cpt_flags_size = %zu;\n",
        nbytes);
    fclose(f);
    fprintf(stderr, "wrote %s (%zu data bytes)\n", argv[1], nbytes);
    return 0;
}
