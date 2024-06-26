#pragma once

#include <vector>
#include <string>
#include <cstring>

#include <zeno/utils/format.h>
#include <zeno/utils/fileio.h>

namespace zeno {

static void write_pfm(const char* path, int w, int h, const float *pixel, bool mono=false) {
    std::string header = zeno::format("PF\n{} {}\n-1.0\n", w, h);
    char channel = 3;

    if (mono) {
        header = zeno::format("Pf\n{} {}\n-1.0\n", w, h);
        channel = 1;
    }

    std::vector<char> data(header.size() + w * h * sizeof(float) * channel);
    memcpy(data.data(), header.data(), header.size());
    memcpy(data.data() + header.size(), pixel, w * h * sizeof(float) * channel);
    zeno::file_put_binary(data, path);
}

}