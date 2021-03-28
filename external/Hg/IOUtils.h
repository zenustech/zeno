#pragma once

#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdio>

namespace hg {

static std::string file_get_content(std::string const &path) {
    std::ifstream fin(path);
    std::string content;
    std::istreambuf_iterator<char> iit(fin), eiit;
    std::back_insert_iterator<std::string> sit(content);
    std::copy(iit, eiit, sit);
    return content;
}

static void *load_raw_file(char const *filename, size_t size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror(filename);
        return NULL;
    }
    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);
    return data;
}

}
