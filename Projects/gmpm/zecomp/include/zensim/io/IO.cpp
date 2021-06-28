#include "IO.h"

#include <filesystem>
// #include <compare>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace zs {

  std::string file_get_content(std::string const &path) {
    std::ifstream fin(path);
    std::string content;
    std::istreambuf_iterator<char> iit(fin), eiit;
    std::back_insert_iterator<std::string> sit(content);
    std::copy(iit, eiit, sit);
    return content;
  }

  void *load_raw_file(char const *filename, size_t size) {
    FILE *fp = fopen(filename, "rb");

    if (!fp) {
      fprintf(stderr, "Error opening file '%s'\n", filename);
      return 0;
    }
    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif
    return data;
  }

}  // namespace zs
