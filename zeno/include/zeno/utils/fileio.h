#pragma once

#include <string>
#include <fstream>
#include <cstdio>

namespace zeno {

static std::string file_get_content(std::string const &path) {
  std::ifstream fin(path);
  std::string content;
  std::istreambuf_iterator<char> iit(fin), eiit;
  std::back_insert_iterator<std::string> sit(content);
  std::copy(iit, eiit, sit);
  return content;
}

static void file_put_content(
        std::string const &path, std::string const &content) {
  std::ofstream fout(path);
  fout << content;
}

static bool file_exists(std::string const &path) {
  std::ifstream fin(path);
  return (bool)fin;
}

static std::vector<char> c_load_file_to_vector(char const *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    perror(filename);
    return {};
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    perror(filename);
    fclose(fp);
    return {};
  }
  long size = ftell(fp);
  if (size == -1) {
    perror(filename);
    fclose(fp);
    return {};
  }
  rewind(fp);
  std::vector<char> res;
  res.resize(size);
  size_t n = fread(res.data(), 1, res.size(), fp);
  if (n != res.size()) {
    perror(filename);
  }
  fclose(fp);
  return res;
}

}
