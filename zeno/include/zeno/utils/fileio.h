#pragma once

#include <string>
#include <fstream>
#include <cstdio>
#include <vector>
#include <filesystem>

namespace zeno {

static std::string file_get_content(std::string const &path) {
  std::ifstream fin(path);
  std::string content;
  std::istreambuf_iterator<char> iit(fin), eiit;
  std::back_insert_iterator<std::string> sit(content);
  std::copy(iit, eiit, sit);
  return content;
}

static void file_put_content(std::string const &path, std::string const &content) {
  std::ofstream fout(path);
  fout << content;
}

static bool file_exists(std::string const &path) {
  std::ifstream fin(path);
  return (bool)fin;
}

template <class Arr = std::vector<char>>
static Arr file_get_binary(std::string const &path) {
  std::string native_path = std::filesystem::u8path(path).string();
  char const *filename = native_path.c_str();
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
  Arr res;
  res.resize(size);
  size_t n = fread(res.data(), res.size(), 1, fp);
  if (n != 1) {
    perror(filename);
  }
  fclose(fp);
  return res;
}

static bool file_put_binary(char const *arr_data, size_t arr_size, std::string const &path) {
  char const *filename = path.c_str();
  FILE *fp = fopen(filename, "wb");
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
  size_t n = fwrite(arr_data, arr_size, 1, fp);
  if (n != 1) {
    perror(filename);
  }
  fclose(fp);
  return true;
}

template <class Arr = std::vector<char>>
static bool file_put_binary(Arr const &arr, std::string const &path) {
    return file_put_binary(std::data(arr), std::size(arr), path);
}

class BinaryReader {
    size_t cur = 0;
    std::vector<char> data;
public:
    bool is_eof() {
        return cur >= data.size();
    }
    BinaryReader(std::vector<char> data_) {
        data = std::move(data_);
    }
    size_t current() const {
        return cur;
    }
    void skip(size_t step) {
        // must use '>' rather than '>='
        if (cur + step > data.size()) {
            throw std::out_of_range("BinaryReader::read");
        }
        cur += step;
    }
    void seek_from_begin(size_t pos) {
        // must use '>' rather than '>='
        if (pos > data.size()) {
            throw std::out_of_range("BinaryReader::read");
        }
        cur = pos;
    }
    template <class T>
    T read_LE() {
        // must use '>' rather than '>='
        if (cur + sizeof(T) > data.size()) {
            throw std::out_of_range("BinaryReader::read");
        }
        T &ret = *(T *)(data.data() + cur);
        cur += sizeof(T);
        return ret;
    }

    // just work for basic type, not work for vec
    template <class T>
    T read_BE() {
        // must use '>' rather than '>='
        if (cur + sizeof(T) > data.size()) {
            throw std::out_of_range("BinaryReader::read");
        }
        T ret = *(T *)(data.data() + cur);
        char* ptr = (char*)&ret;
        for (auto i = 0; i < sizeof(T) / 2; i++) {
            std::swap(ptr[i], ptr[sizeof(T) - 1 - i]);
        }
        cur += sizeof(T);
        return ret;
    }
};
}
