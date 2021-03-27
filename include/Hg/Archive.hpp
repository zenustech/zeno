#pragma once

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace hg {

struct Archive {
private:
  std::map<std::string, std::vector<char>> archives;
  static std::unique_ptr<Archive> _instance;
  struct _PrivCtor {};

  static Archive &instance() {
    if (!_instance)
      _instance = std::make_unique<Archive>(_PrivCtor{});
    return *_instance;
  }

  std::vector<char> const &_getBytes(std::string const &name) {
    //assert(archives.find(name) != archives.end());
    return archives.at(name);
  }

public:
  Archive(_PrivCtor) {}

  static int add(std::string const &name, char const *data, size_t size) {
    std::vector<char> &arr = instance().archives[name];
    arr.resize(size);
    std::memcpy(arr.data(), data, size);
    return 1;
  }

  static std::vector<char> const &getBytes(std::string const &name) {
    return instance()._getBytes(name);
  }

  static std::string getString(std::string const &name) {
    std::vector<char> const &bytes = getBytes(name);
    return std::string(bytes.data(), bytes.size());
  }
};

#ifdef HG_ARCHIVE_IMPLEMENTATION
std::unique_ptr<Archive> Archive::_instance;
#endif

}
