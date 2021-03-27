#pragma once

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

public:
  Archive(_PrivCtor) {}

  static int add(std::string const &name, std::vector<char> const &data) {
    instance().archives[name] = data;
    return 1;
  }

  static std::vector<char> const &getBytes(std::string const &name) {
    return instance().archives.at(name);
  }

  static std::string getString(std::string const &name) {
    return std::string(getBytes().data());
  }
};

}
