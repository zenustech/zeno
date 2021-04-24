#pragma once

#include <string>

namespace zenbase {

struct ShaderMacros {
  std::string lines;

  void add(std::string const &name, std::string const &value) {
    lines += "#define " + name + " " + value + "\n";
  }

  void apply(std::string &source) {
    source = "#version 330 core\n" + lines + "/**************/\n" + source;
  }
};

}
