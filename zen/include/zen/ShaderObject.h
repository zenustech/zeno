#pragma once

#include <zen/zen.h>
#include <Hg/IterUtils.h>
#include <vector>

namespace zenbase {

struct ShaderObject : zen::IObject {
  std::string vert;
  std::string frag;

  std::vector<char> serialize() {
    return hg::assign_conv<std::vector<char>>(vert + '\0' + frag);
  }
};

}
