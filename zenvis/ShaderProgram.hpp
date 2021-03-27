#pragma once

#include "stdafx.hpp"
#include <Hg/Archive.hpp>

namespace zenvis {

struct ShaderProgram : Program {
  std::unique_ptr<Shader> vs, fs;

  ShaderProgram(std::string name) {
    vs = std::make_unique<Shader>(GL_VERTEX_SHADER);
    fs = std::make_unique<Shader>(GL_FRAGMENT_SHADER);
    vs->compile(hg::Archive::getString(name + ".vert"));
    fs->compile(hg::Archive::getString(name + ".frag"));
    attach(*vs);
    attach(*fs);
    link();
  }
};

}
