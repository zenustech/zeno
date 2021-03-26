#pragma once

#include "stdafx.hpp"

struct ShaderProgram : Program {
  std::unique_ptr<Shader> vs, fs;

  ShaderProgram(std::string name) {
    vs = std::make_unique<Shader>(GL_VERTEX_SHADER);
    fs = std::make_unique<Shader>(GL_FRAGMENT_SHADER);
    std::string basedir = "/home/bate/Develop/Mn/ZenProjects/OpenglVis/";
    vs->compile(file_get_content(basedir + name + ".vert"));
    fs->compile(file_get_content(basedir + name + ".frag"));
    attach(*vs);
    attach(*fs);
    link();
  }
};
