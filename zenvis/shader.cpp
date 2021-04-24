#include <Hg/IterUtils.h>
#include <Hg/Archive.hpp>
#include "stdafx.hpp"
#include "shader.hpp"

namespace zenvis {

struct MyProgram : Program {
  std::unique_ptr<Shader> vs, fs;

  MyProgram(std::vector<char> const &shader) {
    vs = std::make_unique<Shader>(GL_VERTEX_SHADER);
    fs = std::make_unique<Shader>(GL_FRAGMENT_SHADER);
    /*auto _ = hg::split_str(hg::assign_conv<std::string>(shader), '\0');
    assert(_.size() == 2);
    auto vert = _[0], frag = _[1];
    printf("=================\n");
    printf("%s\n", vert.c_str());
    printf("=================\n");
    printf("%s\n", frag.c_str());
    printf("=================\n");*/
    auto vert = hg::Archive::getString("particles.vert");
    auto frag = hg::Archive::getString("particles.frag");
    vs->compile(vert);
    fs->compile(frag);
    attach(*vs);
    attach(*fs);
    link();
  }
};


std::map<std::string, std::unique_ptr<MyProgram>> prog_cache;

Program *compile_program(std::vector<char> const &shader)
{
  auto key = hg::assign_conv<std::string>(shader);
  if (prog_cache.find(key) != prog_cache.end())
    return prog_cache.at(key).get();

  auto prog = std::make_unique<MyProgram>(shader);
  auto prog_ptr = prog.get();
  prog_cache[key] = std::move(prog);
  return prog_ptr;
}

}
