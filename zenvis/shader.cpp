#include <Hg/StrUtils.h>
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
    auto _ = hg::split_str(hg::assign_conv<std::string>(shader), '\0');
    assert(_.size() == 2);
    auto vert = _[0], frag = _[1];
    init(vert, frag);
  }

  MyProgram(std::string const &vert, std::string const &frag) {
    init(vert, frag);
  }

  void init(std::string const &vert, std::string const &frag) {
    printf("=================\n");
    printf("(VERT)\n%s\n", vert.c_str());
    printf("=================\n");
    printf("(FRAG)\n%s\n", frag.c_str());
    printf("=================\n");
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

Program *compile_program(std::string const &vert, std::string const &frag)
{
  std::string key = "simple";
  if (prog_cache.find(key) != prog_cache.end())
    return prog_cache.at(key).get();
  auto prog = std::make_unique<MyProgram>(vert, frag);
  auto prog_ptr = prog.get();
  prog_cache[key] = std::move(prog);
  return prog_ptr;
}

}
