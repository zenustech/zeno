#include "stdafx.hpp"
#include "MyShader.hpp"
#include <Hg/StrUtils.h>
#include <Hg/IterUtils.h>
#include <Hg/Archive.hpp>
#include "stdafx.hpp"


namespace zenvis {

struct MyProgram : Program {
  std::unique_ptr<Shader> vs, fs;

  MyProgram(std::string const &vert, std::string const &frag) {
    vs = std::make_unique<Shader>(GL_VERTEX_SHADER);
    fs = std::make_unique<Shader>(GL_FRAGMENT_SHADER);
    //printf("=================\n");
    //printf("(VERT)\n%s\n", vert.c_str());
    //printf("=================\n");
    //printf("(FRAG)\n%s\n", frag.c_str());
    //printf("=================\n");
    vs->compile(vert);
    fs->compile(frag);
    attach(*vs);
    attach(*fs);
    link();
  }
};


static std::map<std::string, std::unique_ptr<Program>> progs;

Program *compile_program(std::string const &vert, std::string const &frag) {
    auto key = vert + frag;
    if (progs.find(key) == progs.end()) {
        progs[key] = std::make_unique<MyProgram>(vert, frag);
    }
    return progs.at(key).get();
}

}
