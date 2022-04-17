#include "stdafx.hpp"
#include "MyShader.hpp"
#include <Hg/StrUtils.h>
#include <Hg/IterUtils.h>
#include <Hg/Archive.hpp>
#include "stdafx.hpp"


namespace zenvis {

struct MyProgram : Program {
  std::unique_ptr<Shader> vs, fs, gs;

  MyProgram(std::string const &vert, std::string const &frag, std::string const &geo="") {
    vs = std::make_unique<Shader>(GL_VERTEX_SHADER);
    fs = std::make_unique<Shader>(GL_FRAGMENT_SHADER);
    gs = std::make_unique<Shader>(GL_GEOMETRY_SHADER);
    //printf("=================\n");
    //printf("(VERT)\n%s\n", vert.c_str());
    //printf("=================\n");
    //printf("(FRAG)\n%s\n", frag.c_str());
    //printf("=================\n");
    vs->compile(vert);
    fs->compile(frag);
    
    attach(*vs);
    attach(*fs);
    if(geo!="")
    {
      std::cout<<"compiling geo shader\n";
      gs->compile(geo);
      attach(*gs);
    }
    link();
  }
};


static std::map<std::string, std::unique_ptr<Program>> progs;

Program *compile_program(std::string const &vert, std::string const &frag, std::string const &geo) {
    auto key = vert + frag + geo;
    if (progs.find(key) == progs.end()) {
        std::unique_ptr<MyProgram> prog;
        try {
            prog = std::make_unique<MyProgram>(vert, frag, geo);
            progs.emplace(key, std::move(prog));
        } catch (ShaderCompileException const &sce) {
            return nullptr;
        }
    }
    return progs.at(key).get();
}

}
