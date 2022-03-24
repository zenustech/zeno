#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <zeno/utils/vec.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>

using std::string;
using zeno::vec3f;

static string vert_code = R"(
    #version 120

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;
    uniform vec3 mLightDir;

    attribute vec3 vPosition;

    void main() {
        vec3 pos = vPosition;
        pos = normalize(mLightDir) * 20 * pos[0];

        gl_Position = mVP * vec4(pos, 1.0);
    }
)";

static string frag_code = R"(
    #version 120

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;


    void main() {
        gl_FragColor = vec4(1, 0.62, 0.17, 1.0);
    }
)";


namespace zenvis {

struct GraphicLightDir : IGraphic {
  std::unique_ptr<Buffer> vbo;
  size_t vertex_count;

  Program *lines_prog;
  std::unique_ptr<Buffer> lines_ebo;
  size_t lines_count;

  GraphicLightDir() {
    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem;
    vec3f origin = vec3f(0, 0, 0);
    vec3f dir = vec3f(1, 1, 1);

    mem.push_back(origin);
    mem.push_back(dir);


    vertex_count = 2;
    lines_count = 1;

    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    lines_prog = compile_program(vert_code, frag_code);
  }
  virtual void drawShadow() override
  {
    
  }
  virtual void draw() override {
    vbo->bind();
    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

    lines_prog->use();
    set_program_uniforms(lines_prog);
    CHECK_GL(glDrawArrays(GL_LINES, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->unbind();
  }


};
std::unique_ptr<IGraphic> makeGraphicLightDir() {
  return std::make_unique<GraphicLightDir>();
}

}
