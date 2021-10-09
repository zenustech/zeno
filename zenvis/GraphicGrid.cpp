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

    attribute vec3 vPosition;
    attribute vec3 vColor;

    varying vec3 position;
    varying vec3 color;

    void main() {
        position = vPosition;
        color = vColor;

        gl_Position = mVP * vec4(position, 1.0);
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

    varying vec3 position;
    varying vec3 color;

    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
)";


namespace zenvis {

struct GraphicGrid : IGraphic {
  std::unique_ptr<Buffer> vbo;
  size_t vertex_count;

  Program *lines_prog;
  std::unique_ptr<Buffer> lines_ebo;
  size_t lines_count;

  GraphicGrid() {
    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem;
    int bound = 5;
    zeno::vec3f color(0.4f, 0.4f, 0.4f);
    for (int i = -bound; i <= bound; i++) {
      mem.push_back(vec3f(i, 0, -bound));
      mem.push_back(color);
      if (i != 0) {
        mem.push_back(vec3f(i, 0, bound));
      } else {
        mem.push_back(vec3f(0, 0, 0));
      }
      mem.push_back(color);
    }
    for (int i = -bound; i <= bound; i++) {
      mem.push_back(vec3f(-bound, 0, i));
      mem.push_back(color);
      if (i != 0) {
        mem.push_back(vec3f(bound, 0, i));
      } else {
        mem.push_back(vec3f(0, 0, 0));
      }
      mem.push_back(color);
    }
    vertex_count = mem.size() / 2;
    lines_count = vertex_count / 2;

    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    lines_prog = compile_program(vert_code, frag_code);
  }

  virtual void draw() override {
    vbo->bind();
    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    lines_prog->use();
    set_program_uniforms(lines_prog);
    CHECK_GL(glDrawArrays(GL_LINES, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
  }


};
std::unique_ptr<IGraphic> makeGraphicGrid() {
  return std::make_unique<GraphicGrid>();
}

}
