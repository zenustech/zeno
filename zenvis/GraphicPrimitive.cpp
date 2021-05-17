#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <zen/vec.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>
#include <zen/PrimitiveObject.h>

namespace zenvis {

struct GraphicPrimitive : IGraphic {
  Program *prog;
  size_t vertex_count;
  std::unique_ptr<Buffer> vbo;

  GraphicPrimitive
    ( zenbase::PrimitiveObject *prim
    , std::string const &path
    ) {
    auto const &pos = prim->add_attr<zen::vec3f>("pos");
    auto const &clr = prim->add_attr<zen::vec3f>("clr");
    vertex_count = pos.size();
    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zen::vec3f> mem(vertex_count * 2);
    for (int i = 0; i < vertex_count; i++) {
        mem[2 * i + 0] = pos[i];
        mem[2 * i + 1] = clr[i];
    }
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    auto vert = hg::file_get_content(path + ".points.vert");
    auto frag = hg::file_get_content(path + ".points.frag");

    if (vert.size() == 0) {
      vert =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"\n"
"attribute vec3 vPosition;\n"
"attribute vec3 vVercolor;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 vercolor;\n"
"\n"
"void main()\n"
"{\n"
"  position = vPosition;\n"
"  vercolor = vVercolor;\n"
"\n"
"  gl_Position = mVP * vec4(position, 1.0);\n"
"  gl_PointSize = 5.0;\n"
"}\n";
    }
    if (frag.size() == 0) {
      frag =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 vercolor;\n"
"\n"
"void main()\n"
"{\n"
"  if (length(gl_PointCoord - vec2(0.5)) > 0.5)\n"
"    discard;\n"
"  gl_FragColor = vec4(vercolor, 1.0);\n"
"}\n";
    }

    prog = compile_program(vert, frag);
  }

  virtual void draw() override {
    prog->use();
    set_program_uniforms(prog);

    vbo->bind();
    vbo->attribute(/*index=*/0,
        /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 6,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/1,
        /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 6,
        GL_FLOAT, /*count=*/3);

    CHECK_GL(glDrawArrays(GL_POINTS, /*first=*/0, /*count=*/vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
  }
};

std::unique_ptr<IGraphic> makeGraphicPrimitive
    ( zenbase::PrimitiveObject *prim
    , std::string const &path
    ) {
  return std::make_unique<GraphicPrimitive>(prim, path);
}

}
