#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "frames.hpp"
#include "shader.hpp"
#include "main.hpp"
#include <zen/vec.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>

namespace zenvis {

struct GraphicPrimitive : IGraphic {
  Program *prog;
  size_t vertex_count;
  std::unique_ptr<Buffer> vbo;

  GraphicPrimitive
    ( std::vector<zen::vec3f> const &pos
    , std::vector<zen::vec3f> const &vel
    ) {
    vertex_count = pos.size();
    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zen::vec3f> mem(vertex_count);
    for (int i = 0; i < vertex_count; i++) {
        mem[2 * i + 0] = pos[i];
        mem[2 * i + 1] = vel[i];
    }
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    prog = compile_program(
        hg::file_get_content("assets/particles.vert"),
        hg::file_get_content("assets/particles.frag"));
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
    ( std::vector<zen::vec3f> const &pos
    , std::vector<zen::vec3f> const &vel
    ) {
  return std::make_unique<GraphicPrimitive>(pos, vel);
}

}
