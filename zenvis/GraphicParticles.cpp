#include "stdafx.hpp"
#include "ShaderProgram.hpp"
#include "IGraphic.hpp"
#include "frames.hpp"
#include "main.hpp"

namespace zenvis {

struct GraphicParticles : IGraphic {
  static inline std::unique_ptr<ShaderProgram> prog_;
  size_t vertex_count;
  std::unique_ptr<Buffer> vbo;

  explicit GraphicParticles(ObjectData const &obj) {
    vertex_count = obj.memory->size() / (6 * sizeof(float));
    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    vbo->bind_data(obj.memory->data(), obj.memory->size());
  }

  virtual void draw() override {
    auto pro = get_program();
    set_program_uniforms(pro);
    glEnable(GL_POINT_SPRITE_ARB);
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
    glDisable(GL_POINT_SPRITE_ARB);
  }

  Program *get_program() {
    if (!prog_)
      prog_ = std::make_unique<ShaderProgram>("particles");
    auto pro = prog_.get();
    pro->use();
    return pro;
  }
};

std::unique_ptr<IGraphic> makeGraphicParticles(ObjectData const &obj) {
  return std::make_unique<GraphicParticles>(obj);
}

}
