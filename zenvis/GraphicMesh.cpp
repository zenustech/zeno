#include "stdafx.hpp"
#include "ShaderProgram.hpp"
#include "IGraphic.hpp"
#include "frames.hpp"
#include "main.hpp"

namespace zenvis {

struct GraphicMesh : IGraphic {
  static inline std::unique_ptr<ShaderProgram> prog_;

  size_t vertex_count;
  std::unique_ptr<Buffer> vbo;

  explicit GraphicMesh(ObjectData const &obj) {
    vertex_count = obj.memory->size() / (6 * sizeof(float));

    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    vbo->bind_data(obj.memory->data(), obj.memory->size());
  }

  virtual void draw() override {
    auto pro = get_program();
    set_program_uniforms(pro);

    vbo->bind();
    vbo->attribute(/*index=*/0,
        /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 8,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/1,
        /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 8,
        GL_FLOAT, /*count=*/2);
    vbo->attribute(/*index=*/2,
        /*offset=*/sizeof(float) * 5, /*stride=*/sizeof(float) * 8,
        GL_FLOAT, /*count=*/3);

    CHECK_GL(glDrawArrays(GL_TRIANGLES, /*first=*/0, /*count=*/vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->disable_attribute(2);
    vbo->unbind();
  }

  Program *get_program() {
    if (!prog_)
      prog_ = std::make_unique<ShaderProgram>("mesh");
    auto pro = prog_.get();
    pro->use();
    return pro;
  }
};

std::unique_ptr<IGraphic> makeGraphicMesh(ObjectData const &obj) {
  return std::make_unique<GraphicMesh>(obj);
}

}
