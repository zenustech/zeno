#include "stdafx.hpp"
#include <zen/zen.h>
#include <zen/MeshObject.h>
#include "ShaderProgram.hpp"
#include "ViewNode.hpp"
#include "main.hpp"
#include <mutex>

namespace zenvis {

struct ViewMesh : IViewNode {
  static inline std::unique_ptr<ShaderProgram> prog_;

  size_t vertex_count;
  std::vector<float> vertex_data;
  bool vertex_updated{false};
  std::mutex vertex_lock;

  std::unique_ptr<Buffer> vbo;

  virtual void apply() override {
    std::lock_guard _(vertex_lock);

    auto mesh = get_input("mesh")->as<zenbase::MeshObject>();

    vertex_data.clear();
    vertex_count = mesh->vertices.size();
    for (int i = 0; i < vertex_count; i++) {
      vertex_data.push_back(mesh->vertices[i].x);
      vertex_data.push_back(mesh->vertices[i].y);
      vertex_data.push_back(mesh->vertices[i].z);
      vertex_data.push_back(mesh->uvs[i].x);
      vertex_data.push_back(mesh->uvs[i].y);
      vertex_data.push_back(mesh->normals[i].x);
      vertex_data.push_back(mesh->normals[i].y);
      vertex_data.push_back(mesh->normals[i].z);
    }
    vertex_updated = true;
  }

  virtual void draw() override {
    std::lock_guard _(vertex_lock);

    if (vertex_updated) {
      vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
      vbo->bind_data(vertex_data);
      vertex_updated = false;
    }

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

static int defViewMesh = zen::defNodeClass<ViewMesh>("ViewMesh",
    { /* inputs: */ {
        "mesh",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
