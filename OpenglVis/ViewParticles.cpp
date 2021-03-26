#include "stdafx.hpp"
#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include "ShaderProgram.hpp"
#include "ViewNode.hpp"
#include "main.hpp"
#include <mutex>

namespace zenvis {

struct ViewParticles : IViewNode {
  static inline std::unique_ptr<ShaderProgram> prog_;

  size_t vertex_count;
  std::vector<float> vertex_data;
  bool vertex_updated{false};
  std::mutex vertex_lock;

  std::unique_ptr<Buffer> vbo;

  virtual void apply() override {
    std::lock_guard _(vertex_lock);

    auto pars = get_input("pars")->as<zenbase::ParticlesObject>();

    vertex_data.clear();
    vertex_count = pars->pos.size();
    for (int i = 0; i < vertex_count; i++) {
      vertex_data.push_back(pars->pos[i].x);
      vertex_data.push_back(pars->pos[i].y);
      vertex_data.push_back(pars->pos[i].z);
      vertex_data.push_back(pars->vel[i].x);
      vertex_data.push_back(pars->vel[i].y);
      vertex_data.push_back(pars->vel[i].z);
    }
    vertex_updated = true;
  }

  virtual void draw() override {  // TODO: implement frame data cacheing
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

  Program *get_program() {
    if (!prog_)
      prog_ = std::make_unique<ShaderProgram>("particles");
    auto pro = prog_.get();
    pro->use();
    return pro;
  }
};

static int defViewParticles = zen::defNodeClass<ViewParticles>("ViewParticles",
    { /* inputs: */ {
        "pars",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
