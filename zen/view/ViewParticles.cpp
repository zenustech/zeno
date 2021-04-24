#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <Hg/StrUtils.h>
#include <cstring>
#include <vector>
#include "ViewNode.h"

namespace zenbase {

struct ViewParticles : ViewNode {
  virtual std::vector<char> get_shader() override {
    return std::vector<char>(0);
  }

  virtual std::vector<char> get_memory() override {
    auto pars = get_input("pars")->as<zenbase::ParticlesObject>();
    size_t vertex_count = pars->pos.size();

    std::vector<char> memory(vertex_count * 6 * sizeof(float));

    size_t memi = 0;
    float *fdata = (float *)memory.data();
    for (int i = 0; i < vertex_count; i++) {
      fdata[memi++] = pars->pos[i].x;
      fdata[memi++] = pars->pos[i].y;
      fdata[memi++] = pars->pos[i].z;
      fdata[memi++] = pars->vel[i].x;
      fdata[memi++] = pars->vel[i].y;
      fdata[memi++] = pars->vel[i].z;
    }

    return memory;
  }

  virtual std::string get_data_type() const override {
    return "PARS";
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
