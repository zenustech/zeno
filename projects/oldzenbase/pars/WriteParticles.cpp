#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/StringObject.h>
#include <cstring>

namespace zeno {

static void writepars(
    const char *path,
    std::vector<glm::vec3> &positions,
    std::vector<glm::vec3> &velocities)
{
  FILE *fp = fopen(path, "w");
  if (!fp) {
    perror(path);
    abort();
  }

  for (auto const &v: positions) {
    fprintf(fp, "v %f %f %f\n", v.x, v.y, v.z);
  }
  for (auto const &v: velocities) {
    fprintf(fp, "#v_vel %f %f %f\n", v.x, v.y, v.z);
  }
  fclose(fp);
}


struct WriteParticles : zeno::INode {
  virtual void apply() override {
    auto path = get_param<std::string>("path"));
    auto pars = get_input("pars")->as<ParticlesObject>();
    writepars(path.c_str(), pars->pos, pars->vel);
  }
};

static int defWriteParticles = zeno::defNodeClass<WriteParticles>("WriteParticles",
    { /* inputs: */ {
    "pars",
    }, /* outputs: */ {
    }, /* params: */ {
    {"writepath", "path", ""},
    }, /* category: */ {
    "particles",
    }});


struct ExportParticles : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto pars = get_input("pars")->as<ParticlesObject>();
    writepars(path->get().c_str(), pars->pos, pars->vel);
  }
};

static int defExportParticles = zeno::defNodeClass<ExportParticles>("ExportParticles",
    { /* inputs: */ {
    "pars",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "particles",
    }});

}
