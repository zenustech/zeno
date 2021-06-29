#include <zeno/zen.h>
#include <zeno/ParticlesObject.h>
#include <zeno/StringObject.h>
#include <cstring>

namespace zen {

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


struct WriteParticles : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto pars = get_input("pars")->as<ParticlesObject>();
    writepars(path.c_str(), pars->pos, pars->vel);
  }
};

static int defWriteParticles = zen::defNodeClass<WriteParticles>("WriteParticles",
    { /* inputs: */ {
    "pars",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "path", ""},
    }, /* category: */ {
    "particles",
    }});


struct ExportParticles : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto pars = get_input("pars")->as<ParticlesObject>();
    writepars(path->get().c_str(), pars->pos, pars->vel);
  }
};

static int defExportParticles = zen::defNodeClass<ExportParticles>("ExportParticles",
    { /* inputs: */ {
    "pars",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "particles",
    }});

}
