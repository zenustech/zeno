#include <Partio.h>
#include <zeno/ParticlesObject.h>
#include <zeno/zeno.h>
static void outputBgeo(std::string path, const std::vector<glm::vec3> &pos,
                       const std::vector<glm::vec3> &vel) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  Partio::ParticleAttribute vH, posH;
  vH = parts->addAttribute("v", Partio::VECTOR, 3);
  posH = parts->addAttribute("position", Partio::VECTOR, 3);
  auto idx = parts->addParticles(pos.size());
  size_t i = 0;
  for (auto itr = idx; itr != parts->end(); ++itr, ++i) {
    // int idx = parts->addParticle();
    float *_p = parts->dataWrite<float>(posH, i);
    float *_v = parts->dataWrite<float>(vH, i);
    _p[0] = pos[i][0];
    _p[1] = pos[i][1];
    _p[2] = pos[i][2];
    _v[0] = vel[i][0];
    _v[1] = vel[i][1];
    _v[2] = vel[i][2];
  }
  printf("writing\n");
  Partio::write(path.c_str(), *parts, /*force compresse*/ false);
  parts->release();
}

namespace zeno {

struct WriteBgeo : zeno::INode {
  virtual void apply() override {
    auto path = get_param<std::string>("path"));
    auto data = get_input("data")->as<ParticlesObject>();
    outputBgeo(path, data->pos, data->vel);
  }
};

static int defWriteBgeo =
    zeno::defNodeClass<WriteBgeo>("WriteBgeo", {/* inputs: */ {
                                                    "data",
                                                },
                                                /* outputs: */ {}, /* params: */
                                                {
                                                    {"string", "path", ""},
                                                },
                                                /* category: */
                                                {
                                                    "particles",
                                                }});

} // namespace zeno
