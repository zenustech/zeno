#include <zeno/PrimitiveObject.h>
#include <zeno/vec.h>
#include <zeno/zeno.h>

#include "../ZensimGeometry.h"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zeno {

struct ToPrimitiveObject : INode {
  virtual void apply() override {
    auto a = has_input("boundMin")
                 ? get_input<NumericObject>("boundMin")->get<vec3f>()
                 : vec3f(-1, -1, -1);
    auto b = has_input("boundMax")
                 ? get_input<NumericObject>("boundMax")->get<vec3f>()
                 : vec3f(+1, +1, +1);
    auto connType = get_param<int>("connType");

    auto &zspars = get_input<ZenoParticles>("ZSParticles")->get();

    auto prim = std::make_shared<PrimitiveObject>();
    auto &pos = prim->add_attr<vec3f>("pos");
    auto &J = prim->add_attr<float>("J");

    zs::match([&pos, &J](auto &zspars) {
      {
        auto X = zspars.attrVector("pos").clone({zs::memsrc_e::host, -1});
        const auto cnt = X.size();
        pos.resize(cnt);
#pragma omp parallel for
        for (int i = 0; i < cnt; ++i)
          pos[i] = vec3f{X[i][0], X[i][1], X[i][2]};
      }
      {
        auto F = zspars.attrMatrix("F").clone({zs::memsrc_e::host, -1});
        const auto cnt = F.size();
        J.resize(cnt);
#pragma omp parallel for
        for (int i = 0; i < cnt; ++i) {
          float sum = 0.f;
          const auto &f = F[i];
          for (int d = 0; d < 9; ++d)
            sum += F[i].data()[d] *;
          J[i] = vec3f{X[i][0], X[i][1], X[i][2]};
        }
      }
    })(zspars);

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ToPrimitiveObject, {/* inputs: */ {
                                   "ZSParticles",
                               },
                               /* outputs: */
                               {
                                   "prim",
                               },
                               /* params: */
                               {

                               },
                               /* category: */
                               {
                                   "ZensimGeometry",
                               }});

} // namespace zeno
