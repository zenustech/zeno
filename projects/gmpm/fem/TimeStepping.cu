#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct ExplicitTimeStepping : INode {
  using dtiles_t = zs::TileVector<double, 32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<double, 3>;
  using mat3 = zs::vec<double, 3, 3>;

  template <typename Model>
  void computeElasticImpulse(zs::CudaExecutionPolicy &cudaPol,
                             const Model &model, const tiles_t &verts,
                             const tiles_t &eles, dtiles_t &vtemp, float dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(zs::range(eles.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             eles = proxy<space>({}, eles), model,
             dt] __device__(int ei) mutable {
              auto DmInv = eles.pack<3, 3>("IB", ei);
              auto dFdX = dFdXMatrix(DmInv);
              auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
              vec3 xs[4] = {verts.pack<3>("x", inds[0]).cast<double>(),
                            verts.pack<3>("x", inds[1]).cast<double>(),
                            verts.pack<3>("x", inds[2]).cast<double>(),
                            verts.pack<3>("x", inds[3]).cast<double>()};
              mat3 F{};
              {
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];
                auto x3x0 = xs[3] - xs[0];
                auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                               x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                F = Ds * DmInv;
              }
              auto P = model.first_piola(F);
              auto vole = eles("vol", ei);
              auto vecP = flatten(P);
              auto dFdXT = dFdX.transpose();
              auto vfdt = -vole * dt * (dFdXT * vecP);

              for (int i = 0; i != 4; ++i) {
                auto vi = inds[i];
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &vtemp("grad", d, vi), vfdt(i * 3 + d));
              }
            });
  }

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto models = zstets->getModel();
    auto dt = get_input2<float>("dt");
    auto &verts = zstets->getParticles();
    auto &eles = zstets->getQuadraturePoints();

    dtiles_t vtemp{verts.get_allocator(), {{"grad", 3}}, verts.size()};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt] __device__(int i) mutable {
              // vtemp.tuple<3>("grad", i) = vec3::zeros();
              // gravity impulse
              auto m = verts("m", i);
              vtemp.tuple<3>("grad", i) = m * vec3{0, -9, 0} * dt;
            });
    match([&](auto &elasticModel) {
      computeElasticImpulse(cudaPol, elasticModel, verts, eles, vtemp, dt);
    })(models.getElasticModel());

    // projection
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             eles = proxy<space>({}, eles), dt] __device__(int vi) mutable {
              if (verts("x", 1, vi) > 0.5)
                vtemp.tuple<3>("grad", vi) = vec3::zeros();
            });

    // update velocity and positions
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             eles = proxy<space>({}, eles), dt] __device__(int vi) mutable {
              auto fdt = vtemp.pack<3>("grad", vi);
              auto m = verts("m", vi);
              auto dv = fdt / m * dt;
              auto vn = verts.pack<3>("v", vi);
              vn += dv;
              verts.tuple<3>("v", vi) = vn;
              verts.tuple<3>("x", vi) = verts.pack<3>("x", vi) + vn * dt;
            });

    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(ExplicitTimeStepping, {{"ZSParticles", {"float", "dt", "0.01"}},
                                  {"ZSParticles"},
                                  {},
                                  {"FEM"}});

} // namespace zeno