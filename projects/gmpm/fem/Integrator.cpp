#include "../ZensimMesh.h"
#include "../ZensimModel.h"
#include "../ZensimGeometry.h"
#include "../ZenoSimulation.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/fem/ElementToDof.hpp"
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

struct ExplicitTimeStepping : zeno::INode {
  void apply() override {
    auto mesh = get_input<ZenoFEMMesh>("mesh");
    auto force_model = get_input<ZenoForceModel>("muscleForce");
    auto damping_model = get_input<ZenoDampingForceModel>("dampForce");
    auto integrator = get_input<ZenoExplicitTimeIntegrator>("integrator");
    auto epsilon = get_input<zeno::NumericObject>("epsilon")->get<float>();
    auto closed_T = get_input<ZenoAffineMatrix>("CT");
    auto far_T = get_input<ZenoAffineMatrix>("FT");

    // set initial guess
    const auto sz = mesh->_mesh->size();
    const auto numEle = mesh->_mesh->quads.size();
    integrator->_x.resize(sz);
    integrator->_v.resize(sz);
    integrator->_f.resize(sz);

    auto depa = std::make_shared<zeno::PrimitiveObject>();
    auto &depa_pos = depa->attr<zeno::vec3f>("pos");
    using value_type = typename ZenoFEMMesh::value_type;
    using vec3 = typename ZenoFEMMesh::vec3;
    using mat3 = typename ZenoFEMMesh::mat3;
    using vec12 = zs::vec<value_type, 12>;
    using vec4 = zs::vec<value_type, 4>;

    for (size_t i = 0; i != mesh->_closeBindPoints.size(); ++i) {
      size_t idx = mesh->_closeBindPoints[i];
      vec3 vert{mesh->_mesh->verts[idx][0], mesh->_mesh->verts[idx][1], mesh->_mesh->verts[idx][2]};
      vert = closed_T->affineMap * vert;
      integrator->_x[idx] = vert;

      depa_pos.emplace_back(mesh->_mesh->verts[idx]);
    }

    for (size_t i = 0; i != mesh->_farBindPoints.size(); ++i) {
      size_t idx = mesh->_farBindPoints[i];
      vec3 vert{mesh->_mesh->verts[idx][0], mesh->_mesh->verts[idx][1], mesh->_mesh->verts[idx][2]};
      vert = far_T->affineMap * vert;
      integrator->_x[idx] = vert;

      depa_pos.emplace_back(mesh->_mesh->verts[idx]);
    }

    set_output("depa", std::move(depa));

    auto ompExec = zs::omp_exec();
    // elm force
    ompExec(integrator->_f, [](vec3 &v) { v = vec3::zeros(); });
    ompExec(zs::range(numEle), [&](std::size_t ei) {
      auto tetIndices = mesh->_tets[ei];
      const auto &X = mesh->_X;
      const auto &V = mesh->_V;
      vec12 u_n{}, v_n{};
      for (int v = 0, base = 0; v != 4; ++v) {
        const auto vi = tetIndices[v];
        const auto &pos = X[vi];
        const auto &vel = V[vi];
        for (int d = 0; d != 3; ++d, ++base) {
          u_n[base] = pos[d];
          v_n[base] = vel[d];
        }
      }
      auto eleForce = eval_tet_force(mesh->_elmVolume[ei], mesh->_elmMass[ei], mesh->_elmYoungModulus[ei], mesh->_elmPoissonRatio[ei], integrator->_dt, integrator->_gravity, mesh->_elmdFdx[ei], mesh->_elmDmInv[ei], force_model, u_n, mesh->_elmAct[ei], mesh->_elmWeight[ei], mesh->_elmOrient[ei], damping_model, damping_model->coeff, v_n);
      // atomic add
      for (int v = 0, base = 0; v != 4; ++v) {
        auto &f = integrator->_f[tetIndices[v]];
        for (int d = 0; d != 3; ++d, ++base)
          zs::atomic_add(zs::exec_omp, &f[d], eleForce[base]);
      }
    });
    // update V
    ompExec(zs::range(numEle), [&](std::size_t ei) {
      auto tetIndices = mesh->_tets[ei];
      auto &X = mesh->_X;
      auto &V = mesh->_V;
      const auto f = integrator->_f[ei];
      const auto dt = integrator->_dt;
      const auto nodalMass = mesh->_elmMass[ei] * 0.25;
      for (int v = 0, base = 0; v != 4; ++v) {
        auto &vel = V[tetIndices[v]];
        for (int d = 0; d != 3; ++d, ++base)
          zs::atomic_add(zs::exec_omp, &vel[d], (value_type)(f[base] * dt / nodalMass));
      }
    });
    // update X
    ompExec(zs::range(sz), [&](std::size_t vi) {
      auto &X = mesh->_X;
      auto &V = mesh->_V;
      const auto dt = integrator->_dt;
      X[vi] += V[vi] * dt;
    });
  }

};

ZENDEFNODE(ExplicitTimeStepping, {
    {{"mesh"},{"muscleForce"},{"dampForce"},{"integrator"},{"epsilon"},{"CT"},{"FT"}},
    {"curentFrame","depa"},
    {},
    {"FEM"},
});

} // namespace zeno