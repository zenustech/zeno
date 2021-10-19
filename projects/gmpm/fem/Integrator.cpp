#include "../ZensimMesh.h"
#include "../ZensimModel.h"
#include "../ZenoSimulation.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

#if 0
struct ExplicitTimeStepping : zeno::INode {
  void apply() override {
    auto mesh = get_input<ZenoFEMMesh>("mesh");
    auto force_model = get_input<ZenoForceModel>("muscleForce");
    auto damping_model = get_input<ZenoDampingForceModel>("dampForce");
    auto integrator = get_input<ZenoExplicitTimeIntegrator>("integrator");
    auto epsilon = get_input<zeno::NumericObject>("epsilon")->get<float>();
    auto closed_T = get_input<TransformMatrix>("CT");
    auto far_T = get_input<TransformMatrix>("FT");

    size_t clen = integrator->_intPtr->GetCouplingLength();
    size_t curID = (integrator->_stepID + clen) % clen;
    size_t preID = (integrator->_stepID + clen - 1) % clen;

    // set initial guess
    integrator->_traj[curID] = integrator->_traj[preID];

    auto depa = std::make_shared<zeno::PrimitiveObject>();
    auto &depa_pos = depa->attr<zeno::vec3f>("pos");
    using value_type = typename ZenoFEMMesh::value_type;
    using vec3 = typename ZenoFEMMesh::vec3;
    using vec4 = vec<value_type, 4>;

    for (size_t i = 0; i != mesh->_closeBindPoints.size(); ++i) {
      size_t idx = mesh->_closeBindPoints[i];
      vec3 vert{mesh->_mesh->verts[idx][0], mesh->_mesh->verts[idx][1], mesh->_mesh->verts[idx][2]};
      vert = closed_T->Mat * vert;
      integrator->_x.segment(idx * 3, 3) = vert.segment(0, 3);

      depa_pos.emplace_back(mesh->_mesh->verts[idx]);
    }

    for (size_t i = 0; i < mesh->_farBindPoints.size(); ++i) {
      size_t idx = mesh->_farBindPoints[i];
      Vec4d vert;
      vert << mesh->_mesh->verts[idx][0], mesh->_mesh->verts[idx][1],
          mesh->_mesh->verts[idx][2], 1.0;
      vert = far_T->Mat * vert;
      integrator->_x.segment(idx * 3, 3) = vert.segment(0, 3);

      depa_pos.emplace_back(mesh->_mesh->verts[idx]);
    }

    set_output("depa", std::move(depa));

    /*
    size_t iter_idx = 0;

    VecXd deriv(mesh->_mesh->size() * 3);
    VecXd ruc(mesh->_freeDoFs.size()), dpuc(mesh->_freeDoFs.size()),
        dp(mesh->_mesh->size() * 3);

    _HValueBuffer.resize(mesh->_connMatrix.nonZeros());
    _HucValueBuffer.resize(mesh->_freeConnMatrix.nonZeros());

    const size_t max_iters = 20;
    const size_t max_linesearch = 20;
    _wolfeBuffer.resize(max_linesearch);
    */
  }
}

ZENDEFNODE(ExplicitTimeStepping, {
    {{"mesh"},{"muscleForce"},{"dampForce"},{"integrator"},{"epsilon"},{"CT"},{"FT"}},
    {"curentFrame","depa"},
    {},
    {"FEM"},
});
#endif

} // namespace zeno