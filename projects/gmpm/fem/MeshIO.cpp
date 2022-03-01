#include "../ZensimGeometry.h"
#include "../ZensimMesh.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <zeno/logger.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

struct MakeFEMMeshFromFile2 : zeno::INode {
  virtual void apply() override {
    using mat3 = typename ZenoFEMMesh::mat3;
    using vec3 = typename ZenoFEMMesh::vec3;
    using vec4i = typename ZenoFEMMesh::vec4i;

    auto node_file = get_input<zeno::StringObject>("NodeFile")->get();
    auto ele_file = get_input<zeno::StringObject>("EleFile")->get();
    auto bou_file = get_input<zeno::StringObject>("BouFile")->get();
    float E = get_input<zeno::NumericObject>("YoungModulus")->get<float>();
    float nu = get_input<zeno::NumericObject>("PossonRatio")->get<float>();
    float d = get_input<zeno::NumericObject>("Damp")->get<float>();
    float density = get_input<zeno::NumericObject>("density")->get<float>();

    auto res = std::make_shared<ZenoFEMMesh>();
    res->_mesh = std::make_shared<PrimitiveObject>();

    // dynamics
    res->LoadVerticesFromFile(node_file); // _mesh->verts, _X
    const auto nm_verts = res->_mesh->verts.size();
    res->_x = res->_X;
    res->_V.resize(nm_verts);
    for (auto &&v : res->_V)
      v = vec3::zeros();
    res->_v = res->_V;

    res->LoadElementsFromFile(
        ele_file); // _mesh->quads, _mesh->tris, _tets, _tris

    res->LoadBindingPoints(bou_file); // _closeBindPoints, _farBindPoints
    res->_bouDoFs.resize(
        (res->_closeBindPoints.size() + res->_farBindPoints.size()) * 3);
    size_t base = 0;
    for (size_t i = 0; i != res->_closeBindPoints.size(); ++i) {
      auto vert_idx = res->_closeBindPoints[i];
      res->_bouDoFs[base++] = vert_idx * 3 + 0;
      res->_bouDoFs[base++] = vert_idx * 3 + 1;
      res->_bouDoFs[base++] = vert_idx * 3 + 2;
    }
    for (size_t i = 0; i != res->_farBindPoints.size(); ++i) {
      auto vert_idx = res->_farBindPoints[i];
      res->_bouDoFs[base++] = vert_idx * 3 + 0;
      res->_bouDoFs[base++] = vert_idx * 3 + 1;
      res->_bouDoFs[base++] = vert_idx * 3 + 2;
    }

    {
      auto tmp = res->_bouDoFs;
      zs::radix_sort(zs::omp_exec(), tmp.begin(), tmp.end(),
                     res->_bouDoFs.begin());
    }
    res->UpdateDoFsMapping(); // _freeDoFs, _DoF2FreeDoF

    const auto nm_elms = res->_mesh->quads.size();
    // aniso
    res->_elmAct.resize(nm_elms);
    for (auto &&v : res->_elmAct)
      v = mat3::identity();
    res->_elmOrient.resize(nm_elms);
    for (auto &&v : res->_elmOrient)
      v = mat3::identity();
    res->_elmWeight.resize(nm_elms);
    for (auto &&v : res->_elmWeight)
      v = vec3{1.0, 0.5, 0.5};
    // material
    res->_elmYoungModulus.resize(nm_elms);
    for (auto &&v : res->_elmYoungModulus)
      v = E;
    res->_elmPoissonRatio.resize(nm_elms);
    for (auto &&v : res->_elmPoissonRatio)
      v = nu;
    res->_elmDamp.resize(nm_elms);
    for (auto &&v : res->_elmDamp)
      v = d;
    res->_elmDensity.resize(nm_elms);
    for (auto &&v : res->_elmDensity)
      v = density;

    res->_elmVolume.resize(nm_elms);
    res->_elmMass.resize(nm_elms);
    res->_elmMinv.resize(nm_elms);
    res->_elmDmInv.resize(nm_elms);
    res->_elmdFdx.resize(nm_elms);

    res->DoPreComputation(); // volume, mass, minv, dminv, dfdx

    // res->relocate(zs::memsrc_e::device, 0);

    set_output("FEMMesh", res);
  }
};

ZENDEFNODE(MakeFEMMeshFromFile2, {
                                     {{"readpath", "NodeFile"},
                                      {"readpath", "EleFile"},
                                      {"readpath", "BouFile"},
                                      {"density"},
                                      {"YoungModulus"},
                                      {"PossonRatio"},
                                      {"Damp"}},
                                     {"FEMMesh"},
                                     {},
                                     {"FEM"},
                                 });

struct MakeIdentityMatrix2 : zeno::INode {
  void apply() override {
    auto ret = std::make_shared<ZenoAffineMatrix>();
    ret->affineMap = ZenoAffineMatrix::mat4::identity();
    set_output("eyeMat4", std::move(ret));
  }
};

ZENDEFNODE(MakeIdentityMatrix2, {{}, {"eyeMat4"}, {}, {"FEM"}});

} // namespace zeno