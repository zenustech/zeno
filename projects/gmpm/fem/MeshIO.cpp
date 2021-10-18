#include "../ZensimMesh.h"
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

struct ZsMakeFEMMeshFromFile : zeno::INode {
  virtual void apply() override {
    auto node_file = get_input<zeno::StringObject>("NodeFile")->get();
    auto ele_file = get_input<zeno::StringObject>("EleFile")->get();
    auto bou_file = get_input<zeno::StringObject>("BouFile")->get();
    float E = get_input<zeno::NumericObject>("YoungModulus")->get<float>();
    float nu = get_input<zeno::NumericObject>("PossonRatio")->get<float>();
    float density = get_input<zeno::NumericObject>("density")->get<float>();

    auto res = std::make_shared<ZenoFEMMesh>();
    res->_mesh = std::make_shared<PrimitiveObject>();

    res->LoadVerticesFromFile(node_file);
    res->LoadElementsFromFile(ele_file);
    res->LoadBoundaryIndicesFromFile(bou_file);

    using mat3 = typename ZenoFEMMesh::mat3;
    using vec3 = typename ZenoFEMMesh::vec3;
    // allocate memory
    size_t nm_elms = res->_mesh->quads.size();
    res->_elmAct.resize(nm_elms);
    for (auto &&v : res->_elmAct)
      v = mat3::identity();
    res->_elmOrient.resize(nm_elms);
    for (auto &&v : res->_elmOrient)
      v = mat3::identity();
    res->_elmWeight.resize(nm_elms);
    for (auto &&v : res->_elmWeight)
      v = vec3{1.0, 0.5, 0.5};
    res->_elmYoungModulus.resize(nm_elms);
    for (auto &&v : res->_elmYoungModulus)
      v = E;
    res->_elmPoissonRatio.resize(nm_elms);
    for (auto &&v : res->_elmPoissonRatio)
      v = nu;
    res->_elmDensity.resize(nm_elms);
    for (auto &&v : res->_elmDensity)
      v = density;

    res->_elmVolume.resize(nm_elms);
    res->_elmdFdx.resize(nm_elms);
    res->_elmMass.resize(nm_elms);
    res->_elmMinv.resize(nm_elms);

    res->DoPreComputation();  // voluem, mass, minv, dfdx
    res->UpdateDoFsMapping(); // _freeDoFs, _DoF2FreeDoF

    // res->relocate(zs::memsrc_e::device, 0);

    set_output("FEMMesh", res);
  }
};

ZENDEFNODE(ZsMakeFEMMeshFromFile, {
                                      {{"readpath", "NodeFile"},
                                       {"readpath", "EleFile"},
                                       {"readpath", "BouFile"},
                                       {"density"},
                                       {"YoungModulus"},
                                       {"PossonRatio"}},
                                      {"FEMMesh"},
                                      {},
                                      {"FEM"},
                                  });

} // namespace zeno