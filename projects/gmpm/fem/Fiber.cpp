#include "../ZensimMesh.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/logger.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

// _elmOrient
struct AddMuscleFibers : zeno::INode {
  virtual void apply() override {
    auto fibers = get_input<zeno::PrimitiveObject>("fibers");
    auto femmesh = get_input<ZenoFEMMesh>("femmesh");
    auto &fposs = fibers->attr<zeno::vec3f>("pos");
    auto &fdirs = fibers->attr<zeno::vec3f>("vel");

    const auto &mpos = femmesh->_mesh->attr<zeno::vec3f>("pos");
    const auto &tets = femmesh->_mesh->quads;

    std::vector<zeno::vec3f> tet_dirs;
    std::vector<zeno::vec3f> tet_pos;
    tet_dirs.resize(tets.size(), zeno::vec3f(0));
    tet_pos.resize(tets.size());

    // Retrieve the center of the tets
    for (size_t elm_id = 0; elm_id != tets.size(); ++elm_id) {
      auto tet = tets[elm_id];
      tet_pos[elm_id] = zeno::vec3f(0);

      for (size_t i = 0; i != 4; ++i)
        tet_pos[elm_id] += mpos[tet[i]];

      tet_pos[elm_id] /= 4;
    }

    float sigma = 2;

    for (size_t i = 0; i != fibers->size(); ++i) {
      auto fpos = fposs[i];
      auto fdir = fdirs[i];
      fdir /= zeno::length(fdir);

      for (size_t elm_id = 0; elm_id != tets.size(); ++elm_id) {
        float dissqrt = zeno::lengthSquared(fpos - tet_pos[elm_id]);
        float weight = exp(-dissqrt / pow(sigma, 2));
        tet_dirs[elm_id] += weight * fdir;
      }
    }

    for (size_t elm_id = 0; elm_id != tets.size(); ++elm_id)
      tet_dirs[elm_id] /= zeno::length(tet_dirs[elm_id]);

    using mat3 = typename ZenoFEMMesh::mat3;
    using vec3 = typename ZenoFEMMesh::vec3;

    for (size_t elm_id = 0; elm_id != tets.size(); ++elm_id) {
      auto dir0 = tet_dirs[elm_id];
      auto ref_dir = dir0;
      ref_dir[0] += 1;
      auto dir1 = zeno::cross(dir0, ref_dir);
      dir1 /= zeno::length(dir1);
      auto dir2 = zeno::cross(dir0, dir1);
      dir2 /= zeno::length(dir2);

      mat3 orient;
      orient = mat3{dir0[0], dir1[0], dir2[0], dir0[1], dir1[1],
                    dir2[1], dir0[2], dir1[2], dir2[2]};

      femmesh->_elmOrient[elm_id] = orient;
    }

    set_output("outMesh", femmesh);

    std::cout << "output fiber geo" << std::endl;

    auto fgeo = std::make_shared<zeno::PrimitiveObject>();
    fgeo->resize(tets.size() * 2);
    float length = 400;
    if (has_input("length")) {
      length = get_input<NumericObject>("length")->get<float>();
    }

    auto &pos = fgeo->attr<zeno::vec3f>("pos");
    auto &lines = fgeo->lines;
    lines.resize(tets.size());
    float dt = 0.01;

    std::cout << "build geo" << std::endl;

    for (size_t elm_id = 0; elm_id != tets.size(); ++elm_id) {
      pos[elm_id] = tet_pos[elm_id];
      auto pend = tet_pos[elm_id] + dt * tet_dirs[elm_id] * length;
      pos[elm_id + tets.size()] = pend;
      lines[elm_id] = zeno::vec2i(elm_id, elm_id + tets.size());
    }

    set_output("fiberGeo", fgeo);
  }
};

ZENDEFNODE(AddMuscleFibers, {
                                {"fibers", "femmesh"},
                                {"outMesh", "fiberGeo"},
                                {},
                                {"FEM"},
                            });

// _elmWeight
struct SetUniformMuscleAnisotropicWeight : zeno::INode {
  virtual void apply() override {
    auto mesh = get_input<ZenoFEMMesh>("inputMesh");
    auto uni_weight =
        get_input<zeno::NumericObject>("weight")->get<zeno::vec3f>();
    using vec3 = typename ZenoFEMMesh::vec3;
    zs::omp_exec()(zs::range(mesh->_mesh->quads.size()), [&](size_t i) {
      mesh->_elmWeight[i] = vec3{uni_weight[0], uni_weight[1], uni_weight[2]};
    });
    // for (size_t i = 0; i != mesh->_mesh->quads.size(); ++i)
    //  mesh->_elmWeight[i] << uni_weight[0], uni_weight[1], uni_weight[2];
    set_output("aniMesh", mesh);
  }
};

ZENDEFNODE(SetUniformMuscleAnisotropicWeight, {
                                                  {{"inputMesh"}, {"weight"}},
                                                  {"aniMesh"},
                                                  {},
                                                  {"FEM"},
                                              });

// _elmAct
struct SetUniformActivation : zeno::INode {
  virtual void apply() override {
    auto mesh = get_input<ZenoFEMMesh>("inputMesh");
    auto uniform_Act =
        get_input<zeno::NumericObject>("uniform_act")->get<zeno::vec3f>();

    using mat3 = typename ZenoFEMMesh::mat3;
    using vec3 = typename ZenoFEMMesh::vec3;
    zs::omp_exec()(zs::range(mesh->_mesh->quads.size()), [&](size_t i) {
      mat3 fdir = mesh->_elmOrient[i];
      vec3 act_vec{uniform_Act[0], uniform_Act[1], uniform_Act[2]};
      mesh->_elmAct[i] = mul(diag_mul(fdir, act_vec), fdir.transpose());
    });
#if 0
    for (size_t i = 0; i < mesh->_mesh->quads.size(); ++i) {
      Mat3x3d fdir = mesh->_elmOrient[i];
      Vec3d act_vec;
      act_vec << uniform_Act[0], uniform_Act[1], uniform_Act[2];
      mesh->_elmAct[i] << fdir * act_vec.asDiagonal() * fdir.transpose();
    }
#endif

    set_output("actMesh", mesh);
  }
};

ZENDEFNODE(SetUniformActivation, {
                                     {{"inputMesh"}, {"uniform_act"}},
                                     {"actMesh"},
                                     {},
                                     {"FEM"},
                                 });

} // namespace zeno