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

struct FiberParticleToFEMFiber2 : zeno::INode {
  virtual void apply() override {
    auto fp = get_input<zeno::PrimitiveObject>("fp");
    auto femmesh = get_input<ZenoFEMMesh>("femMesh");

    auto fiber = std::make_shared<zeno::PrimitiveObject>();
    fiber->add_attr<float>("elmID");
    fiber->add_attr<zeno::vec3f>("vel");
    fiber->resize(femmesh->_mesh->quads.size());

    const auto &meshTets = femmesh->_tets;
    const auto &meshPos = femmesh->_X;

    auto &fiberElmId = fiber->attr<float>("elmID");
    auto &fiberPos = fiber->attr<zeno::vec3f>("pos");
    auto &fiberVel = fiber->attr<zeno::vec3f>("vel");
    constexpr float sigma = 2;

    auto ompExec = zs::omp_exec();
    ompExec(zs::range(meshTets.size()), [&](std::size_t elm_id) {
      fiberElmId[elm_id] = float(elm_id);
      auto tet = meshTets[elm_id];
      // 
      auto &fpos = fiberPos[elm_id];
      for (size_t i = 0; i != 4; ++i)  {
        auto p = meshPos[tet[i]];
        fpos += zeno::vec3f{p[0], p[1], p[2]};
      }
      fpos /= 4;

      //
      auto &fdir = fiberVel[elm_id];
      fdir = zeno::vec3f(0);
      for (size_t i = 0; i < fp->size(); ++i) {
        const auto &ppos = fp->verts[i];
        const auto &pdir = fp->attr<zeno::vec3f>("vel")[i];

        float dissqrt = zeno::lengthSquared(fpos - ppos);
        float weight = exp(-dissqrt / pow(sigma, 2));

        fdir += pdir * weight;
      }
      fdir /= zeno::length(fdir);
    });

    set_output("fiberOut", fiber);
  }
};

ZENDEFNODE(FiberParticleToFEMFiber2,
           {{"fp", "femMesh"}, {"fiberOut"}, {}, {"FEM"}});

// _elmOrient
struct AddMuscleFibers2 : zeno::INode {
  virtual void apply() override {
    auto fibers = get_input<zeno::PrimitiveObject>("fibers");
    auto femmesh = get_input<ZenoFEMMesh>("femmesh");
    auto &fposs = fibers->attr<zeno::vec3f>("pos");
    const auto &fdirs = fibers->attr<zeno::vec3f>("vel");

    using mat3 = typename ZenoFEMMesh::mat3;
    using vec3 = typename ZenoFEMMesh::vec3;

    const auto &meshTets = femmesh->_tets;

    auto &fiberElmId = fibers->attr<float>("elmID");
    auto &fiberPos = fibers->attr<zeno::vec3f>("pos");
    auto &fiberVel = fibers->attr<zeno::vec3f>("vel");
    assert(fibers->size() == meshTets.size());

    for (size_t elm_id = 0; elm_id != meshTets.size(); ++elm_id) {
      auto dir0 = fdirs[elm_id];
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
    fgeo->resize(meshTets.size() * 2);
    float length = 400;
    if (has_input("length")) {
      length = get_input<NumericObject>("length")->get<float>();
    }

    auto &pos = fgeo->attr<zeno::vec3f>("pos");
    auto &lines = fgeo->lines;
    lines.resize(meshTets.size());
    float dt = 0.01;

    std::cout << "build geo" << std::endl;

    for (size_t elm_id = 0; elm_id != meshTets.size(); ++elm_id) {
      pos[elm_id] = fiberPos[elm_id];
      auto pend = fiberPos[elm_id] + dt * fiberVel[elm_id] * length;
      pos[elm_id + meshTets.size()] = pend;
      lines[elm_id] = zeno::vec2i(elm_id, elm_id + meshTets.size());
    }

    set_output("fiberGeo", fgeo);
  }
};

ZENDEFNODE(AddMuscleFibers2, {
                                 {"fibers", "femmesh"},
                                 {"outMesh", "fiberGeo"},
                                 {},
                                 {"FEM"},
                             });

// _elmWeight
struct SetUniformMuscleAnisotropicWeight2 : zeno::INode {
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

ZENDEFNODE(SetUniformMuscleAnisotropicWeight2, {
                                                   {{"inputMesh"}, {"weight"}},
                                                   {"aniMesh"},
                                                   {},
                                                   {"FEM"},
                                               });

// _elmAct
struct SetUniformActivation2 : zeno::INode {
  virtual void apply() override {
    auto mesh = get_input<ZenoFEMMesh>("inputMesh");
    auto uniform_Act =
        get_input<zeno::NumericObject>("uniform_act")->get<zeno::vec3f>();

    using mat3 = typename ZenoFEMMesh::mat3;
    using vec3 = typename ZenoFEMMesh::vec3;
    zs::omp_exec()(zs::range(mesh->_mesh->quads.size()), [&](size_t i) {
      mat3 fdir = mesh->_elmOrient[i];
      vec3 act_vec{uniform_Act[0], uniform_Act[1], uniform_Act[2]};
      mesh->_elmAct[i] = (diag_mul(fdir, act_vec) * fdir.transpose());
    });

    set_output("actMesh", mesh);
  }
};

ZENDEFNODE(SetUniformActivation2, {
                                      {{"inputMesh"}, {"uniform_act"}},
                                      {"actMesh"},
                                      {},
                                      {"FEM"},
                                  });

} // namespace zeno