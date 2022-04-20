#include "Libs/AdvectionOp.h"
#include "Libs/Assembler.h"
#include "Libs/BasicOp.h"
#include "Libs/Builder.h"
#include "Libs/EulerGasDenseGrid.h"
#include "Libs/GasAssembler.h"
#include "Libs/GasSimulator.h"
#include "Libs/IdealGas.h"
#include "Libs/ProjectionOp.h"
#include "Libs/StateDense.h"
#include "Libs/TVDRK.h"
#include "Libs/WENO.h"
#include <omp.h>
#include <stdio.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
namespace zeno {

struct ZenCompressAero : zeno::IObject {
  typename ZenEulerGas::FieldHelperDenseDouble3Ptr gas;
};

struct DenseField : zeno::IObject {
  virtual std::string getType() { return std::string(); }
};

template <class T> struct DenseFieldWrapper : DenseField {
  DenseFieldWrapper() {
    dx = 0;
    bmin = openvdb::Coord(0, 0, 0);
  }
  T *m_grid;
  float dx;
  openvdb::Coord bmin;
  int ni, nj, nk;
  size_t m_size;
  size_t size() { return m_size; }
  std::string spatialType;

  virtual std::string getType() {
    if (std::is_same<T, double>::value) {
      return std::string("FloatGrid");
    } else if (std::is_same<T, int>::value) {
      return std::string("Int32Grid");
    } else if (std::is_same<T, ZenEulerGas::Vector<double, 3>>::value) {
      return std::string("Vec3fGrid");
    } else {
      return std::string("");
    }
  }
};

using DenseFloatGrid = DenseFieldWrapper<double>;
using DenseIntGrid = DenseFieldWrapper<int>;
using DenseFloat3Grid = DenseFieldWrapper<ZenEulerGas::Vector<double, 3>>;

struct DenseFieldToVDB : zeno::INode {
  virtual void apply() override {
    auto type = get_input("inDenseField")->as<DenseField>()->getType();

    if (type == std::string("FloatGrid")) {
      auto inField = get_input("inDenseField")->as<DenseFloatGrid>();
      int ni = inField->ni;
      int nj = inField->nj;
      int nk = inField->nk;
      auto oField = zeno::IObject::make<VDBFloatGrid>();
      auto transform =
          openvdb::math::Transform::createLinearTransform(inField->dx);
      if (inField->spatialType == std::string("center")) {
        transform->postTranslate(openvdb::Vec3d{0.5, 0.5, 0.5} *
                                 double(inField->dx));
      }

      oField->m_grid->setTransform(transform);

      auto writeTo = oField->m_grid->getAccessor();

#pragma omp parallel for
      for (size_t index = 0; index < inField->size(); index++) {
        int i = index % (ni + 4);
        int j = index / (ni + 4) % (nj + 4);
        int k = index / ((ni + 4) * (nj + 4));
        if (i > 1 && i < ni + 2 && j > 1 && j < nj + 2 && k > 1 && k < nk + 2) {
          openvdb::Coord xyz(i, j, k);
          xyz = xyz + inField->bmin - openvdb::Coord(2, 2, 2);
          writeTo.setValue(xyz, inField->m_grid[index]);
        }
      }
      set_output("VDBField", oField);
    } else if (type == std::string("Int32Grid")) {
      auto inField = get_input("inDenseField")->as<DenseIntGrid>();
      int ni = inField->ni;
      int nj = inField->nj;
      int nk = inField->nk;
      auto oField = zeno::IObject::make<VDBFloatGrid>();
      auto transform =
          openvdb::math::Transform::createLinearTransform(inField->dx);
      if (inField->spatialType == std::string("center")) {
        transform->postTranslate(openvdb::Vec3d{0.5, 0.5, 0.5} *
                                 double(inField->dx));
      }

      oField->m_grid->setTransform(transform);

      auto writeTo = oField->m_grid->getAccessor();

#pragma omp parallel for
      for (size_t index = 0; index < inField->size(); index++) {
        int i = index % (ni + 4);
        int j = index / (ni + 4) % (nj + 4);
        int k = index / ((ni + 4) * (nj + 4));
        if (i > 1 && i < ni + 2 && j > 1 && j < nj + 2 && k > 1 && k < nk + 2) {
          openvdb::Coord xyz(i, j, k);
          xyz = xyz + inField->bmin - openvdb::Coord(2, 2, 2);
          writeTo.setValue(xyz, inField->m_grid[index]);
        }
      }
      set_output("VDBField", oField);
    } else if (type == std::string("Float3Grid")) {
      auto inField = get_input("inDenseField")->as<DenseFloat3Grid>();
      int ni = inField->ni;
      int nj = inField->nj;
      int nk = inField->nk;
      auto oField = zeno::IObject::make<VDBFloat3Grid>();
      auto transform =
          openvdb::math::Transform::createLinearTransform(inField->dx);
      if (inField->spatialType == std::string("center")) {
        transform->postTranslate(openvdb::Vec3d{0.5, 0.5, 0.5} *
                                 double(inField->dx));
      }
      oField->m_grid->setTransform(transform);

      auto writeTo = oField->m_grid->getAccessor();

#pragma omp parallel for
      for (size_t index = 0; index < inField->size(); index++) {
        int i = index % (ni + 4);
        int j = index / (ni + 4) % (nj + 4);
        int k = index / ((ni + 4) * (nj + 4));
        if (i > 1 && i < ni + 2 && j > 1 && j < nj + 2 && k > 1 && k < nk + 2) {
          openvdb::Coord xyz(i, j, k);
          xyz = xyz + inField->bmin - openvdb::Coord(2, 2, 2);
          openvdb::Vec3R value(inField->m_grid[index][0],
                               inField->m_grid[index][1],
                               inField->m_grid[index][2]);
          writeTo.setValue(xyz, value);
        }
      }
      set_output("VDBField", oField);
    }
  }
};

ZENDEFNODE(DenseFieldToVDB, {
                                {"inDenseField"},
                                {"VDBField"},
                                {},
                                {"CompressibleFlow"},
                            });

struct GetDenseField : zeno::INode {
  virtual void apply() override {
    auto field = get_param<std::string>(("FieldName"));
    auto gas = get_input("inSolverData")->as<ZenCompressAero>()->gas;
    if (field == std::string("p")) {
      auto oField = zeno::IObject::make<DenseFloatGrid>();
      oField->m_grid = &(gas->Pf[0]);
      oField->dx = gas->dx;
      oField->bmin = openvdb::Coord(gas->grid.bbmin[0], gas->grid.bbmin[1],
                                    gas->grid.bbmin[2]);

      oField->ni = gas->grid.bbmax[0] - gas->grid.bbmin[0];
      oField->nj = gas->grid.bbmax[1] - gas->grid.bbmin[1];
      oField->nk = gas->grid.bbmax[2] - gas->grid.bbmin[2];

      oField->spatialType = std::string("vertex");
      oField->m_size = gas->Pf.size();
      set_output("outDenseField", oField);
    }
    if (field == std::string("rho")) {
      auto oField = zeno::IObject::make<DenseFloatGrid>();
      oField->m_grid = &(gas->rhof[0]);
      oField->dx = gas->dx;
      oField->bmin = openvdb::Coord(gas->grid.bbmin[0], gas->grid.bbmin[1],
                                    gas->grid.bbmin[2]);

      oField->ni = gas->grid.bbmax[0] - gas->grid.bbmin[0];
      oField->nj = gas->grid.bbmax[1] - gas->grid.bbmin[1];
      oField->nk = gas->grid.bbmax[2] - gas->grid.bbmin[2];

      oField->spatialType = std::string("center");
      oField->m_size = gas->Pf.size();
      set_output("outDenseField", oField);
    }
    if (field == std::string("u")) {
      auto oField = zeno::IObject::make<DenseFloat3Grid>();
      oField->m_grid = &(gas->uf[0]);
      oField->dx = gas->dx;
      oField->bmin = openvdb::Coord(gas->grid.bbmin[0], gas->grid.bbmin[1],
                                    gas->grid.bbmin[2]);

      oField->ni = gas->grid.bbmax[0] - gas->grid.bbmin[0];
      oField->nj = gas->grid.bbmax[1] - gas->grid.bbmin[1];
      oField->nk = gas->grid.bbmax[2] - gas->grid.bbmin[2];

      oField->spatialType = std::string("center");
      oField->m_size = gas->Pf.size();
      set_output("outDenseField", oField);
    }
    if (field == std::string("cellType")) {
      auto oField = zeno::IObject::make<DenseIntGrid>();
      oField->m_grid = &(gas->cell_type[0]);
      oField->dx = gas->dx;
      oField->bmin = openvdb::Coord(gas->grid.bbmin[0], gas->grid.bbmin[1],
                                    gas->grid.bbmin[2]);

      oField->ni = gas->grid.bbmax[0] - gas->grid.bbmin[0];
      oField->nj = gas->grid.bbmax[1] - gas->grid.bbmin[1];
      oField->nk = gas->grid.bbmax[2] - gas->grid.bbmin[2];

      oField->spatialType = std::string("center");
      oField->m_size = gas->Pf.size();
      set_output("outDenseField", oField);
    }
  }
};
ZENDEFNODE(GetDenseField, {
                              {"inSolverData"},
                              {"outDenseField"},
                              {{"string", "FieldName", "p u rho"}},
                              {"CompressibleFlow"},
                          });

struct MakeCompressibleFlow : zeno::INode {
  virtual void apply() override {
    auto bmin =
        get_input("bmin")->as<zeno::NumericObject>()->get<zeno::vec3f>();
    auto bmax =
        get_input("bmax")->as<zeno::NumericObject>()->get<zeno::vec3f>();
    auto dx = get_input("dx")->as<zeno::NumericObject>()->get<float>();
    auto q_ambStr = get_input("q_amb")->as<zeno::StringObject>()->get();

    auto flowData = zeno::IObject::make<ZenCompressAero>();
    int nx, ny, nz;
    nx = (bmax[0] - bmin[0]) / dx;
    ny = (bmax[1] - bmin[1]) / dx;
    nz = (bmax[2] - bmin[2]) / dx;
    ZenEulerGas::Array<double, 5, 1> q_amb;
    sscanf(q_ambStr.c_str(), "%lf %lf %lf %lf %lf", &(q_amb[0]), &(q_amb[1]),
           &(q_amb[2]), &(q_amb[3]), &(q_amb[4]));
    printf("%lf %lf %lf %lf %lf\n", q_amb[0], q_amb[1], q_amb[2], q_amb[3],
           q_amb[4]);
    ZenEulerGas::Array<int, 3, 1> ibmin, ibmax;
    ibmin[0] = bmin[0] / dx;
    ibmin[1] = bmin[1] / dx;
    ibmin[2] = bmin[2] / dx;
    ibmax[0] = ibmin[0] + nx;
    ibmax[1] = ibmin[1] + ny;
    ibmax[2] = ibmin[2] + nz;
    flowData->gas =
        new ZenEulerGas::FieldHelperDenseDouble3(q_amb, ibmin, ibmax, dx);

    set_output("CompressibleFlow", flowData);
  }
};

ZENDEFNODE(MakeCompressibleFlow, {
                                     {"bmin", "bmax", "dx", "q_amb"},
                                     {"CompressibleFlow"},
                                     {},
                                     {"CompressibleFlow"},
                                 });

struct CompressibleSimStates : zeno::IObject {
  ZenEulerGas::solverControld data;
};

struct makeCompressibleSim : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    auto simParam = zeno::IObject::make<CompressibleSimStates>();
    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));
    simParam->data = sim.getSolverControl();
    set_output("outFlowData", get_input("inFlowData"));
    set_output("SimParam", simParam);
  }
};
ZENDEFNODE(makeCompressibleSim, {
                                    {"inFlowData"},
                                    {"outFlowData", "SimParam"},
                                    {},
                                    {"CompressibleFlow"},
                                });

struct BackupField : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));

    sim.backup();

    set_output("outFlowData", get_input("inFlowData"));
  }
};
ZENDEFNODE(BackupField, {
                            {"inFlowData"},
                            {"outFlowData"},
                            {},
                            {"CompressibleFlow"},
                        });

struct InitField : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    auto simParam = get_input("simParam")->as<CompressibleSimStates>();
    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));

    sim.setSolverControl(simParam->data);
    sim.initialize();
    simParam->data = sim.getSolverControl();

    set_output("outFlowData", get_input("inFlowData"));
  }
};
ZENDEFNODE(InitField, {
                          {"inFlowData", "simParam"},
                          {"outFlowData"},
                          {},
                          {"CompressibleFlow"},
                      });

struct MakeVelocityPressure : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    auto simParam = get_input("simParam")->as<CompressibleSimStates>();
    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));

    sim.setSolverControl(simParam->data);
    sim.convert_q_to_primitives();
    simParam->data = sim.getSolverControl();

    // counting solid cell num
    int num = 0;
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::SOLID)
            num++;
        });
    std::cout << "solid cell num after EOS is " << num << std::endl;

    set_output("outFlowData", get_input("inFlowData"));
  }
};
ZENDEFNODE(MakeVelocityPressure, {
                                     {"inFlowData", "simParam"},
                                     {"outFlowData"},
                                     {},
                                     {"CompressibleFlow"},
                                 });

struct CompressibleMarkDOF : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));

    // counting solid cell num
    int num = 0;
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::SOLID)
            num++;
        });
    std::cout << "solid cell num before marking dof is " << num << std::endl;

    sim.mark_dof();
    // counting solid cell num
    num = 0;
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::SOLID)
            num++;
        });
    // std::cout << "solid cell num after marking dof is " << num << std::endl;

    // // just trial
    // flowData->gas
    //     ->cell_type[flowData->gas->grid[ZenEulerGas::Vector<int,
    //     3>::Zero()].idx] = ZenEulerGas::CellType::SOLID;

    set_output("outFlowData", get_input("inFlowData"));
  }
};
ZENDEFNODE(CompressibleMarkDOF, {
                                    {"inFlowData"},
                                    {"outFlowData"},
                                    {},
                                    {"CompressibleFlow"},
                                });

struct MarkMovingSolidCellsByVDB : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    auto simParam = get_input("simParam")->as<CompressibleSimStates>();
    auto sdf_vdb = get_input("SDFVDBField")->as<VDBFloatGrid>();
    auto solid_vel_vdb = get_input("SolidVelVDBField")->as<VDBFloat3Grid>();
    auto sdf_access = sdf_vdb->m_grid->getAccessor();
    auto solid_vel_access = solid_vel_vdb->m_grid->getAccessor();

    // counting solid cell num
    int num = 0;
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::SOLID)
            num++;
        });
    std::cout << "solid cell num before marking sdf is " << num << std::endl;

    // reset cell type to origin
    // reset solid velocity to zero
    flowData->gas->cell_type = flowData->gas->cell_type_origin;
    std::fill(flowData->gas->us.begin(), flowData->gas->us.end(),
              ZenEulerGas::Vector<double, 3>::Zero());
    // loop from bbmin to bbmax
    // check if sdf is neg, if yes set that grid (if origin is gas) as
    // solid, and then copy the velocity to solid cell
    // remember to handle for marking(treat solid just like bound), adv(disable
    // mixed bc, all reflective)+prj(source term), which are in other files
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          openvdb::Coord xyz(I(0), I(1), I(2));
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::GAS) {
            if (sdf_access.getValue(xyz) < 0) {
              std::cout << "found a solid cell" << std::endl;

              flowData->gas->cell_type[idx] = ZenEulerGas::CellType::SOLID;
              auto vel = solid_vel_access.getValue(xyz);
              flowData->gas->us[idx] =
                  ZenEulerGas::Vector<double, 3>(vel.x(), vel.y(), vel.z());

              std::cout << "solid cell vel is " << flowData->gas->us[idx]
                        << std::endl;
            }
          }
        });
    // then, for the newly released gas cell, set the
    // value by adj values or amb
    // fix_newly_released_gas_cell_qs
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::GAS &&
              flowData->gas->cell_type_backup[idx] ==
                  ZenEulerGas::CellType::SOLID) {
            ZenEulerGas::Array<double, 5, 1> sum_Qs =
                ZenEulerGas::Array<double, 5, 1>::Zero();
            int sum_Qs_n = 0;
            flowData->gas->iterateKernel(
                [&](const ZenEulerGas::Vector<int, 3> &I,
                    const ZenEulerGas::Vector<int, 3> &adj_I) {
                  // dont fall outof the bbox
                  if (!flowData->gas->grid.in_bbox(adj_I, 0))
                    return;
                  if (flowData->gas->grid[adj_I].idx < 0)
                    return;
                  int adj_idx = flowData->gas->grid[adj_I].idx;
                  if (flowData->gas->cell_type_backup[adj_idx] ==
                      ZenEulerGas::CellType::GAS) {
                    sum_Qs += flowData->gas->q_backup[adj_idx];
                    sum_Qs_n++;
                  }
                },
                I, -1, 2);
            if (sum_Qs_n > 0) {
              sum_Qs /= (double)sum_Qs_n;
              flowData->gas->q[idx] = sum_Qs;
            } else {
              std::cout << "a gas cell released from solid has no gas neighbor"
                        << std::endl;
              flowData->gas->q[idx] = simParam->data.q_amb;
            }
          }
        });
    // counting solid cell num
    num = 0;
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::SOLID)
            num++;
        });
    std::cout << "solid cell num after marking sdf is " << num << std::endl;
    // set output reference for flowData
    set_output("outFlowData", get_input("inFlowData"));
    // following this node you should add markDofNode and BackupNode
    // sequentially
  }
};
ZENDEFNODE(MarkMovingSolidCellsByVDB, {
                                          {
                                              "inFlowData",
                                              "simParam",
                                              "SDFVDBField",
                                              "SolidVelVDBField",
                                          },
                                          {"outFlowData"},
                                          {},
                                          {"CompressibleFlow"},
                                      });

struct CompressibleAdvection : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    auto simParam = get_input("simParam")->as<CompressibleSimStates>();
    auto rk_order =
        get_input("RK_Order")->as<zeno::NumericObject>()->get<int>();
    auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();

    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));

    sim.setSolverControl(simParam->data);
    sim.advection(dt, rk_order);
    simParam->data = sim.getSolverControl();

    // counting solid cell num
    int num = 0;
    flowData->gas->iterateGridParallel(
        [&](const ZenEulerGas::Vector<int, 3> &I) {
          int idx = flowData->gas->grid[I].idx;
          if (flowData->gas->cell_type[idx] == ZenEulerGas::CellType::SOLID)
            num++;
        });
    // std::cout << "solid cell num after marking dof is " << num << std::endl;

    set_output("outFlowData", get_input("inFlowData"));
  }
};
ZENDEFNODE(CompressibleAdvection,
           {
               {"inFlowData", "simParam", "RK_Order", "dt"},
               {"outFlowData"},
               {},
               {"CompressibleFlow"},
           });

struct CompressibleProjection : zeno::INode {
  virtual void apply() override {
    auto flowData = get_input("inFlowData")->as<ZenCompressAero>();
    auto simParam = get_input("simParam")->as<CompressibleSimStates>();
    auto rk_order =
        get_input("RK_Order")->as<zeno::NumericObject>()->get<int>();
    auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();

    ZenEulerGas::zenCompressSim sim(
        flowData->gas->dx, flowData->gas->grid.bbmin, flowData->gas->grid.bbmax,
        flowData->gas->m_q_amb, *(flowData->gas));

    sim.setSolverControl(simParam->data);
    sim.projection(dt, rk_order);
    simParam->data = sim.getSolverControl();
    set_output("outFlowData", get_input("inFlowData"));
  }
};
ZENDEFNODE(CompressibleProjection,
           {
               {"inFlowData", "simParam", "RK_Order", "dt"},
               {"outFlowData"},
               {},
               {"CompressibleFlow"},
           });

} // namespace zeno
