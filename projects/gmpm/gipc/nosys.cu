#include "../Structures.hpp"
#include "../Utils.hpp"
#include "GIPC.cuh"
#include "cuda_tools.h"
#include "device_fem_data.cuh"
#include "femEnergy.cuh"
#include "mesh.h"
#include "zeno/types/PrimitiveObject.h"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct ZSTets : IObject {
  tetrahedra_obj tetMesh;
  device_TetraData d_tetMesh;
  bool initYet = false;
};

struct ToZSTets : INode {
  void apply() override {
    std::shared_ptr<ZSTets> ret{};
    if (has_input("ZSTets")) {
      ret = get_input<ZSTets>("ZSTets");
    } else {
      ret = std::make_shared<ZSTets>();
      ret->tetMesh = tetrahedra_obj{};
    }

    auto &tetMesh = ret->tetMesh;
    auto prim = get_input<PrimitiveObject>("prim");
    auto &pos = prim->attr<zeno::vec3f>("pos");
    auto &quads = prim->quads.values;
    double scale = 1.;
    double3 position_offset = make_double3(0, 0, 0);
    {
      const int numVerts = pos.size();
      const int numEles = quads.size();
      // auto ompExec = zs::omp_exec();

      tetMesh.vertexNum += numVerts;
      /// verts
      double xmin = std::numeric_limits<double>::max(),
             ymin = std::numeric_limits<double>::max(),
             zmin = std::numeric_limits<double>::max();
      double xmax = std::numeric_limits<double>::lowest(),
             ymax = std::numeric_limits<double>::lowest(),
             zmax = std::numeric_limits<double>::lowest();
      for (int vi = 0; vi != numVerts; ++vi) {
        const auto [x, y, z] = pos[vi];
        tetMesh.boundaryTypies.push_back(0);
        auto vertex = make_double3(scale * x - position_offset.x,
                                   scale * y - position_offset.y,
                                   scale * z - position_offset.z);
        tetMesh.vertexes.push_back(vertex);
        tetMesh.velocities.push_back(make_double3(0, 0, 0));
        tetMesh.d_velocities.push_back(make_double3(0, 0, 0));
        tetMesh.masses.push_back(0);
        tetMesh.d_positions.push_back(make_double3(0, 0, 0));

        __GEIGEN__::Matrix3x3d constraint;
        __GEIGEN__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

        tetMesh.constraints.push_back(constraint);

        if (xmin > vertex.x)
          xmin = vertex.x;
        if (ymin > vertex.y)
          ymin = vertex.y;
        if (zmin > vertex.z)
          zmin = vertex.z;
        if (xmax < vertex.x)
          xmax = vertex.x;
        if (ymax < vertex.y)
          ymax = vertex.y;
        if (zmax < vertex.z)
          zmax = vertex.z;
      };
      tetMesh.minTConer = make_double3(xmin, ymin, zmin);
      tetMesh.maxTConer = make_double3(xmax, ymax, zmax);

      tetMesh.tetrahedraNum += numEles;
      for (int ei = 0; ei != numEles; ++ei) {
        auto quad = quads[ei];
        uint4 tetrahedra;
        tetrahedra.x = quad[0] + tetMesh.tetraheraOffset;
        tetrahedra.y = quad[1] + tetMesh.tetraheraOffset;
        tetrahedra.z = quad[2] + tetMesh.tetraheraOffset;
        tetrahedra.w = quad[3] + tetMesh.tetraheraOffset;
        tetMesh.tetrahedras.push_back(tetrahedra);
        tetMesh.tetra_fiberDir.push_back(make_double3(0, 0, 0));
        ;
      }

      double boxTVolum = (tetMesh.maxTConer.x - tetMesh.minTConer.x) *
                         (tetMesh.maxTConer.y - tetMesh.minTConer.y) *
                         (tetMesh.maxTConer.z - tetMesh.minTConer.z);
      double boxVolum = (tetMesh.maxConer.x - tetMesh.minConer.x) *
                        (tetMesh.maxConer.y - tetMesh.minConer.y) *
                        (tetMesh.maxConer.z - tetMesh.minConer.z);

      if (boxTVolum > boxVolum) {
        tetMesh.maxConer = tetMesh.maxTConer;
        tetMesh.minConer = tetMesh.minTConer;
      }
      // V_prev = vertexes;
      tetMesh.tetraheraOffset = tetMesh.vertexNum;
      tetMesh.D12x12Num = 0;
      tetMesh.D9x9Num = 0;
      tetMesh.D6x6Num = 0;
      tetMesh.D3x3Num = 0;
    }
    set_output("ZSTets", ret);
  }
};

ZENDEFNODE(ToZSTets, {{"ZSTets", {"prim"}}, {"ZSTets"}, {}, {"IPC"}});

void initFEM(tetrahedra_obj &mesh);

static lbvh_f bvh_f;
static lbvh_e bvh_e;
static GIPC ipc;
constexpr auto density = 1e3;

struct ZSGIPC : INode {
  void apply() override {
    auto zstets = get_input<ZSTets>("ZSTets");
    auto &tetMesh = zstets->tetMesh;
    auto &d_tetMesh = zstets->d_tetMesh;

    // init
    if (!zstets->initYet) {
      zstets->initYet = true;
// tetMesh = tetrahedra_obj{};
#if 0
      tetMesh.load_tetrahedraVtk("tets/sphere1K.vtk", 0.5,
                                 make_double3(0, -0.9, 0));
      tetMesh.load_tetrahedraVtk("tets/cube.vtk", 0.5,
                                 make_double3(0.2, 0.1, 0.2));
#endif
      tetMesh.zsGetSurface();
      initFEM(tetMesh);
      // Init_CUDA();
      d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum, tetMesh.tetrahedraNum);

      CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses, tetMesh.masses.data(),
                                tetMesh.vertexNum * sizeof(double),
                                cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum, tetMesh.volum.data(),
                                tetMesh.tetrahedraNum * sizeof(double),
                                cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes, tetMesh.vertexes.data(),
                                tetMesh.vertexNum * sizeof(double3),
                                cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes, tetMesh.vertexes.data(),
                                tetMesh.vertexNum * sizeof(double3),
                                cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(
          d_tetMesh.tetrahedras, tetMesh.tetrahedras.data(),
          tetMesh.tetrahedraNum * sizeof(uint4), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(
          cudaMemcpy(d_tetMesh.DmInverses, tetMesh.DM_inverse.data(),
                     tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                     cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(
          cudaMemcpy(d_tetMesh.Constraints, tetMesh.constraints.data(),
                     tetMesh.vertexNum * sizeof(__GEIGEN__::Matrix3x3d),
                     cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(
          cudaMemcpy(d_tetMesh.BoundaryType, tetMesh.boundaryTypies.data(),
                     tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice));

      CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities, tetMesh.velocities.data(),
                                tetMesh.vertexNum * sizeof(double3),
                                cudaMemcpyHostToDevice));

      ipc.vertexNum = tetMesh.vertexNum;
      ipc.tetrahedraNum = tetMesh.tetrahedraNum;
      ipc._vertexes = d_tetMesh.vertexes;
      ipc._rest_vertexes = d_tetMesh.rest_vertexes;
      ipc.surf_vertexNum = tetMesh.surfVerts.size();
      ipc.surface_Num = tetMesh.surface.size();
      ipc.edge_Num = tetMesh.surfEdges.size();
      ipc.IPC_dt = get_input2<float>("dt");
      ipc.MAX_CCD_COLLITION_PAIRS_NUM =
          1 * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10)) *
               std::max((ipc.IPC_dt / 0.01), 2.0));
      ipc.MAX_COLLITION_PAIRS_NUM =
          (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2) * 3;

      printf("vertNum: %d      tetraNum: %d      faceNum: %d\n", ipc.vertexNum,
             ipc.tetrahedraNum, ipc.surface_Num);
      printf("surfVertNum: %d      surfEdgesNum: %d\n", ipc.surf_vertexNum,
             ipc.edge_Num);
      printf("maxCollisionPairsNum_CCD: %d      maxCollisionPairsNum: %d\n",
             ipc.MAX_CCD_COLLITION_PAIRS_NUM, ipc.MAX_COLLITION_PAIRS_NUM);

      ipc.MALLOC_DEVICE_MEM();

      CUDA_SAFE_CALL(cudaMemcpy(ipc._faces, tetMesh.surface.data(),
                                ipc.surface_Num * sizeof(uint3),
                                cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(ipc._edges, tetMesh.surfEdges.data(),
                                ipc.edge_Num * sizeof(uint2),
                                cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts, tetMesh.surfVerts.data(),
                                ipc.surf_vertexNum * sizeof(uint32_t),
                                cudaMemcpyHostToDevice));
      ipc.initBVH();

      ipc.sortMesh(d_tetMesh);
      CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(), d_tetMesh.vertexes,
                                tetMesh.vertexNum * sizeof(double3),
                                cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(tetMesh.surface.data(), ipc._faces,
                                ipc.surface_Num * sizeof(uint3),
                                cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(tetMesh.surfEdges.data(), ipc._edges,
                                ipc.edge_Num * sizeof(uint2),
                                cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy(tetMesh.surfVerts.data(), ipc._surfVerts,
                                ipc.surf_vertexNum * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));

      CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes, d_tetMesh.o_vertexes,
                                ipc.vertexNum * sizeof(double3),
                                cudaMemcpyDeviceToDevice));

      d_tetMesh.init(tetMesh);
      ipc.p_verts = &d_tetMesh.verts;
      ipc.p_eles = &d_tetMesh.eles;
      ipc.p_vtemp = &d_tetMesh.vtemp;
      ipc.p_etemp = &d_tetMesh.etemp;
      d_tetMesh.retrieve();   // zs
      ipc.retrieveSurfaces(); // zs

      ipc.buildBVH();
      ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer,
               tetMesh.maxConer);

      ipc.RestNHEnergy = ipc.Energy_Add_Reduction_Algorithm(7, d_tetMesh) *
                         ipc.IPC_dt * ipc.IPC_dt;
      ipc.buildCP();
      ipc.pcg_data.b = d_tetMesh.fb;
      ipc._moveDir = ipc.pcg_data.dx;

      ipc.computeXTilta(d_tetMesh);
    }
    // done init
    ipc.IPC_Solver(d_tetMesh);
    // copy data back
    CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(), ipc._vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToHost));

    set_output("ZSTets", zstets);
  }
};
ZENDEFNODE(ZSGIPC,
           {{"ZSTets", {"float", "dt", "0.01"}}, {"ZSTets"}, {}, {"IPC"}});

struct ZSTetsToPrim : INode {
  void apply() override {
    auto ret = get_input<ZSTets>("ZSTets");
    auto &tetMesh = ret->tetMesh;
    auto prim = std::make_shared<PrimitiveObject>();
    auto &pos = prim->attr<zeno::vec3f>("pos");

    prim->resize(tetMesh.vertexes.size());
    for (int i = 0; i != tetMesh.vertexes.size(); ++i) {
      auto v = tetMesh.vertexes[i];
      pos[i][0] = v.x;
      pos[i][1] = v.y;
      pos[i][2] = v.z;
    }
    prim->tris.resize(tetMesh.surface.size());
    auto &tris = prim->tris.values;
    for (int i = 0; i != tetMesh.surface.size(); ++i) {
      auto inds = tetMesh.surface[i];
      tris[i][0] = inds.x;
      tris[i][1] = inds.y;
      tris[i][2] = inds.z;
    }
    set_output("prim", prim);
  }
};

ZENDEFNODE(ZSTetsToPrim, {{"ZSTets"}, {"prim"}, {}, {"IPC"}});

//
void initFEM(tetrahedra_obj &mesh) {

  double massSum = 0;
  float angleX = zs::g_pi / 4, angleY = -zs::g_pi / 4, angleZ = zs::g_pi / 2;
  __GEIGEN__::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
  __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
  __GEIGEN__::__set_Mat_val(rotationZ, cos(angleZ), -sin(angleZ), 0,
                            sin(angleZ), cos(angleZ), 0, 0, 0, 1);
  __GEIGEN__::__set_Mat_val(rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0,
                            sin(angleY), 0, cos(angleY));
  __GEIGEN__::__set_Mat_val(rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0,
                            sin(angleX), cos(angleX));

  for (int i = 0; i < mesh.tetrahedraNum; i++) {
    __GEIGEN__::Matrix3x3d DM;
    __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);

    __GEIGEN__::Matrix3x3d DM_inverse;
    __GEIGEN__::__Inverse(DM, DM_inverse);

    double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

    mesh.masses[mesh.tetrahedras[i].x] += vlm * density / 4;
    mesh.masses[mesh.tetrahedras[i].y] += vlm * density / 4;
    mesh.masses[mesh.tetrahedras[i].z] += vlm * density / 4;
    mesh.masses[mesh.tetrahedras[i].w] += vlm * density / 4;

    massSum += vlm * density;

    mesh.DM_inverse.push_back(DM_inverse);
    mesh.volum.push_back(vlm);
  }

  mesh.meanMass = massSum / mesh.vertexNum;
  mesh.meanVolum = mesh.meanMass / density;
}

} // namespace zeno
