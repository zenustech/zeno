#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"

namespace zs {

  OpenVDBStruct loadVec3fGridFromVdbFile(const std::string &fn) {
    using GridType = openvdb::Vec3fGrid;
    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr grids = file.getGrids();
    file.close();

    using SDFPtr = typename GridType::Ptr;
    SDFPtr grid;
    for (openvdb::GridPtrVec::iterator iter = grids->begin(); iter != grids->end(); ++iter) {
      openvdb::GridBase::Ptr it = *iter;
      if ((*iter)->isType<GridType>()) {
        grid = openvdb::gridPtrCast<GridType>(*iter);
        /// display meta data
        for (openvdb::MetaMap::MetaIterator it = grid->beginMeta(); it != grid->endMeta(); ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
        break;
      }
    }
    return OpenVDBStruct{grid};
  }
  SparseLevelSet<3> convertLevelSetGridToSparseLevelSet(const OpenVDBStruct &sdf,
                                                        const OpenVDBStruct &vel) {
    using SDFGridType = openvdb::FloatGrid;
    using SDFTreeType = SDFGridType::TreeType;
    using SDFRootType = SDFTreeType::RootNodeType;  // level 3 RootNode
    assert(SDFRootType::LEVEL == 3);
    using SDFInt1Type = SDFRootType::ChildNodeType;  // level 2 InternalNode
    using SDFInt2Type = SDFInt1Type::ChildNodeType;  // level 1 InternalNode
    using SDFLeafType = SDFTreeType::LeafNodeType;   // level 0 LeafNode

    using VelGridType = openvdb::Vec3fGrid;
    using VelTreeType = VelGridType::TreeType;
    using VelRootType = VelTreeType::RootNodeType;  // level 3 RootNode
    assert(VelRootType::LEVEL == 3);
    using VelInt1Type = VelRootType::ChildNodeType;  // level 2 InternalNode
    using VelInt2Type = VelInt1Type::ChildNodeType;  // level 1 InternalNode
    using VelLeafType = VelTreeType::LeafNodeType;   // level 0 LeafNode

    using SDFPtr = typename SDFGridType::Ptr;
    using VelPtr = typename VelGridType::Ptr;
    const SDFPtr &sdfGridPtr = sdf.as<SDFPtr>();
    const VelPtr &velGridPtr = vel.as<VelPtr>();

    using IV = typename SparseLevelSet<3>::table_t::key_t;
    using TV = vec<f32, 3>;

    SparseLevelSet<3> ret{};
    const auto leafCount = sdfGridPtr->tree().leafCount();
    ret._sideLength = 8;
    ret._space = 512;
    ret._dx = sdfGridPtr->transform().voxelSize()[0];
    ret._backgroundValue = sdfGridPtr->background();
    {
      auto v = velGridPtr->background();
      ret._backgroundVecValue = TV{v[0], v[1], v[2]};
    }
    ret._table = typename SparseLevelSet<3>::table_t{leafCount, memsrc_e::host, -1};
    ret._tiles = typename SparseLevelSet<3>::tiles_t{
        {{"sdf", 1}, {"vel", 3}}, leafCount * ret._space, memsrc_e::host, -1};
    {
      openvdb::CoordBBox box = sdfGridPtr->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = sdfGridPtr->indexToWorld(box.min());
      auto world_max = sdfGridPtr->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        ret._min(d) = world_min[d];
        ret._max(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = sdfGridPtr->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          ret._min(d) = pos[d] < ret._min(d) ? pos[d] : ret._min(d);
          ret._max(d) = pos[d] > ret._max(d) ? pos[d] : ret._max(d);
        }
      }
    }
    openvdb::Mat4R v2w = sdfGridPtr->transform().baseMap()->getAffineMap()->getMat4();

    sdfGridPtr->transform().print();
    fmt::print("background value: {}. dx: {}. box: [{}, {}, {} ~ {}, {}, {}]\n",
               ret._backgroundValue, ret._dx, ret._min[0], ret._min[1], ret._min[2], ret._max[0],
               ret._max[1], ret._max[2]);

    auto w2v = v2w.inverse();
    vec<float, 4, 4> transform;
    for (auto &&[r, c] : ndrange<2>(4)) transform(r, c) = w2v[r][c];  /// use [] for access
    ret._w2v = transform;

    auto table = proxy<execspace_e::host>(ret._table);
    auto tiles = proxy<execspace_e::host>({"sdf", "vel"}, ret._tiles);
    table.clear();

    SDFTreeType::LeafCIter sdfIter = sdfGridPtr->tree().cbeginLeaf();
    VelTreeType::LeafCIter velIter = velGridPtr->tree().cbeginLeaf();
    for (; sdfIter && velIter; ++sdfIter, ++velIter) {
      const SDFTreeType::LeafNodeType &sdfNode = *sdfIter;
      const VelTreeType::LeafNodeType &velNode = *velIter;
      if (sdfNode.onVoxelCount() != velNode.onVoxelCount()) {
        fmt::print("sdf grid and vel grid structure not consistent!\n");
      }
      if (sdfNode.onVoxelCount() > 0) {
        IV coord{};
        {
          auto cell = sdfNode.beginValueOn();
          for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        }

        auto blockid = coord;
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d)
          blockid[d] += (coord[d] < 0 ? -ret._sideLength + 1 : 0);
        blockid = blockid / ret._sideLength;
        auto blockno = table.insert(blockid);

        int cellid = 0;
        auto sdfCell = sdfNode.beginValueAll();
        auto velCell = velNode.beginValueAll();
        for (; sdfCell && velCell; ++sdfCell, ++velCell, ++cellid) {
          auto sdf = sdfCell.getValue();
          auto vel = velCell.getValue();
          tiles.val("sdf", blockno * ret._space + cellid) = sdf;
          tiles.template tuple<3>("vel", blockno * ret._space + cellid)
              = TV{vel[0], vel[1], vel[2]};
        }
      }
    }
    if (sdfIter || velIter) fmt::print("sdf grid and vel grid structure not consistent!\n");
    return ret;
  }
  SparseLevelSet<3> convertLevelSetGridToSparseLevelSet(const OpenVDBStruct &sdf,
                                                        const OpenVDBStruct &vel,
                                                        const MemoryHandle mh) {
    return convertLevelSetGridToSparseLevelSet(sdf, vel).clone(mh);
  }

  tuple<DenseGrid<float, int, 3>, DenseGrid<vec<float, 3>, int, 3>, vec<float, 3>, vec<float, 3>>
  readPhiVelFromVdbFile(const std::string &fn, float dx) {
    constexpr int dim = 3;
    using TV = vec<float, dim>;
    using IV = vec<int, dim>;
    using PhiTreeT = typename openvdb::FloatGrid::TreeType;
    // using openvdb::Vec3fGrid = typename openvdb::Grid<
    //    typename openvdb::tree::Tree4<openvdb::Vec3f, 5, 4, 3>::Type>;
    using VelTreeT = typename openvdb::Vec3fGrid::TreeType;

    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr my_grids = file.getGrids();
    file.close();
    typename openvdb::FloatGrid::Ptr phigrid;
    typename openvdb::Vec3fGrid::Ptr velgrid;
    for (openvdb::GridPtrVec::iterator iter = my_grids->begin(); iter != my_grids->end(); ++iter) {
      if ((*iter)->isType<openvdb::FloatGrid>()) {
        if (openvdb::gridPtrCast<openvdb::FloatGrid>(*iter)->metaValue<std::string>("name") == "surface") {
          phigrid = openvdb::gridPtrCast<openvdb::FloatGrid>(*iter);
          for (openvdb::MetaMap::MetaIterator it = phigrid->beginMeta(); it != phigrid->endMeta();
               ++it) {
            const std::string &name = it->first;
            openvdb::Metadata::Ptr value = it->second;
            std::string valueAsString = value->str();
            std::cout << name << " = " << valueAsString << std::endl;
          }
        }
      } else if ((*iter)->isType<openvdb::Vec3fGrid>()) {
        if (openvdb::gridPtrCast<openvdb::Vec3fGrid>(*iter)->metaValue<std::string>("name") == "vel") {
          velgrid = openvdb::gridPtrCast<openvdb::Vec3fGrid>(*iter);
          for (openvdb::MetaMap::MetaIterator it = velgrid->beginMeta(); it != velgrid->endMeta();
               ++it) {
            const std::string &name = it->first;
            openvdb::Metadata::Ptr value = it->second;
            std::string valueAsString = value->str();
            std::cout << name << " = " << valueAsString << std::endl;
          }
        }
      }
    }

    /// bounding box
    TV bmin, bmax;
    {
      openvdb::CoordBBox box = phigrid->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = phigrid->indexToWorld(box.min());
      auto world_max = phigrid->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        bmin(d) = world_min[d];
        bmax(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = phigrid->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          bmin(d) = pos[d] < bmin(d) ? pos[d] : bmin(d);
          bmax(d) = pos[d] > bmax(d) ? pos[d] : bmax(d);
        }
      }
    }

    vec<int, 3> extents = ((bmax - bmin) / dx).cast<int>() + 1;

    /// phi
    auto sample = [&phigrid, &velgrid, dim](const TV &X_input) -> std::tuple<float, openvdb::Vec3f> {
      TV X = TV::zeros();
      for (int d = 0; d < dim; d++) X(d) = X_input(d);
      openvdb::tools::GridSampler<PhiTreeT, openvdb::tools::BoxSampler> phi_interpolator(
          phigrid->constTree(), phigrid->transform());
      openvdb::tools::GridSampler<VelTreeT, openvdb::tools::BoxSampler> vel_interpolator(
          velgrid->constTree(), velgrid->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = phi_interpolator.wsSample(P);  // ws denotes world space
      auto vel = vel_interpolator.wsSample(P);   // ws denotes world space
      return std::make_tuple((float)phi, vel);
    };
    printf(
        "Vdb file domain [%f, %f, %f] - [%f, %f, %f]; resolution {%d, %d, "
        "%d}\n",
        bmin(0), bmin(1), bmin(2), bmax(0), bmax(1), bmax(2), extents(0), extents(1), extents(2));
    DenseGrid<float, int, 3> phi(extents, 2 * dx);
    DenseGrid<TV, int, 3> vel(extents, TV::zeros());
#pragma omp parallel for
    for (int x = 0; x < extents(0); ++x)
      for (int y = 0; y < extents(1); ++y)
        for (int z = 0; z < extents(2); ++z) {
          IV X = vec<int, 3>{x, y, z};
          TV x = X * dx + bmin;
          auto [p, v] = sample(x);
          phi(X) = p;
          vel(X) = TV{v(0), v(1), v(2)};
        }
    return zs::make_tuple(phi, vel, bmin, bmax);
  }

}  // namespace zs