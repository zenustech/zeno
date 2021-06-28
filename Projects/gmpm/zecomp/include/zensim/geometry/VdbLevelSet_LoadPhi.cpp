#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/memory/MemoryResource.h"

namespace zs {

  OpenVDBStruct loadFloatGridFromVdbFile(const std::string &fn) {
    using GridType = openvdb::FloatGrid;
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

  AdaptiveFloatGrid convertFloatGridToAdaptiveGrid(const OpenVDBStruct &grid,
                                                   const MemoryHandle mh) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using SDFPtr = typename GridType::Ptr;
    const SDFPtr &gridPtr = grid.as<SDFPtr>();
    AdaptiveFloatGrid ret{mh};
    ret.dx = gridPtr->transform().voxelSize()[0];

    return ret;
  }

  SparseLevelSet<3> convertFloatGridToSparseLevelSet(const OpenVDBStruct &grid) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using SDFPtr = typename GridType::Ptr;
    const SDFPtr &gridPtr = grid.as<SDFPtr>();
    using IV = typename SparseLevelSet<3>::table_t::key_t;
    using TV = vec<f32, 3>;

    SparseLevelSet<3> ret{};
    const auto leafCount = gridPtr->tree().leafCount();
    ret._sideLength = 8;
    ret._space = 512;
    ret._dx = gridPtr->transform().voxelSize()[0];
    ret._backgroundValue = gridPtr->background();
    ret._table = typename SparseLevelSet<3>::table_t{leafCount, memsrc_e::host, -1};
    ret._tiles = typename SparseLevelSet<3>::tiles_t{
        {{"sdf", 1}}, leafCount * ret._space, memsrc_e::host, -1};
    {
      openvdb::CoordBBox box = gridPtr->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = gridPtr->indexToWorld(box.min());
      auto world_max = gridPtr->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        ret._min(d) = world_min[d];
        ret._max(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = gridPtr->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          ret._min(d) = pos[d] < ret._min(d) ? pos[d] : ret._min(d);
          ret._max(d) = pos[d] > ret._max(d) ? pos[d] : ret._max(d);
        }
      }
    }
    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();

    gridPtr->transform().print();
    fmt::print("background value: {}. dx: {}. box: [{}, {}, {} ~ {}, {}, {}]\n",
               ret._backgroundValue, ret._dx, ret._min[0], ret._min[1], ret._min[2], ret._max[0],
               ret._max[1], ret._max[2]);

    auto w2v = v2w.inverse();
    vec<float, 4, 4> transform;
    for (auto &&[r, c] : ndrange<2>(4)) transform(r, c) = w2v[r][c];  /// use [] for access
    ret._w2v = transform;

    auto table = proxy<execspace_e::host>(ret._table);
    auto tiles = proxy<execspace_e::host>({"sdf"}, ret._tiles);
    table.clear();
    for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
      const TreeType::LeafNodeType &node = *iter;
      if (node.onVoxelCount() > 0) {
        auto cell = node.beginValueOn();
        IV coord{};
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        auto blockid = coord;
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d)
          blockid[d] += (coord[d] < 0 ? -ret._sideLength + 1 : 0);
        blockid = blockid / ret._sideLength;
        auto blockno = table.insert(blockid);
        int cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          auto sdf = cell.getValue();
          tiles.val("sdf", blockno * ret._space + cellid) = sdf;
          // tiles.template tuple<3>("vel", blockno * ret._space + cellid) = TV::zeros();
        }
      }
    }
    return ret;
  }
  SparseLevelSet<3> convertFloatGridToSparseLevelSet(const OpenVDBStruct &grid,
                                                     const MemoryHandle mh) {
    return convertFloatGridToSparseLevelSet(grid).clone(mh);
  }

  void checkFloatGrid(OpenVDBStruct &grid) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using SDFPtr = typename GridType::Ptr;
    SDFPtr &gridPtr = grid.as<SDFPtr>();
    fmt::print("grid meta -> voxel dx {}, {}, {}\n", gridPtr->transform().voxelSize()[0],
               gridPtr->transform().voxelSize()[1], gridPtr->transform().voxelSize()[2]);
    float dx = gridPtr->transform().voxelSize()[0];
    getchar();
#if 0
    for (GridType::ValueOnIter iter = gridPtr->beginValueOn(); iter.test(); ++iter) {
      // Print the coordinates of all voxels whose vector value has
      // a length greater than -10, and print the bounding box coordinates
      // of all tiles whose vector value length is greater than 10.
      // if (iter.getValue() > -10)
      if (iter.isValueOn()) {
        auto box = iter.getBoundingBox();
        auto st = box.getStart();
        auto ed = box.getEnd();

        fmt::print("iter-> level {}, box ({}, {}, {}) - ({}, {}, {}).\t", iter.getLevel(), st[0],
                   st[1], st[2], ed[0], ed[1], ed[2]);

        switch (iter.getDepth()) {
          case 0: {
            RootType *node = nullptr;
            iter.getNode<RootType>(node);
            if (node) {
              fmt::print("it is a root node");
            }
            break;
          }
          case 1: {
            Int1Type *node = nullptr;
            iter.getNode<Int1Type>(node);
            if (node) {
              fmt::print("it is a depth 1 node");
            }
            break;
          }
          case 2: {
            Int2Type *node = nullptr;
            iter.getNode<Int2Type>(node);
            if (node) {
              fmt::print("it is a depth 2 node");
            }
            break;
          }
          case 3: {
            LeafType *node = nullptr;
            iter.getNode<LeafType>(node);
            if (node) {
            }
            break;
          }
        }
        if (!iter.isValueOn()) {
          fmt::print("not active value???\n");
          getchar();
        }
        if (iter.isVoxelValue()) {
          fmt::print("voxel {}, {}, {}, value {}\n", iter.getCoord()[0], iter.getCoord()[1],
                     iter.getCoord()[2], iter.getValue());
        } else {
          fmt::print("wow, it is a box??? adaptive value {}\n", iter.getValue());
          getchar();
        }
      }
    }
#else
    i64 leafCount = 0;
    for (TreeType::NodeIter iter = gridPtr->tree().beginNode(); iter.test(); ++iter) {
      // Print the coordinates of all voxels whose vector value has
      // a length greater than -10, and print the bounding box coordinates
      // of all tiles whose vector value length is greater than 10.
      // if (iter.getValue() > -10)
      auto box = iter.getBoundingBox();
      auto st = box.getStart();
      auto ed = box.getEnd();
      auto coord = iter.getCoord();

      fmt::print("node iter -> depth{}, level {}, box ({}, {}, {}) - ({}, {}, {}). ",
                 iter.getDepth(), iter.getLevel(), st[0], st[1], st[2], ed[0], ed[1], ed[2]);

      switch (iter.getDepth()) {
        case 0: {
          RootType *node = nullptr;
          iter.getNode<RootType>(node);
          fmt::print("root childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            // getchar();
            for (auto cell = node->beginValueOn(); cell; ++cell) {
              fmt::print("\troot local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
              getchar();
            }
          }
          break;
        }
        case 1: {
          Int1Type *node = nullptr;
          iter.getNode<Int1Type>(node);
          fmt::print("int1 childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            for (int i = 0; i < 32; ++i)
              for (int j = 0; j < 32; ++j)
                for (int k = 0; k < 32; ++k) {
                  if (node->isValueOn(openvdb::Coord(i, j, k)))
                    fmt::print("int1 local xyz child ({}, {}, {}) is active\n", i, j, k);
                }
            for (auto cell = node->beginValueOn(); cell; ++cell) {
              fmt::print("\tint1 local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
              getchar();
            }
          }
          break;
        }
        case 2: {
          Int2Type *node = nullptr;
          iter.getNode<Int2Type>(node);
          fmt::print("int2 childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            for (int i = 0; i < 16; ++i)
              for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k) {
                  if (node->isValueOn(openvdb::Coord(i, j, k)))
                    fmt::print("int2 local xyz child ({}, {}, {}) is active\n", i, j, k);
                }
            for (auto cell = node->beginValueOn(); cell; ++cell) {
              fmt::print("\tint2 local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
              getchar();
            }
          }
          break;
        }
        case 3: {
          leafCount++;
          LeafType *node = nullptr;
          iter.getNode<LeafType>(node);
          fmt::print("leaf childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            for (int i = 0; i < 8; ++i)
              for (int j = 0; j < 8; ++j)
                for (int k = 0; k < 8; ++k) {
                  if (node->isValueOn(openvdb::Coord(i, j, k)))
                    fmt::print("\tleaf local xyz child ({}, {}, {}) is active\n", i, j, k);
                }
            for (auto cell = node->beginValueOn(); cell; ++cell)
              fmt::print("\tleaf local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
            // getchar();
          }
          break;
        }
      }
    }
    fmt::print("check grid: leaf count {}\n", leafCount);
#endif
  }

  // https://www.openvdb.org/documentation/doxygen/codeExamples.html#sGettingMetadata
  tuple<DenseGrid<float, int, 3>, vec<float, 3>, vec<float, 3>> readPhiFromVdbFile(
      const std::string &fn, float dx) {
    constexpr int dim = 3;
    using TV = vec<float, dim>;
    using IV = vec<int, dim>;
    using TreeT = typename openvdb::FloatGrid::TreeType;
    using VelTreeT = typename openvdb::Vec3fGrid::TreeType;

    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr my_grids = file.getGrids();
    file.close();
    int count = 0;
    typename openvdb::FloatGrid::Ptr grid;
    for (openvdb::GridPtrVec::iterator iter = my_grids->begin(); iter != my_grids->end(); ++iter) {
      openvdb::GridBase::Ptr it = *iter;
      if ((*iter)->isType<openvdb::FloatGrid>()) {
        grid = openvdb::gridPtrCast<openvdb::FloatGrid>(*iter);
        count++;
        /// display meta data
        for (openvdb::MetaMap::MetaIterator it = grid->beginMeta(); it != grid->endMeta(); ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      } else if ((*iter)->isType<openvdb::Vec3fGrid>()) {
        auto velgrid = openvdb::gridPtrCast<openvdb::Vec3fGrid>((*iter));
        for (openvdb::MetaMap::MetaIterator it = velgrid->beginMeta(); it != velgrid->endMeta();
             ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      }
    }
    ZS_WARN_IF(count != 1, "Vdb file to load should only contain one levelset.");

    /// bounding box
    TV bmin, bmax;
    {
      openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = grid->indexToWorld(box.min());
      auto world_max = grid->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        bmin(d) = world_min[d];
        bmax(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = grid->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          bmin(d) = pos[d] < bmin(d) ? pos[d] : bmin(d);
          bmax(d) = pos[d] > bmax(d) ? pos[d] : bmax(d);
        }
      }
    }

    vec<int, 3> extents = ((bmax - bmin) / dx).cast<int>() + 1;

    /// phi
    auto sample = [&grid](const TV &X_input) -> float {
      TV X = X_input;
      openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
          grid->constTree(), grid->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = interpolator.wsSample(P);  // ws denotes world space
      return (float)phi;
    };
    printf(
        "Vdb file domain [%f, %f, %f] - [%f, %f, %f]; resolution {%d, %d, "
        "%d}\n",
        bmin(0), bmin(1), bmin(2), bmax(0), bmax(1), bmax(2), extents(0), extents(1), extents(2));
    DenseGrid<float, int, 3> phi(extents, 2 * dx);
#pragma omp parallel for
    for (int x = 0; x < extents(0); ++x)
      for (int y = 0; y < extents(1); ++y)
        for (int z = 0; z < extents(2); ++z) {
          IV X = vec<int, 3>{x, y, z};
          TV x = X * dx + bmin;
          phi(X) = sample(x);
        }
    return zs::make_tuple(phi, bmin, bmax);
  }

}  // namespace zs