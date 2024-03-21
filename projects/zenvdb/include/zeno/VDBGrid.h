#pragma once

#include <optional>
#include <vector>
#include <zeno/zeno.h>

#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/openvdb.h>
#include <string.h>
#include "packed3grids.h"
namespace zeno {

template <typename GridT>
typename GridT::Ptr readFloatGrid(const std::string &fn) {
  openvdb::io::File file(fn);
  file.open();
  openvdb::GridPtrVecPtr my_grids = file.getGrids();
  file.close();
  int count = 0;
  typename GridT::Ptr grid;
  for (openvdb::GridPtrVec::iterator iter = my_grids->begin();
       iter != my_grids->end(); ++iter) {
    openvdb::GridBase::Ptr it = *iter;
    if ((*iter)->isType<GridT>()) {
      grid = openvdb::gridPtrCast<GridT>(*iter);
      count++;
      /// display meta data
      for (openvdb::MetaMap::MetaIterator it = grid->beginMeta();
           it != grid->endMeta(); ++it) {
        const std::string &name = it->first;
        openvdb::Metadata::Ptr value = it->second;
        std::string valueAsString = value->str();
        std::cout << name << " = " << valueAsString << std::endl;
      }
    }
  }
  std::cout << "count = " << count << std::endl;
  return grid;
}

template <typename GridT>
void writeFloatGrid(const std::string &fn, typename GridT::Ptr grid) {
  openvdb::io::File(fn).write({grid});
}

// match([](auto &gridPtr) {...})(someGeneralVdbGrid);

struct VDBGrid : zeno::IObject {
  virtual void output(std::string path) = 0;
  virtual void input(std::string path) = 0;
  
  virtual const openvdb::math::Transform& getTransform() = 0;
  virtual void setTransform(openvdb::math::Transform::Ptr const &trans) = 0;
  virtual std::string method_node(std::string const &op) override {
      if (op == "view") {
          return "INTERN_PreViewVDB";
      }
      return {};
  }

  // using GeneralVdbGrid = variant<typename SomeGrid::Ptr, >;
  // virtual GeneralVdbGrid getGrid() = 0;
  virtual openvdb::CoordBBox evalActiveVoxelBoundingBox() = 0;
  virtual openvdb::Vec3d indexToWorld(openvdb::Coord &c) = 0;
  virtual openvdb::Vec3d worldToIndex(openvdb::Vec3d &c) = 0;
  virtual void setName(std::string const &name) = 0;
  virtual void setGridClass(std::string const &gridClass) = 0;
  virtual std::string getType() const =0;
  virtual zeno::vec3f getVoxelSize() const=0;
  virtual void dilateTopo(int l) =0;

  virtual ~VDBGrid() override = default;
};

template <typename GridT>
struct VDBGridWrapper : zeno::IObjectClone<VDBGridWrapper<GridT>, VDBGrid> {
  typename GridT::Ptr m_grid;

  virtual ~VDBGridWrapper() override = default;

  VDBGridWrapper() { m_grid = GridT::create(); }

  VDBGridWrapper(typename GridT::Ptr &&ptr) { m_grid = std::move(ptr); }

  VDBGridWrapper(VDBGridWrapper const &other) {
      if (other.m_grid)
          m_grid = other.m_grid->deepCopy();
  }

  VDBGridWrapper &operator=(VDBGridWrapper const &other) {
      if (other.m_grid)
          m_grid = other.m_grid->deepCopy();
      else
          m_grid = nullptr;
      return *this;
  }

  // using VDBGrid::GeneralVdbGrid;
  // GeneralVdbGrid getGrid() override {
  //   return m_grid;
  // }

  openvdb::CoordBBox evalActiveVoxelBoundingBox() override {
    return m_grid->evalActiveVoxelBoundingBox();
  }
  openvdb::Vec3d indexToWorld(openvdb::Coord &c) override {
    return m_grid->transform().indexToWorld(c);
  }
  openvdb::Vec3d worldToIndex(openvdb::Vec3d &c) override {
    return m_grid->transform().worldToIndex(c);
  }
  virtual void output(std::string path) override {
    //writeFloatGrid<GridT>(path, m_grid);
    openvdb::io::File(path).write({ m_grid });
  }

  virtual void input(std::string path) override {
    m_grid = readFloatGrid<GridT>(path);
  }

  virtual const openvdb::math::Transform& getTransform() override {
    return m_grid->transform();
  }

  virtual void
  setTransform(openvdb::math::Transform::Ptr const &trans) override {
    m_grid->setTransform(trans);
  }
  virtual void
  dilateTopo(int l) override {
    openvdb::tools::dilateActiveValues(
      m_grid->tree(), l,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
  }

  virtual zeno::vec3f getVoxelSize() const override {
      auto del = m_grid->voxelSize();
      return zeno::vec3f(del[0], del[1], del[2]);
  }

  virtual void setName(std::string const &name) override {
      m_grid->setName(name);
  }

  virtual void setGridClass(std::string const &gridClass) override {
      if (gridClass == "UNKNOWN")
          m_grid->setGridClass(openvdb::GridClass::GRID_UNKNOWN);
      else if (gridClass == "LEVEL_SET")
          m_grid->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
      else if (gridClass == "FOG_VOLUME")
          m_grid->setGridClass(openvdb::GridClass::GRID_FOG_VOLUME);
      else if (gridClass == "STAGGERED")
          m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
  }

  virtual std::string getType() const override {
    if (std::is_same<GridT, openvdb::FloatGrid>::value) {
      return std::string("FloatGrid");
    } else if (std::is_same<GridT, openvdb::Int32Grid>::value) {
      return std::string("Int32Grid");
    } else if (std::is_same<GridT, openvdb::Vec3fGrid>::value) {
      return std::string("Vec3fGrid");
    } else if (std::is_same<GridT, openvdb::Vec3IGrid>::value) {
      return std::string("Vec3IGrid");
    } else if (std::is_same<GridT, openvdb::points::PointDataGrid>::value) {
      return std::string("PointDataGrid");
    } else {
      return std::string("");
    }
  }
};

template <>
struct VDBGridWrapper<openvdb::Vec3fGrid> : zeno::IObjectClone<VDBGridWrapper<openvdb::Vec3fGrid>, VDBGrid> {
  using GridT = openvdb::Vec3fGrid;

  typename GridT::Ptr m_grid;

  ///
  std::optional<packed_FloatGrid3> m_packedGrid;

  bool hasPackedGrid() const noexcept {
    return m_packedGrid.has_value();
  }
  decltype(auto) refPackedGrid() {
    if (!hasPackedGrid()) throw std::runtime_error("packed version of vec3fgrid is not initialized!");
    return *m_packedGrid;
  }
  decltype(auto) refPackedGrid() const {
    if (!hasPackedGrid()) throw std::runtime_error("packed version of vec3fgrid is not initialized!");
    return *m_packedGrid;
  }
  ///

  virtual ~VDBGridWrapper() override = default;

  VDBGridWrapper() { m_grid = GridT::create(); }

  VDBGridWrapper(typename GridT::Ptr &&ptr) { m_grid = std::move(ptr); }

  VDBGridWrapper(VDBGridWrapper const &other) {
      if (other.m_grid)
          m_grid = other.m_grid->deepCopy();
  }

  VDBGridWrapper &operator=(VDBGridWrapper const &other) {
      if (other.m_grid) {
          m_grid = other.m_grid->deepCopy();
          if (other.hasPackedGrid())
            m_packedGrid = other.refPackedGrid().fullCopy();
      } else {
          m_grid = nullptr;
          m_packedGrid = {};
      }
      return *this;
  }

  // using VDBGrid::GeneralVdbGrid;
  // GeneralVdbGrid getGrid() override {
  //   return m_grid;
  // }

  openvdb::CoordBBox evalActiveVoxelBoundingBox() override {
    return m_grid->evalActiveVoxelBoundingBox();
  }
  openvdb::Vec3d indexToWorld(openvdb::Coord &c) override {
    return m_grid->transform().indexToWorld(c);
  }
  openvdb::Vec3d worldToIndex(openvdb::Vec3d &c) override {
    return m_grid->transform().worldToIndex(c);
  }
  virtual void output(std::string path) override {
    //writeFloatGrid<GridT>(path, m_grid);
    openvdb::io::File(path).write({ m_grid });
    // refPackedGrid().to_vec3(m_grid);
  }

  virtual void input(std::string path) override {
    m_grid = readFloatGrid<GridT>(path);
    m_packedGrid = packed_FloatGrid3{};
    auto &packed = refPackedGrid();
    packed.from_vec3(m_grid);
  }

  virtual const openvdb::math::Transform& getTransform() override {
    return m_grid->transform();
  }

  virtual void
  setTransform(openvdb::math::Transform::Ptr const &trans) override {
    m_grid->setTransform(trans);
  }
  virtual void
  dilateTopo(int l) override {
    openvdb::tools::dilateActiveValues(
      m_grid->tree(), l,
      openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
      if (hasPackedGrid()) {
        auto &packed = refPackedGrid();
        packed.from_vec3(m_grid);
      }
  }

  virtual zeno::vec3f getVoxelSize() const override {
      auto del = m_grid->voxelSize();
      return zeno::vec3f(del[0], del[1], del[2]);
  }

  virtual void setName(std::string const &name) override {
      m_grid->setName(name);
  }

  virtual void setGridClass(std::string const &gridClass) override {
      if (gridClass == "UNKNOWN")
          m_grid->setGridClass(openvdb::GridClass::GRID_UNKNOWN);
      else if (gridClass == "LEVEL_SET")
          m_grid->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
      else if (gridClass == "FOG_VOLUME")
          m_grid->setGridClass(openvdb::GridClass::GRID_FOG_VOLUME);
      else if (gridClass == "STAGGERED")
          m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
  }

  virtual std::string getType() const override {
    if (std::is_same<GridT, openvdb::FloatGrid>::value) {
      return std::string("FloatGrid");
    } else if (std::is_same<GridT, openvdb::Int32Grid>::value) {
      return std::string("Int32Grid");
    } else if (std::is_same<GridT, openvdb::Vec3fGrid>::value) {
      return std::string("Vec3fGrid");
    } else if (std::is_same<GridT, openvdb::Vec3IGrid>::value) {
      return std::string("Vec3IGrid");
    } else if (std::is_same<GridT, openvdb::points::PointDataGrid>::value) {
      return std::string("PointDataGrid");
    } else {
      return std::string("");
    }
  }
};

struct TBBConcurrentIntArray : zeno::IObject {
  tbb::concurrent_vector<openvdb::Index32> m_data;
};

using VDBFloatGrid = VDBGridWrapper<openvdb::FloatGrid>;
using VDBIntGrid = VDBGridWrapper<openvdb::Int32Grid>;
using VDBFloat3Grid = VDBGridWrapper<openvdb::Vec3fGrid>;
using VDBInt3Grid = VDBGridWrapper<openvdb::Vec3IGrid>;
using VDBPointsGrid = VDBGridWrapper<openvdb::points::PointDataGrid>;

} // namespace zeno
