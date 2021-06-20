#pragma once

#include <zen/zen.h>
#include <vector>

#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <string.h>
namespace zen {


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


struct VDBGrid : zen::IObject {
    virtual void output(std::string path) = 0;
    virtual void input(std::string path) = 0;
    virtual void setTransform(openvdb::math::Transform::Ptr const &trans) = 0;
    virtual std::string getType() {return std::string();}
};



template <typename GridT>
//struct VDBGridWrapper : zen::IObjectClone<VDBGridWrapper<GridT>, VDBGrid> {
struct VDBGridWrapper : VDBGrid {
  typename GridT::Ptr m_grid;

  VDBGridWrapper(){ m_grid = GridT::create(); }

  virtual void output(std::string path) override {
    writeFloatGrid<GridT>(path, m_grid);
  }

  virtual void input(std::string path) override {
    m_grid = readFloatGrid<GridT>(path);
  }

  virtual void setTransform(openvdb::math::Transform::Ptr const &trans) override {
    m_grid->setTransform(trans);
  }

  virtual std::string getType(){
    if(std::is_same<GridT, openvdb::FloatGrid>::value)
    {
      return std::string("FloatGrid");
    }
    else if(std::is_same<GridT, openvdb::Int32Grid>::value)
    {
        return std::string("Int32Grid");
    }
    else if(std::is_same<GridT, openvdb::Vec3fGrid>::value)
    {
      return std::string("Vec3fGrid");
    }
    else if(std::is_same<GridT, openvdb::Vec3IGrid>::value)
    {
      return std::string("Vec3IGrid");
    }
    else if(std::is_same<GridT, openvdb::points::PointDataGrid>::value)
    {
      return std::string("PointDataGrid");
    }
    else {
      return std::string("");
    }
  }
};

struct TBBConcurrentIntArray : zen::IObject {
  tbb::concurrent_vector<openvdb::Index32> m_data;
};


using VDBFloatGrid = VDBGridWrapper<openvdb::FloatGrid>;
using VDBIntGrid = VDBGridWrapper<openvdb::Int32Grid>;
using VDBFloat3Grid = VDBGridWrapper<openvdb::Vec3fGrid>;
using VDBInt3Grid = VDBGridWrapper<openvdb::Vec3IGrid>;
using VDBPointsGrid = VDBGridWrapper<openvdb::points::PointDataGrid>;


}
