#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include <algorithm>
#include <numeric>

#include "LevelSet.h"
#include "VdbSampler.h"
#include "zensim/Logger.hpp"

namespace zs {

  openvdb::FloatGrid::Ptr readFileToLevelset(const std::string &fn) {
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
      }
    }
    ZS_WARN_IF(count != 1, "Vdb file to load should only contain one levelset.");
    return grid;
  }

  extern std::vector<std::array<float, 3>> sample_from_levelset(openvdb::FloatGrid::Ptr vdbls,
                                                                float dx, float ppc);

  std::vector<std::array<float, 3>> sample_from_vdb_file(const std::string &filename, float dx,
                                                         float ppc) {
    return sample_from_levelset(readFileToLevelset(filename), dx, ppc);
  }

}  // namespace zs