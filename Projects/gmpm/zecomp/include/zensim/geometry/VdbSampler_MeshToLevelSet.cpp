#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Mat.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Tuple.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>

#include <algorithm>
#include <numeric>

#include "LevelSet.h"
#include "VdbSampler.h"
#include "zensim/Logger.hpp"

namespace zs {

  openvdb::FloatGrid::Ptr readMeshToLevelset(const std::string &filename, float h) {
    using Vec3ui = openvdb::math::Vec3ui;
    std::vector<openvdb::Vec3f> vertList;
    std::vector<Vec3ui> faceList;
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Failed to open. Terminating.\n";
      exit(-1);
    }

    int ignored_lines = 0;
    std::string line;

    while (!infile.eof()) {
      std::getline(infile, line);
      auto ed = line.find_first_of(" ");
      if (line.substr(0, ed) == std::string("v")) {
        std::stringstream data(line);
        char c;
        openvdb::Vec3f point;
        data >> c >> point[0] >> point[1] >> point[2];
        vertList.push_back(point);
      } else if (line.substr(0, ed) == std::string("f")) {
        std::stringstream data(line);
        char c;
        int v0, v1, v2;
        data >> c >> v0 >> v1 >> v2;
        faceList.push_back(Vec3ui(v0 - 1, v1 - 1, v2 - 1));
      } else {
        ++ignored_lines;
      }
    }
    infile.close();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    points.resize(vertList.size());
    triangles.resize(faceList.size());
    tbb::parallel_for(0, (int)vertList.size(), 1, [&](int p) {
      points[p] = openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
    });
    tbb::parallel_for(0, (int)faceList.size(), 1, [&](int p) {
      triangles[p] = openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
    });
    openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
        *openvdb::math::Transform::createLinearTransform(h), points, triangles, 3.0);
    return grid;
  }

  extern std::vector<std::array<float, 3>> sample_from_levelset(openvdb::FloatGrid::Ptr vdbls,
                                                                float dx, float ppc);

  std::vector<std::array<float, 3>> sample_from_obj_file(const std::string &filename, float dx,
                                                         float ppc) {
    return sample_from_levelset(readMeshToLevelset(filename, dx), dx, ppc);
  }

}  // namespace zs