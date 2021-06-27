#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Mat.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Tuple.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/util/Util.h>

#include <algorithm>
#include <numeric>

#include "PoissonDisk.hpp"
#include "VdbSampler.h"
#include "zensim/Logger.hpp"
#include "zensim/types/Iterator.h"

namespace zs {

  void initialize_openvdb() { openvdb::initialize(); }

  std::vector<std::array<float, 3>> sample_from_levelset(openvdb::FloatGrid::Ptr vdbls, float dx,
                                                         float ppc) {
    using TV = vec<float, 3>;
    using STDV3 = std::array<float, 3>;
    TV minCorner, maxCorner;
    {
      openvdb::CoordBBox box = vdbls->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = vdbls->indexToWorld(box.min());
      auto world_max = vdbls->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        minCorner(d) = world_min[d];
        maxCorner(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = vdbls->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          minCorner(d) = pos[d] < minCorner(d) ? pos[d] : minCorner(d);
          maxCorner(d) = pos[d] > maxCorner(d) ? pos[d] : maxCorner(d);
        }
      }
    }

    PoissonDisk<float, 3> pd{};
    pd.minCorner = minCorner;
    pd.maxCorner = maxCorner;
    pd.setDistanceByPpc(dx, ppc);

    auto sample = [&vdbls](const TV &X_input) -> float {
      TV X = X_input;
      openvdb::tools::GridSampler<typename openvdb::FloatGrid::TreeType, openvdb::tools::BoxSampler>
          interpolator(vdbls->constTree(), vdbls->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = interpolator.wsSample(P);  // ws denotes world space
      return (float)phi;
    };

    std::vector<STDV3> samples = pd.sample([&](const TV &x) { return sample(x) <= 0.f; });
    return samples;
  }

}  // namespace zs