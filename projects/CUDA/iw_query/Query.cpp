#include <zeno/core/INode.h>
#include <zeno/core/defNode.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include "bvh.h"
#include "distanceQueries.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/resource/Filesystem.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <chrono>
#include <zeno/utils/log.h>

namespace zeno {

///
/// ref: https://github.com/ingowald/closestSurfacePointQueries
///
struct QueryNearestPoints : INode {
    void apply() override {
        auto points = get_input<PrimitiveObject>("points");
        auto prim = get_input<PrimitiveObject>("target_prim");
        auto idTag = get_input2<std::string>("idTag");
        auto distTag = get_input2<std::string>("distTag");
        auto cpTag = get_input2<std::string>("closestPointTag");

        using v3f = ospcommon::vec3f;
        using v3i = ospcommon::vec3i;

        static_assert(sizeof(v3f) == sizeof(zeno::vec3f), "v3f, vec3f size mismatch!");
        static_assert(sizeof(v3i) == sizeof(zeno::vec3i), "v3i, vec3i size mismatch!");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &pos = points->attr<zeno::vec3f>("pos");
        auto &ids = points->add_attr<int>(idTag);
        auto &dists = points->add_attr<float>(distTag);
        auto &cps = points->add_attr<zeno::vec3f>(cpTag);

        std::vector<v3i> vertIndex(prim->size());
        auto &vertices = prim->verts.values;
        pol(enumerate(vertIndex), [](int id, v3i &v) { v = v3i{id, id, id}; });

        ///
        /// begin routine
        ///
        auto begin = std::chrono::system_clock::now();
        // create the actual scene:
        distance_query_scene scene =
            rtdqNewTriangleMeshfi(&vertices[0][0], &vertices[0][1], &vertices[0][2], 3, &vertIndex[0].x,
                                  &vertIndex[0].y, &vertIndex[0].z, 3, vertIndex.size());
        auto done_build = std::chrono::system_clock::now();

        // perform the queries - all together, in a single thread
        rtdqComputeClosestPointsfi(scene, &cps[0][0], &cps[0][1], &cps[0][2], 3, &dists[0], 1, &ids[0], 1, &pos[0][0],
                                   &pos[0][1], &pos[0][2], 3, points->size());
        auto done_all = std::chrono::system_clock::now();

        std::chrono::duration<double> buildTime = done_build - begin;
        std::chrono::duration<double> queryTime = done_all - done_build;
        std::cout << "time to build tree " << buildTime.count() << "s" << std::endl;
        std::cout << "time to query " << points->size() << " points: " << queryTime.count() << "s" << std::endl;
        std::cout << "(this is " << (queryTime.count() / points->size()) << " seconds/prim)" << std::endl;

        rtdqDestroy(scene);
        ///
        /// done routine
        ///

        set_output("points", points);
    }
};

ZENDEFNODE(QueryNearestPoints, {/* inputs: */
                                {
                                    {"PrimitiveObject", "points", ""},
                                    {"PrimitiveObject", "target_prim", ""},
                                    {"string", "idTag", "bvh_id"},
                                    {"string", "distTag", "bvh_dist"},
                                    {"string", "closestPointTag", "cp"},
                                },
                                /* outputs: */
                                {{"PrimitiveObject", "points", ""}},
                                /* params: */
                                {},
                                /* category: */
                                {"query"}});

} // namespace zeno