#include "Structures.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Bht.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/graph/Coloring.hpp"
#include "zensim/graph/ConnectedComponents.hpp"
#include "zensim/math/matrix/SparseMatrixOperations.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/ListObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

struct SpawnGuidelines : INode {
    virtual void apply() override {
        using bvh_t = zs::LBvh<3, int, float>;
        using bv_t = typename bvh_t::Box;

        auto points = get_input<PrimitiveObject>("points");
        auto nrmAttr = get_input2<std::string>("normalTag");
        auto length = get_input2<float>("length");
        auto numSegments = get_input2<int>("segments");

	if (numSegments < 1) {
		throw std::runtime_error("the number of segments must be positive");
	}

	auto numLines = points->verts.size();
        auto const &roots = points->attr<vec3f>("pos");
        auto const &nrm = points->attr<vec3f>(nrmAttr);

        using namespace zs;
        auto pol = omp_exec();
        constexpr auto space = execspace_e::openmp;

	auto prim = std::make_shared<PrimitiveObject>();
	prim->verts.resize(numLines * (numSegments + 1));
	prim->loops.resize(numLines * (numSegments + 1));
	prim->polys.resize(numLines);

	auto &pos = prim->attr<vec3f>("pos");
        pol(enumerate(prim->polys.values),
            [&roots, &nrm, &pos, numSegments, segLength = length / numSegments](int polyi, vec2i &tup) {
                auto offset = polyi * (numSegments + 1);
                tup[0] = offset;
                tup[1] = (numSegments + 1);

                auto rt = roots[polyi];
                pos[offset] = rt;
                auto step = nrm[polyi] * segLength;
                for (int i = 0; i != numSegments; ++i) {
                    rt += step;
                    pos[++offset] = rt;
                }
            });
	pol(enumerate(prim->loops.values), [](int vi, int &loopid) {
		loopid = vi;
	});
	// copy point attrs to polys attrs
        
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(SpawnGuidelines, {
                                 {
                                     {"PrimitiveObject", "points"},
                                     {"string", "normalTag", "nrm"},
                                     {"float", "length", "0.5"},
                                     {"int", "segments", "5"},
                                 },
                                 {
                                     {"PrimitiveObject", "prim"},
                                 },
                                 {},
                                 {"zs_hair"},
                             });

}