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
#include "zensim/zpc_tpls/fmt/format.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/VolumeToSpheres.h>
#include <openvdb/tree/LeafManager.h>
#include <zeno/ListObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

struct SpawnGuidelines : INode {
    virtual void apply() override {
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
        pol(enumerate(prim->loops.values), [](int vi, int &loopid) { loopid = vi; });
        // copy point attrs to polys attrs
        for (auto &[key, srcArr] : points->verts.attrs) {
            auto const &k = key;
            match(
                [&k, &prim](
                    auto &srcArr) -> std::enable_if_t<variant_contains<RM_CVREF_T(srcArr[0]), AttrAcceptAll>::value> {
                    using T = RM_CVREF_T(srcArr[0]);
                    prim->polys.add_attr<T>(k);
                },
                [](...) {})(srcArr);
        }
        pol(range(points->size()), [&](int vi) {
            {
                for (auto &[key, srcArr] : points->verts.attrs) {
                    auto const &k = key;
                    match(
                        [&k, &prim, vi](auto &srcArr)
                            -> std::enable_if_t<variant_contains<RM_CVREF_T(srcArr[0]), AttrAcceptAll>::value> {
                            using T = RM_CVREF_T(srcArr[0]);
                            auto &arr = prim->polys.attr<T>(k);
                            arr[vi] = srcArr[vi];
                        },
                        [](...) {})(srcArr);
                }
            }
        });

        set_output("guide_lines", std::move(prim));
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
                                    {"PrimitiveObject", "guide_lines"},
                                },
                                {},
                                {"zs_hair"},
                            });

// after guideline simulation, mostly for rendering
struct GenerateHairs : INode {
    virtual void apply() override {
        using bvh_t = zs::LBvh<3, int, float>;
        using bv_t = typename bvh_t::Box;

        auto points = get_input<PrimitiveObject>("points");
        auto guideLines = get_input<PrimitiveObject>("guide_lines");
        bool interpAttrs = get_input2<bool>("interpAttrs");

        using namespace zs;
        auto pol = omp_exec();
        constexpr auto space = execspace_e::openmp;

        bvh_t bvh;
        {
            zs::Vector<bv_t> bvs{guideLines->polys.size()};
            pol(range(guideLines->polys.size()), [&verts = guideLines->verts.values, &polys = guideLines->polys.values,
                                                  &loops = guideLines->loops.values, &bvs](int polyI) {
                auto p = vec_to_other<zs::vec<float, 3>>(verts[loops[polys[polyI][0]]]);
                bvs[polyI] = bv_t{p, p};
            });
            bvh.build(pol, bvs);
        }
        auto numHairs = points->verts.size();
        std::vector<int> gid(numHairs);
        std::vector<int> numPoints(numHairs), hairPolyOffsets(numHairs);
        pol(range(numHairs), [&gid, &points, &verts = guideLines->verts.values, &polys = guideLines->polys.values,
                              &loops = guideLines->loops.values, &numPoints, lbvhv = proxy<space>(bvh)](int vi) {
            using vec3 = zs::vec<float, 3>;
            auto pi = vec3::from_array(points->verts.values[vi]);
            auto [id, _] = lbvhv.find_nearest(
                pi,
                [&](int j, float &dist, int &id) {
                    float d = zs::limits<float>::max();
                    d = zs::dist_pp(pi, vec3::from_array(verts[loops[polys[j][0]]]));

                    if (d < dist) {
                        dist = d;
                        id = j;
                    }
                },
                true_c);
            gid[vi] = id;
            numPoints[vi] = polys[id][1];
        });
        exclusive_scan(pol, std::begin(numPoints), std::end(numPoints), std::begin(hairPolyOffsets));

        /// hairs
        auto prim = std::make_shared<PrimitiveObject>();
        auto numTotalPoints = hairPolyOffsets.back() + numPoints.back();
        prim->verts.resize(numTotalPoints);
        prim->loops.resize(numTotalPoints);
        prim->polys.resize(numHairs);

        pol(enumerate(prim->polys.values),
            [&numPoints, &hairPolyOffsets, &gid, &pos = prim->attr<vec3f>("pos"), &points,
             &glPos = guideLines->attr<vec3f>("pos"), &polys = guideLines->polys.values,
             &loops = guideLines->loops.values](int hairId, vec2i &tup) {
                auto offset = hairPolyOffsets[hairId];
                auto numPts = numPoints[hairId];
                tup[0] = offset;
                tup[1] = numPts;

                auto glId = gid[hairId];
                auto glOffset = polys[glId][0];
                auto getGlVert = [&](int i) { return glPos[loops[glOffset + i]]; };

                auto lastHairVert = points->verts.values[hairId];
                auto lastGlVert = getGlVert(0);
                auto curGlVert = getGlVert(1);
                pos[offset] = lastHairVert;
                for (int i = 1; i != numPts; ++i) {
                    curGlVert = getGlVert(i);
                    lastHairVert += curGlVert - lastGlVert;
                    pos[++offset] = lastHairVert;
                    lastGlVert = curGlVert;
                }
            });
        pol(enumerate(prim->loops.values), [](int vi, int &loopid) { loopid = vi; });

        /// @note override guideline attribs to hairs
        if (get_input2<bool>("interpAttrs")) {
            for (auto &[key, srcArr] : guideLines->polys.attrs) {
                auto const &k = key;
                match(
                    [&k, &prim](auto &srcArr)
                        -> std::enable_if_t<variant_contains<RM_CVREF_T(srcArr[0]), AttrAcceptAll>::value> {
                        using T = RM_CVREF_T(srcArr[0]);
                        prim->polys.add_attr<T>(k);
                    },
                    [](...) {})(srcArr);
            }
            pol(range(prim->polys.size()), [&](int hairId) {
                for (auto &[key, srcArr] : guideLines->polys.attrs) {
                    auto const &k = key;
                    match(
                        [&k, &prim, &gid, hairId](auto &srcArr)
                            -> std::enable_if_t<variant_contains<RM_CVREF_T(srcArr[0]), AttrAcceptAll>::value> {
                            using T = RM_CVREF_T(srcArr[0]);
                            auto &arr = prim->polys.attr<T>(k);
                            arr[hairId] = srcArr[gid[hairId]]; // assign the value same as its guideline
                        },
                        [](...) {})(srcArr);
                }
            });
        }

        set_output("prim", prim);
    }
};

ZENDEFNODE(GenerateHairs,
           {
               {{"PrimitiveObject", "points"}, {"PrimitiveObject", "guide_lines"}, {"bool", "interpAttrs", "1"}},
               {
                   {"PrimitiveObject", "prim"},
               },
               {},
               {"zs_hair"},
           });

} // namespace zeno

struct RepelPoints : zeno::INode {
    virtual void apply() override {
        auto points = get_input<zeno::PrimitiveObject>("points");
        auto collider = get_input2<zeno::VDBFloatGrid>("vdb_collider");
        auto sep_dist = get_input2<float>("sep_dist");
        auto maxIters = get_input2<int>("max_iter");
        auto grid = collider->m_grid;
        grid->tree().voxelizeActiveTiles();
        auto dx = collider->getVoxelSize()[0];
        openvdb::Vec3fGrid::Ptr gridGrad = openvdb::tools::gradient(*grid);

        using namespace zs;
        auto pol = omp_exec();
        constexpr auto space = execspace_e::openmp;
        auto &pos = points->attr<zeno::vec3f>("pos");
        pol(range(pos.size()), [&grid, &gridGrad, &pos, dx, sep_dist, maxIters](int vi) {
            // auto p = vec_to_other<openvdb::Vec3R>(pos[vi]);
            auto p = zeno::vec_to_other<openvdb::Vec3R>(pos[vi]);
            auto mi = maxIters;
            for (auto sdf = openvdb::tools::BoxSampler::sample(grid->getConstUnsafeAccessor(), grid->worldToIndex(p));
                 sdf < sep_dist && mi-- > 0;
                 sdf = openvdb::tools::BoxSampler::sample(grid->getConstUnsafeAccessor(), grid->worldToIndex(p))) {

                auto ddd =
                    openvdb::tools::BoxSampler::sample(gridGrad->getConstUnsafeAccessor(), gridGrad->worldToIndex(p));

                // fmt::print("pt[{}] current at <{}, {}, {}>, sdf: {}, direction <{}, {}, {}>\n", vi, p[0], p[1], p[2], sdf, ddd[0], ddd[1], ddd[2]);

                // p += ddd.normalize() * dx;
                p += ddd.normalize() * -sdf;
                // p += ddd;
            }
            pos[vi] = zeno::other_to_vec<3>(p);
        });
        set_output("prim", points);
    }
};

ZENDEFNODE(RepelPoints, {
                            {
                                {"PrimitiveObject", "points"},
                                {"vdb_collider"},
                                {"float", "sep_dist", "0"},
                                {"int", "max_iter", "100"},
                            },
                            {
                                {"PrimitiveObject", "prim"},
                            },
                            {},
                            {"zs_hair"},
                        });
