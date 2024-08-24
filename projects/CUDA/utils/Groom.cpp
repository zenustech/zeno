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
        if (has_input("vdb_collider")) {
            auto collider = get_input2<zeno::VDBFloatGrid>("vdb_collider");
            auto sep_dist = get_input2<float>("sep_dist");
            auto maxIters = get_input2<int>("max_iter");
            auto grid = collider->m_grid;
            grid->tree().voxelizeActiveTiles();
            auto dx = collider->getVoxelSize()[0];
            openvdb::Vec3fGrid::Ptr gridGrad = openvdb::tools::gradient(*grid);
            auto getSdf = [&grid](openvdb::Vec3R p) {
                return openvdb::tools::BoxSampler::sample(grid->getConstUnsafeAccessor(), grid->worldToIndex(p));
            };
            auto getGrad = [&gridGrad](openvdb::Vec3R p) {
                return openvdb::tools::BoxSampler::sample(gridGrad->getConstUnsafeAccessor(),
                                                          gridGrad->worldToIndex(p));
            };
            pol(enumerate(prim->polys.values),
                /// @note cap [sep_dist] in case repel points outside the narrowband where grad is invalid
                [&getSdf, &getGrad, dx, sep_dist = std::min(sep_dist, grid->background()), maxIters, &roots, &nrm, &pos,
                 numSegments, segLength = length / numSegments](int polyi, vec2i &tup) {
                    auto offset = polyi * (numSegments + 1);
                    tup[0] = offset;
                    tup[1] = (numSegments + 1);

                    auto rt = roots[polyi];
                    pos[offset] = rt;

                    auto p = zeno::vec_to_other<openvdb::Vec3R>(rt);
                    auto lastStep = zeno::vec_to_other<openvdb::Vec3R>(nrm[polyi] * segLength);
                    auto prevPos = p;
                    for (int i = 0; i != numSegments; ++i) {
                        p += lastStep;
                        auto mi = maxIters;
                        int cnt = 0;
                        for (auto sdf = getSdf(p); sdf < sep_dist && mi-- > 0; sdf = getSdf(p)) {
                            auto d = getGrad(p);
                            d.normalize();
                            if (sdf < 0)
                                p += d * -sdf;
                            // p += d * dx;
                            else
                                p += d * (sep_dist - sdf);
                            // p += d * -dx;
                        }
#if 0
                        auto fwd = lastStep = p - prevPos;
                        if (std::abs(fwd.length() - segLength) > segLength * 0.001) {
                            auto g = getGrad(p);
                            auto side = g.cross(fwd);
                            g = g.cross(side);
                            if (g.normalize() && fwd.normalize()) {
                                auto yita = std::acos(fwd.dot(g));
                                auto theta = zs::g_pi - yita;
                                if (yita > zs::g_half_pi)
                                    p += segLength / std::tan(theta) * g;
                                else
                                    p += segLength * std::tan(theta) * g;
                            }
                        }
#endif

                        lastStep = (p - prevPos);
                        lastStep.normalize();
                        lastStep *= segLength;
                        prevPos = p;
                        pos[++offset] = zeno::other_to_vec<3>(p);
                    }
                });
        } else {
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
        }
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
                                    {gParamType_Primitive, "points"},
                                    {gParamType_String, "normalTag", "nrm"},
                                    {gParamType_Float, "length", "0.5"},
                                    {gParamType_Int, "segments", "5"},
                                    {"vdb_collider"},
                                    {gParamType_Float, "sep_dist", "0"},
                                    {gParamType_Int, "max_iter", "100"},
                                },
                                {
                                    {gParamType_Primitive, "guide_lines"},
                                },
                                {},
                                {"zs_hair"},
                            });

struct StepGuidelines : INode {
    virtual void apply() override {
        using namespace zs;
        using V3 = zs::vec<float, 3>;
        auto pol = omp_exec();
        constexpr auto space = execspace_e::openmp;

        auto gls = get_input<PrimitiveObject>("guide_lines");
        auto dt = get_input2<float>("dt");
        auto &pos = gls->attr<vec3f>("pos");
        auto &vel = gls->attr<vec3f>("vel");
        const auto &polys = gls->polys;
        auto &loops = gls->loops;

        auto numLines = polys.size();

        {
            auto weightTag = get_input2<std::string>("weightTag");
            auto idTag = get_input2<std::string>("idTag");
            auto &ws = gls->attr<vec3f>(weightTag);
            auto &ids = gls->attr<float>(idTag); // ref: pnbvhw.cpp
            const auto boundaryPrim = get_input<PrimitiveObject>("boundary_prim");
            const auto &boundaryPos = boundaryPrim->attr<vec3f>("pos");
            const auto &boundaryTris = boundaryPrim->tris.values;
            /// move guideline roots
            pol(polys, [&](vec2i poly) {
                auto ptNo = loops[poly[0]];
                auto tri = boundaryTris[(int)ids[ptNo]];
                auto w = ws[ptNo];
                pos[ptNo] = w[0] * boundaryPos[tri[0]] + w[1] * boundaryPos[tri[1]] + w[2] * boundaryPos[tri[2]];
            });
        }

        /// setup solver
        std::vector<float> grads;
        zs::TileVector<float, 32> vtemp{
            {
                // linear solve
                {"grad", 3},
                {"dir", 3},
                {"temp", 3},
                {"r", 3},
                {"p", 3},
                {"q", 3},
                // system state
                {"xn", 3},
                {"vn", 3},
                {"x0", 3}, // original model positions
                {"xtilde", 3},
                {"xhat", 3}, // initial positions at the current substep (constraint, extforce, velocity update)
            },
            pos.size()},
            etemp{{{"K", 9}}, loops.size()};

        /// @note physics properties
        constexpr float mass = 1.f;
        constexpr float vol = 200.f;
        constexpr float k = 1.e5f;
        constexpr V3 grav{0, -9.8, 0};

        // collider
        openvdb::FloatGrid::Ptr grid;
        openvdb::Vec3fGrid::Ptr gridGrad;
        std::function<void()> resolveCollision;
        bool hasCollider = has_input("vdb_collider");
        if (hasCollider) {
            auto collider = get_input2<zeno::VDBFloatGrid>("vdb_collider");
            grid = collider->m_grid;
            grid->tree().voxelizeActiveTiles();
            gridGrad = openvdb::tools::gradient(*grid);
        }
        auto getSdf = [&grid](openvdb::Vec3R p) {
            return openvdb::tools::BoxSampler::sample(grid->getConstUnsafeAccessor(), grid->worldToIndex(p));
        };
        auto getGrad = [&gridGrad](openvdb::Vec3R p) {
            return openvdb::tools::BoxSampler::sample(gridGrad->getConstUnsafeAccessor(), gridGrad->worldToIndex(p));
        };
        if (hasCollider) {
            auto sep_dist = get_input2<float>("sep_dist");
            auto maxIters = get_input2<int>("collision_iters");

            resolveCollision = [&pol, &loops, &grid, &gridGrad, &getSdf, &getGrad, &vtemp, dx = grid->voxelSize()[0],
                                sep_dist, maxIters, mass, space_c = wrapv<space>{}]() {
                constexpr auto space = RM_CVREF_T(space_c)::value;

                /// @note cap [sep_dist] in case repel points outside the narrowband where grad is invalid
                pol(loops.values, [&getSdf, &getGrad, dx, sep_dist = std::min(sep_dist, grid->background()), maxIters,
                                   vtemp = proxy<space>(vtemp), xnOffset = vtemp.getPropertyOffset("xn"),
                                   gradOffset = vtemp.getPropertyOffset("grad"), mass](int ptNo) {
                    auto p_ = vtemp.pack(dim_c<3>, xnOffset, ptNo);
                    auto p = openvdb::Vec3R(p_[0], p_[1], p_[2]);
#if 0
                    auto mi = maxIters;
                    for (auto sdf = getSdf(p); sdf < sep_dist && mi-- > 0; sdf = getSdf(p)) {
                        auto d = getGrad(p);
                        d.normalize();
                        if (sdf < 0)
                            // p += d * -sdf;
                            p += d * dx;
                        else
                            //p += d * (sep_dist - sdf);
                            p += d * -dx;
                    }
                    vtemp.tuple(dim_c<3>, xnOffset, ptNo) = V3{p[0], p[1], p[2]};
#else
                    auto sdf = getSdf(p);
                    auto mi = maxIters;
                    auto p0 = p;
                    for (auto sdf = getSdf(p); sdf < 0 && mi-- > 0; sdf = getSdf(p)) {
                        auto d = getGrad(p);
                        d.normalize();
                        if (sdf < 0)
                            p += d * dx;
                    }
#if 0
                    if (sdf < 0) {
                        auto d = getGrad(p);
                        d.normalize();
                        p += d * -sdf;
                    }
#endif
                    auto d_ = p - p0;
                    auto delta = V3{d_[0], d_[1], d_[2]} * 0.1;
                    vtemp.tuple(dim_c<3>, gradOffset, ptNo) = vtemp.pack(dim_c<3>, gradOffset, ptNo) + mass * delta;
#endif
                    // auto delta = (vtemp.pack(dim_c<3>, "grad", ptNo)) / mass;
                    // auto xnp1 = vtemp.pack(dim_c<3>, "xn", ptNo) + delta;
                });
            };
        }

        // if "rl" prop not exist, record
        if (!loops.has_attr("rl")) {
            auto &rls = loops.add_attr<float>("rl");
            pol(range(polys.size()), [&](int polyI) {
                auto poly = polys[polyI];
                auto i = poly[0];
                for (int k = 1; k < poly[1]; ++k) {
                    auto j = i + 1;
                    auto xi = vec_to_other<V3>(pos[loops[i]]);
                    auto xj = vec_to_other<V3>(pos[loops[j]]);
                    rls[i] = (xi - xj).norm();
                    i = j;
                }
            });
        }
        const auto &rls = loops.attr<float>("rl");

        pol(range(loops.size()), [&, vtemp = proxy<space>({}, vtemp)](int loopI) mutable {
            auto ptNo = loops[loopI];
            auto x = vec_to_other<V3>(pos[ptNo]);
            auto v = vec_to_other<V3>(vel[ptNo]);
            vtemp.tuple(dim_c<3>, "xn", ptNo) = x;
            vtemp.tuple(dim_c<3>, "vn", ptNo) = v;
        });

        auto dot = [&, temp = zs::Vector<float>{vtemp.get_allocator(), vtemp.size() + 1},
                    space_c = wrapv<space>{}](zs::SmallString tag0, zs::SmallString tag1) mutable {
            constexpr auto space = RM_CVREF_T(space_c)::value;
            pol(range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp), temp = proxy<space>(temp), offset0 = vtemp.getPropertyOffset(tag0),
                 offset1 = vtemp.getPropertyOffset(tag1)](int vi) mutable {
                    temp[vi] = vtemp.pack(dim_c<3>, offset0, vi).dot(vtemp.pack(dim_c<3>, offset1, vi));
                });
            reduce(pol, temp.begin(), temp.end() - 1, temp.end() - 1);
            return temp.getVal(vtemp.size());
        };
        auto maxIters = get_input2<int>("num_substeps");
        /// substeps
        for (int subi = 0; subi != maxIters; ++subi) {
            /// @brief newton krylov
            /// @note compute gradient
            pol(range(vtemp.size()), [&, vtemp = proxy<space>({}, vtemp)](int loopI) mutable {
                auto ptNo = loops[loopI];
                auto x = vtemp.pack(dim_c<3>, "xn", ptNo);
                auto v = vtemp.pack(dim_c<3>, "vn", ptNo);
                auto xtilde = x + v * dt; // inertia
                xtilde += grav * dt * dt; // gravity
                vtemp.tuple(dim_c<3>, "xtilde", ptNo) = xtilde;
                vtemp.tuple(dim_c<3>, "xhat", ptNo) = x;
                vtemp.tuple(dim_c<3>, "grad", ptNo) = V3::zeros(); // clear gradient
            });
            // collision
            if (resolveCollision)
                resolveCollision();
            // elasticity + inertia
            pol(range(polys.size()),
                [&, etemp = proxy<space>({}, etemp), vtemp = proxy<space>({}, vtemp)](int polyI) mutable {
                    auto poly = polys[polyI];
                    auto i = poly[0];
                    auto vi = loops[i];
                    auto xi = vtemp.pack(dim_c<3>, "xn", vi);
                    // inertial
                    vtemp.tuple(dim_c<3>, "grad", vi) =
                        vtemp.pack(dim_c<3>, "grad", vi) -
                        mass * (vtemp.pack(dim_c<3>, "xn", vi) - vtemp.pack(dim_c<3>, "xtilde", vi));

                    for (int k = 1; k < poly[1]; ++k) {
                        auto rl = rls[i];

                        auto j = i + 1;
                        auto vj = loops[j];
                        auto xj = vtemp.pack(dim_c<3>, "xn", vj);
                        auto xij = xj - xi;
                        auto lij = xij.norm();
                        auto dij = xij / lij;
                        auto gij = k * (lij - rl) * dij;
                        auto vfdt2 = gij * (dt * dt) * vol;
                        // elasticity
                        vtemp.tuple(dim_c<3>, "grad", vi) = vtemp.pack(dim_c<3>, "grad", vi) + vfdt2;
                        vtemp.tuple(dim_c<3>, "grad", vj) = vtemp.pack(dim_c<3>, "grad", vj) - vfdt2;

                        // inertial
                        vtemp.tuple(dim_c<3>, "grad", vj) =
                            vtemp.pack(dim_c<3>, "grad", vj) -
                            mass * (vtemp.pack(dim_c<3>, "xn", vj) - vtemp.pack(dim_c<3>, "xtilde", vj));

                        // elasticity hessian component
                        auto K = k * (zs::vec<float, 3, 3>::identity() -
                                      rl / lij * (zs::vec<float, 3, 3>::identity() - dyadic_prod(dij, dij)));
                        etemp.tuple(dim_c<3, 3>, "K", i) = K * dt * dt * vol;

                        // iterate
                        i = j;
                        vi = vj;
                        xi = xj;
                    }
                });
            // project gradient
            auto project = [&pol, &vtemp, &polys_ = polys, &loops_ = loops,
                            space_c = wrapv<space>{}](zs::SmallString tag) {
                constexpr auto space = RM_CVREF_T(space_c)::value;
                auto &polys = polys_;
                auto &loops = loops_;
                pol(range(polys.size()), [&polys, &loops, vtemp = proxy<space>(vtemp),
                                          tagOffset = vtemp.getPropertyOffset(tag)](int polyI) mutable {
                    auto poly = polys[polyI];
                    auto vi = loops[poly[0]];
                    vtemp.tuple(dim_c<3>, tagOffset, vi) = V3::zeros();
                });
            };
            project("grad");
            /// @note solve
            {
#if 0
                // explicit integration
                pol(range(loops.size()), [&, vtemp = proxy<space>({}, vtemp)](int loopI) mutable {
                    auto ptNo = loops[loopI];
                    auto delta = (vtemp.pack(dim_c<3>, "grad", ptNo)) / mass;
                    auto xnp1 = vtemp.pack(dim_c<3>, "xn", ptNo) + delta;
                    vtemp.tuple(dim_c<3>, "xn", ptNo) = xnp1;
                });
#else
                // mass precondition
                auto precondition = [&pol, &vtemp, mass, k, space_c = wrapv<space>{}](zs::SmallString srcTag,
                                                                                   zs::SmallString dstTag) {
                    constexpr auto space = RM_CVREF_T(space_c)::value;
                    pol(range(vtemp.size()),
                        [vtemp = proxy<space>(vtemp), srcPropOffset = vtemp.getPropertyOffset(srcTag),
                         dstPropOffset = vtemp.getPropertyOffset(dstTag), mass, k] ZS_LAMBDA(int vi) mutable {
                            vtemp.tuple(dim_c<3>, dstPropOffset, vi) = vtemp.pack(dim_c<3>, srcPropOffset, vi) / (mass + k);
                        });
                };
                auto multiply = [&, space_c = wrapv<space>{}](zs::SmallString srcTag, zs::SmallString dstTag) {
                    constexpr auto space = RM_CVREF_T(space_c)::value;
                    auto srcOffset = vtemp.getPropertyOffset(srcTag);
                    auto dstOffset = vtemp.getPropertyOffset(dstTag);
                    // inertia
                    pol(range(vtemp.size()),
                        [vtemp = proxy<space>(vtemp), dstOffset, srcOffset, mass] ZS_LAMBDA(int vi) mutable {
                            vtemp.tuple(dim_c<3>, dstOffset, vi) = vtemp.pack(dim_c<3>, srcOffset, vi) * mass;
                        });
                    // elasticity
                    pol(range(polys.size()),
                        [&, etemp = proxy<space>(etemp), vtemp = proxy<space>(vtemp),
                         Koffset = etemp.getPropertyOffset("K"), dstOffset, srcOffset](int polyI) mutable {
                            auto poly = polys[polyI];
                            auto i = poly[0];
                            auto vi = loops[i];
                            auto diri = vtemp.pack(dim_c<3>, srcOffset, vi);
                            for (int k = 1; k < poly[1]; ++k) {
                                auto j = i + 1;
                                auto vj = loops[j];
                                auto dirj = vtemp.pack(dim_c<3>, srcOffset, vj);

                                auto K = etemp.pack(dim_c<3, 3>, Koffset, i);
                                auto deltai = K * diri - K * dirj;
                                auto deltaj = -deltai;

                                vtemp.tuple(dim_c<3>, dstOffset, vi) = vtemp.pack(dim_c<3>, dstOffset, vi) + deltai;
                                vtemp.tuple(dim_c<3>, dstOffset, vj) = vtemp.pack(dim_c<3>, dstOffset, vj) + deltaj;

                                // iterate
                                i = j;
                                vi = vj;
                            }
                        });
                };
                /// @note cg
                constexpr float cgRel = 0.01;
                constexpr int CGCap = 1000;
                auto dirOffset = vtemp.getPropertyOffset("dir");
                auto gradOffset = vtemp.getPropertyOffset("grad");
                auto tempOffset = vtemp.getPropertyOffset("temp");
                auto rOffset = vtemp.getPropertyOffset("r");
                auto pOffset = vtemp.getPropertyOffset("p");
                auto qOffset = vtemp.getPropertyOffset("q");
                auto vtempv = proxy<space>(vtemp);
                pol(range(vtemp.size()),
                    [vtemp = vtempv, dirOffset](int i) mutable { vtemp.tuple(dim_c<3>, dirOffset, i) = V3::zeros(); });
                multiply("dir", "temp");
                project("temp");
                // r = grad - temp
                pol(range(vtemp.size()), [vtemp = vtempv, rOffset, gradOffset, tempOffset](int i) mutable {
                    vtemp.tuple(dim_c<3>, rOffset, i) =
                        vtemp.pack(dim_c<3>, gradOffset, i) - vtemp.pack(dim_c<3>, tempOffset, i);
                });
                precondition("r", "q");
                pol(range(vtemp.size()), [vtemp = vtempv, pOffset, qOffset](int i) mutable {
                    vtemp.tuple(dim_c<3>, pOffset, i) = vtemp.pack(dim_c<3>, qOffset, i);
                });
                auto zTrk = dot("r", "q");
                auto residualPreconditionedNorm2 = zTrk;
                auto localTol2 = zs::sqr(cgRel) * residualPreconditionedNorm2;
                int iter = 0;
                for (; iter != CGCap; ++iter) {
                    if (residualPreconditionedNorm2 <= localTol2)
                        break;
                    multiply("p", "temp");
                    project("temp"); // project production

                    auto alpha = zTrk / dot("temp", "p");
                    pol(range(vtemp.size()),
                        [vtemp = vtempv, dirOffset, pOffset, rOffset, tempOffset, alpha] ZS_LAMBDA(int vi) mutable {
                            vtemp.tuple(dim_c<3>, dirOffset, vi) =
                                vtemp.pack(dim_c<3>, dirOffset, vi) + alpha * vtemp.pack(dim_c<3>, pOffset, vi);
                            vtemp.tuple(dim_c<3>, rOffset, vi) =
                                vtemp.pack(dim_c<3>, rOffset, vi) - alpha * vtemp.pack(dim_c<3>, tempOffset, vi);
                        });

                    precondition("r", "q");

                    auto zTrkLast = zTrk;
                    zTrk = dot("q", "r");
                    if (zs::isnan(zTrk, zs::exec_seq)) {
                        iter = CGCap;
                        residualPreconditionedNorm2 =
                            (localTol2 / (cgRel * cgRel)) + std::max((localTol2 / (cgRel * cgRel)), (float)1);
                        continue;
                    }
                    auto beta = zTrk / zTrkLast;
                    pol(range(vtemp.size()), [vtemp = vtempv, beta, pOffset, qOffset] ZS_LAMBDA(int vi) mutable {
                        vtemp.tuple(dim_c<3>, pOffset, vi) =
                            vtemp.pack(dim_c<3>, qOffset, vi) + beta * vtemp.pack(dim_c<3>, pOffset, vi);
                    });

                    residualPreconditionedNorm2 = zTrk;
                }
                zeno::log_info(fmt::format("cg ends in {} iters.", iter));
                pol(range(vtemp.size()),
                    [vtemp = vtempv, xnOffset = vtemp.getPropertyOffset("xn"), dirOffset] ZS_LAMBDA(int vi) mutable {
                        vtemp.tuple(dim_c<3>, xnOffset, vi) =
                            vtemp.pack(dim_c<3>, xnOffset, vi) + vtemp.pack(dim_c<3>, dirOffset, vi);
                    });
#endif
            }

            /// @note update velocity
            pol(range(vtemp.size()), [&, vtemp = proxy<space>({}, vtemp)](int loopI) mutable {
                auto ptNo = loops[loopI];
                auto vn = (vtemp.pack(dim_c<3>, "xn", ptNo) - vtemp.pack(dim_c<3>, "xhat", ptNo)) / dt;
                vtemp.tuple(dim_c<3>, "vn", ptNo) = vn;
            });
        }
        /// @note write back position and velocity
        pol(range(vtemp.size()), [&, vtemp = proxy<space>({}, vtemp)](int loopI) mutable {
            auto ptNo = loops[loopI];
            auto xn = vtemp.pack(dim_c<3>, "xn", ptNo);
            auto vn = vtemp.pack(dim_c<3>, "vn", ptNo);
            pos[ptNo] = other_to_vec<3>(xn);
            vel[ptNo] = other_to_vec<3>(vn);
        });

        /// resolve collision

        set_output("guide_lines", std::move(gls));
    }
};

ZENDEFNODE(StepGuidelines, {
                               {
                                   {gParamType_Primitive, "guide_lines"},
                                   {gParamType_Primitive, "boundary_prim"},
                                   {gParamType_String, "weightTag", "bvh_ws"},
                                   {gParamType_String, "idTag", "bvh_id"},
                                   {gParamType_Float, "dt", "0.05"},
                                   {gParamType_Int, "num_substeps", "1"},
                                   {"vdb_collider"},
                                   {gParamType_Float, "sep_dist", "0"},
                                   {gParamType_Int, "collision_iters", "5"},
                               },
                               {
                                   {gParamType_Primitive, "guide_lines"},
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
                    float d = zs::detail::deduce_numeric_max<float>();
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
               {{gParamType_Primitive, "points"}, {gParamType_Primitive, "guide_lines"}, {gParamType_Bool, "interpAttrs", "1"}},
               {
                   {gParamType_Primitive, "prim"},
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

                ddd.normalize();
                // p += ddd * dx;
                if (sdf < 0)
                    p += ddd * -sdf;
                else
                    p += ddd * (sep_dist - sdf);
            }
            pos[vi] = zeno::other_to_vec<3>(p);
        });
        set_output("prim", points);
    }
};

ZENDEFNODE(RepelPoints, {
                            {
                                {gParamType_Primitive, "points"},
                                {"vdb_collider"},
                                {gParamType_Float, "sep_dist", "0"},
                                {gParamType_Int, "max_iter", "100"},
                            },
                            {
                                {gParamType_Primitive, "prim"},
                            },
                            {},
                            {"zs_hair"},
                        });
