#include "FastCloth.cuh"
#include "TopoUtils.hpp"
#include "collision_energy/vertex_face_sqrt_collision.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

namespace zeno {

void FastClothSystem::findConstraints(zs::CudaExecutionPolicy &pol, T dHat) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    nPP.setVal(0);

    zs::CppTimer timer;
    timer.tick();
    if (enableContact) {
        if (enableContactSelf) {
            auto pBvs = retrieve_bounding_volumes(pol, vtemp, "xn", svInds, zs::wrapv<1>{}, 0);
            svBvh.refit(pol, pBvs);
            /// @note all cloth edge lower-bound constraints inheritly included
            findCollisionConstraints(pol, dHat, false);
        }
        if (hasBoundary()) {
            auto pBvs = retrieve_bounding_volumes(pol, vtemp, "xn", *coPoints, zs::wrapv<1>{}, coOffset);
            bouSvBvh.refit(pol, pBvs);
            findCollisionConstraints(pol, dHat, true);
            // for repulsion
            // findBoundaryCellCollisionConstraints(pol, dHat);
        }
        frontManageRequired = false;
    }
    /// @note check upper-bound constraints for cloth edges
    for (auto &primHandle : prims) {
        auto &ses = primHandle.getSurfEdges();
        pol(Collapse{ses.size()},
            [ses = proxy<space>({}, ses), vtemp = proxy<space>({}, vtemp), E = proxy<space>(E), nE = proxy<space>(nE),
             threshold = L * L - epsSlack, vOffset = primHandle.vOffset] __device__(int sei) mutable {
                const auto vij = ses.pack(dim_c<2>, "inds", sei).reinterpret_bits(int_c) + vOffset;
                const auto &vi = vij[0];
                const auto &vj = vij[1];
                auto pi = vtemp.pack(dim_c<3>, "xn", vi);
                auto pj = vtemp.pack(dim_c<3>, "xn", vj);
                if (auto d2 = dist2_pp(pi, pj); d2 > threshold) {
                    auto no = atomic_add(exec_cuda, &nE[0], 1);
                    E[no] = vij;
                }
            });
    }
}

#define PROFILE_CD 0

void FastClothSystem::findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat, bool withBoundary) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    pol.profile(PROFILE_CD);
    /// pt
    const auto &svbvh = withBoundary ? bouSvBvh : svBvh;
    auto &svfront = withBoundary ? boundarySvFront : selfSvFront;
    pol(Collapse{svfront.size()},
        [svInds = proxy<space>({}, svInds), eles = proxy<space>({}, withBoundary ? *coPoints : svInds),
         // exclTris = withBoundary ? proxy<space>(exclBouSts) : proxy<space>(exclSts),
         vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(svbvh), front = proxy<space>(svfront),
         PP = proxy<space>(PP), nPP = proxy<space>(nPP), dHat2 = dHat * dHat, thickness = dHat,
         voffset = withBoundary ? coOffset : 0, frontManageRequired = frontManageRequired] __device__(int i) mutable {
            auto vi = front.prim(i);
            vi = reinterpret_bits<int>(svInds("inds", vi));
            auto pi = vtemp.pack(dim_c<3>, "xn", vi);
            auto bv = bv_t{get_bounding_box(pi - thickness, pi + thickness)};
            auto f = [&](int svI) {
                // if (exclTris[stI]) return;
                auto vj = reinterpret_bits<int>(eles("inds", svI)) + voffset;
                if (vi > vj)
                    return;
                auto pj = vtemp.pack(dim_c<3>, "xn", vj);
                // edge or not

                if (auto d2 = dist2_pp(pi, pj); d2 < dHat2) {
                    auto no = atomic_add(exec_cuda, &nPP[0], 1);
                    PP[no] = pair_t{vi, vj};
                }
            };
            if (frontManageRequired)
                bvh.iter_neighbors(bv, i, front, f);
            else
                bvh.iter_neighbors(bv, front.node(i), f);
        });
    if (frontManageRequired)
        svfront.reorder(pol);
    pol.profile(false);
}

void FastClothSystem::computeConstraintGradients(zs::CudaExecutionPolicy &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto [npp, ne] = getConstraintCnt();
    fmt::print("dcd broad phase [pp, edge constraints]: {}, {}", npp, ne);
    pol(range(npp),
        [vtemp = proxy<space>({}, vtemp), tempPP = proxy<space>({}, tempPP), PP = proxy<space>(PP), rho = rho, mu = mu,
         Btot2 = (B + Btight) * (B + Btight), eps = epsSlack] __device__(int i) mutable {
            auto pp = PP[i];
            zs::vec<T, 3> vs[2] = {vtemp.pack(dim_c<3>, "xn", pp[0]), vtemp.pack(dim_c<3>, "xn", pp[1])};
            const auto &a = vs[0];
            const auto &b = vs[1];
            const auto t2 = a[0] * 2;
            const auto t3 = a[1] * 2;
            const auto t4 = a[2] * 2;
            const auto t5 = b[0] * 2;
            const auto t6 = b[1] * 2;
            const auto t7 = b[2] * 2;
            {

                const auto t8 = -t5;
                const auto t9 = -t6;
                const auto t10 = -t7;
                const auto t11 = t2 + t8;
                const auto t12 = t3 + t9;
                const auto t13 = t4 + t10;
                const auto t14 = rho * t11;
                const auto t15 = rho * t12;
                const auto t16 = rho * t13;
                auto grad = zs::vec<T, 6>{-t14, -t15, -t16, t14, t15, t16};
                tempPP.tuple(dim_c<6>, "softG", i) = grad;
            }
            {
                auto t8 = -Btot2;
                auto t9 = -b[0];
                auto t11 = -b[1];
                auto t13 = -b[2];
                auto t15 = 1 / eps;
                auto t10 = -t5;
                auto t12 = -t6;
                auto t14 = -t7;
                auto t16 = t15 * t15;
                auto t17 = a[0] + t9;
                auto t18 = a[1] + t11;
                auto t19 = a[2] + t13;
                auto t20 = t2 + t10;
                auto t21 = t3 + t12;
                auto t22 = t4 + t14;
                auto t23 = t17 * t17;
                auto t24 = t18 * t18;
                auto t25 = t19 * t10;
                auto t26 = t8 + t23 + t24 + t25;
                auto t27 = t26 * t26;
                auto t28 = t26 * t26 * t26;
                auto t32 = t15 * t20 * t26 * 2;
                auto t33 = t15 * t21 * t26 * 2;
                auto t34 = t15 * t22 * t26 * 2;
                auto t29 = t15 * t27;
                auto t30 = t16 * t28;
                auto t35 = t16 * t20 * t27 * 3;
                auto t36 = t16 * t21 * t27 * 3;
                auto t37 = t16 * t22 * t27 * 3;
                auto t31 = -t30;
                auto t38 = -t35;
                auto t39 = -t36;
                auto t40 = -t37;
                auto t41 = t20 + t32 + t38;
                auto t42 = t21 + t33 + t39;
                auto t43 = t22 + t34 + t40;
                auto t44 = t26 + t29 + t31;
                auto t45 = 1 / t44;
                auto t46 = mu * t41 * t45;
                auto t47 = mu * t42 * t45;
                auto t48 = mu * t43 * t45;
                auto grad = zs::vec<T, 6>{-t46, -t47, -t48, t46, t47, t48};
                tempPP.tuple(dim_c<6>, "hardG", i) = grad;
            }
        });

    pol(range(ne), [vtemp = proxy<space>({}, vtemp), tempE = proxy<space>({}, tempE), E = proxy<space>(E), rho = rho,
                    mu = mu, L2 = L * L, eps = epsSlack] __device__(int i) mutable {
        auto e = E[i];
        zs::vec<T, 3> vs[2] = {vtemp.pack(dim_c<3>, "xn", e[0]), vtemp.pack(dim_c<3>, "xn", e[1])};
        const auto &a = vs[0];
        const auto &b = vs[1];
        const auto t2 = a[0] * 2;
        const auto t3 = a[1] * 2;
        const auto t4 = a[2] * 2;
        const auto t5 = b[0] * 2;
        const auto t6 = b[1] * 2;
        const auto t7 = b[2] * 2;
        {
            const auto t8 = -t5;
            const auto t9 = -t6;
            const auto t10 = -t7;
            const auto t11 = t2 + t8;
            const auto t12 = t3 + t9;
            const auto t13 = t4 + t10;
            const auto t14 = rho * t11;
            const auto t15 = rho * t12;
            const auto t16 = rho * t13;
            auto grad = zs::vec<T, 6>{t14, t15, t16, -t14, -t15, -t16};
            tempE.tuple(dim_c<6>, "softG", i) = grad;
        }
        {
            auto t8 = -L2;
            auto t12 = -b[0];
            auto t14 = -b[1];
            auto t16 = -b[2];
            auto t18 = 1 / eps;
            auto t9 = -t2;
            auto t10 = -t3;
            auto t11 = -t4;
            auto t13 = -t5;
            auto t15 = -t6;
            auto t17 = -t7;
            auto t19 = t18 * t18;
            auto t20 = a[0] + t12;
            auto t21 = a[1] + t14;
            auto t22 = a[2] + t16;
            auto t23 = t2 + t13;
            auto t24 = t3 + t15;
            auto t25 = t4 + t17;
            auto t26 = t20 * t20;
            auto t27 = t21 * t21;
            auto t28 = t22 * t22;
            auto t29 = -t26;
            auto t30 = -t27;
            auto t31 = -t28;
            auto t32 = t8 + t26 + t27 + t28;
            auto t33 = t32 * t32;
            auto t34 = t32 * t32 * t32;
            auto t37 = t18 * t23 * t32 * 2;
            auto t38 = t18 * t24 * t32 * 2;
            auto t39 = t18 * t25 * t32 * 2;
            auto t35 = t18 * t33;
            auto t36 = t19 * t34;
            auto t40 = t19 * t23 * t33 * 3;
            auto t41 = t19 * t24 * t33 * 3;
            auto t42 = t19 * t25 * t33 * 3;
            auto t43 = t5 + t9 + t37 + t40;
            auto t44 = t6 + t10 + t38 + t41;
            auto t45 = t7 + t11 + t39 + t42;
            auto t46 = L2 + t29 + t30 + t31 + t35 + t36;
            auto t47 = 1 / t46;
            auto t48 = mu * t43 * t47;
            auto t49 = mu * t44 * t47;
            auto t50 = mu * t45 * t47;
            auto grad = zs::vec<T, 6>{-t48, -t49, -t50, t48, t49, t50};
            tempE.tuple(dim_c<6>, "hardG", i) = grad;
        }
    });
}

} // namespace zeno