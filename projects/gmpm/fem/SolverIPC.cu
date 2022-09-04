#include "../Utils.hpp"
#include "Solver.cuh"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"

namespace zeno {

template <typename VecT, int N = VecT::template range_t<0>::value,
          zs::enable_if_all<N % 3 == 0, N == VecT::template range_t<1>::value> = 0>
__forceinline__ __device__ void rotate_hessian(zs::VecInterface<VecT> &H, const typename IPCSystem::mat3 BCbasis[N / 3],
                                               const int BCorder[N / 3], const int BCfixed[], bool projectDBC) {
    // hessian rotation: trans^T hess * trans
    // left trans^T: multiplied on rows
    // right trans: multiplied on cols
    constexpr int NV = N / 3;
    // rotate and project
    for (int vi = 0; vi != NV; ++vi) {
        int offsetI = vi * 3;
        for (int vj = 0; vj != NV; ++vj) {
            int offsetJ = vj * 3;
            IPCSystem::mat3 tmp{};
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    tmp(i, j) = H(offsetI + i, offsetJ + j);
            // rotate
            tmp = BCbasis[vi].transpose() * tmp * BCbasis[vj];
            // project
            if (projectDBC) {
                for (int i = 0; i != 3; ++i) {
                    bool clearRow = i < BCorder[vi];
                    for (int j = 0; j != 3; ++j) {
                        bool clearCol = j < BCorder[vj];
                        if (clearRow || clearCol)
                            tmp(i, j) = (vi == vj && i == j ? 1 : 0);
                    }
                }
            } else {
                for (int i = 0; i != 3; ++i) {
                    bool clearRow = i < BCorder[vi] && BCfixed[vi] == 1;
                    for (int j = 0; j != 3; ++j) {
                        bool clearCol = j < BCorder[vj] && BCfixed[vj] == 1;
                        if (clearRow || clearCol)
                            tmp(i, j) = (vi == vj && i == j ? 1 : 0);
                    }
                }
            }
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    H(offsetI + i, offsetJ + j) = tmp(i, j);
        }
    }
    return;
}

void IPCSystem::computeBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag,
                                                 bool includeHessian) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    using mat12 = zs::vec<T, 12, 12>;
    using mat3 = zs::vec<T, 3, 3>;
    using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
    auto numPP = nPP.getVal();
    pol(range(numPP),
        [vtemp = proxy<space>({}, vtemp), tempPP = proxy<space>({}, tempPP), PP = proxy<space>(PP), gTag, xi2 = xi * xi,
         dHat = dHat, activeGap2, kappa = kappa, projectDBC = projectDBC, includeHessian] __device__(int ppi) mutable {
            auto pp = PP[ppi];
            auto x0 = vtemp.pack<3>("xn", pp[0]);
            auto x1 = vtemp.pack<3>("xn", pp[1]);
#if 0
            auto ppGrad = dist_grad_pp(x0, x1);
            auto dist2 = dist2_pp(x0, x1);
            if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
            auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto grad = ppGrad * (-barrierDistGrad);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pp[0]), grad(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pp[1]), grad(1, d));
            }
            // hessian
            if (!includeHessian)
                return;
            auto ppHess = dist_hess_pp(x0, x1);
            auto ppGrad_ = Vec6View{ppGrad.data()};
            ppHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) * dyadic_prod(ppGrad_, ppGrad_) +
                      barrierDistGrad * ppHess);
            // make pd
            make_pd(ppHess);
#else
            auto v0 = x1 - x0;
            auto Ds = v0;
            auto dis = v0.norm();

            auto vec_normal = -v0.normalized();
            auto target = vec3{0, 1, 0};

            auto vec = vec_normal.cross(target);
            T cos = vec_normal.dot(target);
            auto rotation = mat3::identity();

            auto d_hat_sqrt = dHat;
            if (cos + 1 == 0) {
                rotation(0, 0) = -1;
                rotation(1, 1) = -1;
            } else {
                mat3 cross_vec{0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0};
                rotation += cross_vec + cross_vec * cross_vec / (1 + cos);
            }

            auto pos0 = x0 + (d_hat_sqrt - dis) * vec_normal;

            auto rotate_uv0 = rotation * pos0;
            auto rotate_uv1 = rotation * x1;

            auto uv0 = rotate_uv0[1];
            auto uv1 = rotate_uv1[1];

            auto u0 = uv1 - uv0;
            auto Dm = u0;
            auto DmInv = 1 / u0;
            auto F = Ds * DmInv;
            T I5 = F.dot(F);

            auto tmp = F * 2;
            vec3 flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (1 + 2 * zs::log(I5) - 1 / I5)) * tmp;

            auto PFPx = zs::vec<T, 3, 6>::zeros();
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) {
                    PFPx(i, j) = i == j ? -DmInv : 0;
                    PFPx(i, 3 + j) = i == j ? DmInv : 0;
                }

            auto grad = -PFPx.transpose() * flatten_pk1;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pp[0]), grad(d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pp[1]), grad(3 + d));
            }

            if (!includeHessian)
                return;
            T lambda0 =
                kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
            // T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * (24 * I5 + 2 * zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 12 * I5 * zs::log(I5) - 12 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 14 * I5 * I5 * zs::log(I5) - 12)) / I5;
            auto Q0 = F / zs::sqrt(I5);
            auto H = lambda0 * dyadic_prod(Q0, Q0);
            auto ppHess = PFPx.transpose() * H * PFPx;
#endif
            // rotate and project
            mat3 BCbasis[2];
            int BCorder[2];
            int BCfixed[2];
            for (int i = 0; i != 2; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", pp[i]);
                BCorder[i] = vtemp("BCorder", pp[i]);
                BCfixed[i] = vtemp("BCfixed", pp[i]);
            }
            rotate_hessian(ppHess, BCbasis, BCorder, BCfixed, projectDBC);
            // pp[0], pp[1]
            tempPP.tuple<36>("H", ppi) = ppHess;
            /// construct P
            for (int vi = 0; vi != 2; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, pp[vi]), ppHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    auto numPE = nPE.getVal();
    pol(range(numPE),
        [vtemp = proxy<space>({}, vtemp), tempPE = proxy<space>({}, tempPE), PE = proxy<space>(PE), gTag, xi2 = xi * xi,
         dHat = dHat, activeGap2, kappa = kappa, projectDBC = projectDBC, includeHessian] __device__(int pei) mutable {
            auto pe = PE[pei];
            auto p = vtemp.pack<3>("xn", pe[0]);
            auto e0 = vtemp.pack<3>("xn", pe[1]);
            auto e1 = vtemp.pack<3>("xn", pe[2]);
#if 0
            auto peGrad = dist_grad_pe(p, e0, e1);
            auto dist2 = dist2_pe(p, e0, e1);
            if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto grad = peGrad * (-barrierDistGrad);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pe[0]), grad(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pe[1]), grad(1, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pe[2]), grad(2, d));
            }
            // hessian
            if (!includeHessian)
                return;
            auto peHess = dist_hess_pe(p, e0, e1);
            auto peGrad_ = Vec9View{peGrad.data()};
            peHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) * dyadic_prod(peGrad_, peGrad_) +
                      barrierDistGrad * peHess);
            // make pd
            make_pd(peHess);
#else
	      auto v0 = e0 - p;
        auto v1 = e1 - p;

        zs::vec<T, 3, 2> Ds{v0[0], v1[0], v0[1], v1[1], v0[2], v1[2]};
        auto triangle_normal = v0.cross(v1).normalized();
        auto target = vec3{0, 1, 0};

        auto vec = triangle_normal.cross(target);
        auto cos = triangle_normal.dot(target);
            
        auto edge_normal = (e0 - e1).cross(triangle_normal).normalized();
        auto dis = (p - e0).dot(edge_normal);

        auto rotation = mat3::identity();
        T d_hat_sqrt = dHat;
        if (cos + 1 == 0.0) {
                rotation(0, 0) = -1;
                rotation(1, 1) = -1;
        }
        else {
              mat3 cross_vec{0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0};
              rotation += cross_vec + cross_vec * cross_vec / (1 + cos);
        }

        auto pos0 = p + (d_hat_sqrt - dis) * edge_normal;

        auto rotate_uv0 = rotation * pos0;
        auto rotate_uv1 = rotation * e0;
        auto rotate_uv2 = rotation * e1;
        auto rotate_normal = rotation * edge_normal;

        using vec2 = zs::vec<T, 2>;
        auto uv0 = vec2(rotate_uv0[0], rotate_uv0[2]);
        auto uv1 = vec2(rotate_uv1[0], rotate_uv1[2]);
        auto uv2 = vec2(rotate_uv2[0], rotate_uv2[2]);
        auto normal = vec2(rotate_normal[0], rotate_normal[2]);

        auto u0 = uv1 - uv0;
        auto u1 = uv2 - uv0;

        using mat2 = zs::vec<T, 2, 2>;
        mat2 Dm{u0(0), u1(0), u0(1), u1(1)};
        auto DmInv = inverse(Dm);

        zs::vec<T, 3, 2> F = Ds * DmInv;
        // T I5 = normal.dot(F.transpose() * F * normal);
        T I5 = (F * normal).l2NormSqr();
        auto nn = dyadic_prod(normal, normal);
        auto fnn = F * nn;
        auto tmp = flatten(fnn) * 2;

        zs::vec<T, 6> flatten_pk1{};
        flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (1 + 2 * zs::log(I5) - 1 / I5)) * tmp;

        zs::vec<T, 6, 9> PFPx = dFdXMatrix(DmInv, wrapv<3>{});

        auto grad = -PFPx.transpose() * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, pe[0]), grad(d));
          atomic_add(exec_cuda, &vtemp(gTag, d, pe[1]), grad(3 + d));
          atomic_add(exec_cuda, &vtemp(gTag, d, pe[2]), grad(6 + d));
        }

        if (!includeHessian) return;
        T lambda0 = kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
        auto q0 = flatten(fnn) / zs::sqrt(I5);
        auto H = lambda0 * dyadic_prod(q0, q0);
        auto peHess = PFPx.transpose() * H * PFPx;
#endif
            // rotate and project
            mat3 BCbasis[3];
            int BCorder[3];
            int BCfixed[3];
            for (int i = 0; i != 3; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", pe[i]);
                BCorder[i] = vtemp("BCorder", pe[i]);
                BCfixed[i] = vtemp("BCfixed", pe[i]);
            }
            rotate_hessian(peHess, BCbasis, BCorder, BCfixed, projectDBC);
            // pe[0], pe[1], pe[2]
            tempPE.tuple<81>("H", pei) = peHess;
            /// construct P
            for (int vi = 0; vi != 3; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, pe[vi]), peHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    auto numPT = nPT.getVal();
    pol(range(numPT),
        [vtemp = proxy<space>({}, vtemp), tempPT = proxy<space>({}, tempPT), PT = proxy<space>(PT), gTag, xi2 = xi * xi,
         dHat = dHat, activeGap2, kappa = kappa, projectDBC = projectDBC, includeHessian] __device__(int pti) mutable {
            auto pt = PT[pti];
            auto p = vtemp.pack<3>("xn", pt[0]);
            auto t0 = vtemp.pack<3>("xn", pt[1]);
            auto t1 = vtemp.pack<3>("xn", pt[2]);
            auto t2 = vtemp.pack<3>("xn", pt[3]);
#if 0
            auto ptGrad = dist_grad_pt(p, t0, t1, t2);
            auto dist2 = dist2_pt(p, t0, t1, t2);
            if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto grad = ptGrad * (-barrierDistGrad);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pt[0]), grad(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pt[1]), grad(1, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pt[2]), grad(2, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, pt[3]), grad(3, d));
            }
            // hessian
            if (!includeHessian)
                return;
            auto ptHess = dist_hess_pt(p, t0, t1, t2);
            auto ptGrad_ = Vec12View{ptGrad.data()};
            ptHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) * dyadic_prod(ptGrad_, ptGrad_) +
                      barrierDistGrad * ptHess);
            // make pd
            make_pd(ptHess);
#else
	      auto v0 = t0 - p;
        auto v1 = t1 - p;
        auto v2 = t2 - p;
        mat3 Ds{v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]};
        auto normal = (t1 - t0).cross(t2 - t0).normalized();
        auto dis = v0.dot(normal);
        auto d_hat_sqrt = dHat;
        zs::vec<T, 9, 12> PDmPx{};
        if (dis > 0) {
          normal = -normal;
        } else {
          dis = -dis;
        }
        auto pos0 = p + normal * (d_hat_sqrt - dis);

        auto u0 = t0 - pos0;
        auto u1 = t1 - pos0;
        auto u2 = t2 - pos0;
        mat3 Dm{u0[0], u1[0], u2[0], u0[1], u1[1], u2[1], u0[2], u1[2], u2[2]};
        auto DmInv = inverse(Dm);
        auto F = Ds * DmInv;
        auto [uu, ss, vv] = math::qr_svd(F);
        auto values = zs::sqr(ss.sum() - 2);
        T I5 = (F * normal).l2NormSqr();

        zs::vec<T, 9> flatten_pk1{};
        {
          auto tmp = flatten(F * dyadic_prod(normal, normal)) * 2;
          flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (1 + 2 * zs::log(I5) - 1 / I5)) * tmp;
        }

        auto PFPx = dFdXMatrix(DmInv, wrapv<3>{});

        auto grad = -PFPx.transpose() * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
              atomic_add(exec_cuda, &vtemp(gTag, d, pt[0]), grad(d));
              atomic_add(exec_cuda, &vtemp(gTag, d, pt[1]), grad(3 + d));
              atomic_add(exec_cuda, &vtemp(gTag, d, pt[2]), grad(6 + d));
              atomic_add(exec_cuda, &vtemp(gTag, d, pt[3]), grad(9 + d));
        }

        if (!includeHessian) return;
        T lambda0 = kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
        auto q0 = flatten(F * dyadic_prod(normal, normal)) / zs::sqrt(I5);
        auto ptHess = PFPx.transpose() * (lambda0 * dyadic_prod(q0, q0)) * PFPx;
#endif
            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", pt[i]);
                BCorder[i] = vtemp("BCorder", pt[i]);
                BCfixed[i] = vtemp("BCfixed", pt[i]);
            }
            rotate_hessian(ptHess, BCbasis, BCorder, BCfixed, projectDBC);
            // pt[0], pt[1], pt[2], pt[3]
            tempPT.tuple<144>("H", pti) = ptHess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, pt[vi]), ptHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    auto numEE = nEE.getVal();
    pol(range(numEE),
        [vtemp = proxy<space>({}, vtemp), tempEE = proxy<space>({}, tempEE), EE = proxy<space>(EE), gTag, xi2 = xi * xi,
         dHat = dHat, activeGap2, kappa = kappa, projectDBC = projectDBC, includeHessian] __device__(int eei) mutable {
            auto ee = EE[eei];
            auto ea0 = vtemp.pack<3>("xn", ee[0]);
            auto ea1 = vtemp.pack<3>("xn", ee[1]);
            auto eb0 = vtemp.pack<3>("xn", ee[2]);
            auto eb1 = vtemp.pack<3>("xn", ee[3]);
#if 0
            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto grad = eeGrad * (-barrierDistGrad);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, ee[0]), grad(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, ee[1]), grad(1, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, ee[2]), grad(2, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, ee[3]), grad(3, d));
            }
            // hessian
            if (!includeHessian)
                return;
            auto eeHess = dist_hess_ee(ea0, ea1, eb0, eb1);
            auto eeGrad_ = Vec12View{eeGrad.data()};
            eeHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) * dyadic_prod(eeGrad_, eeGrad_) +
                      barrierDistGrad * eeHess);
            // make pd
            make_pd(eeHess);
#else
	      auto v0 = ea1 - ea0;
        auto v1 = eb0 - ea0;
        auto v2 = eb1 - ea0;
        mat3 Ds{v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]};
        auto normal = v0.cross(eb1 - eb0).normalized();
        auto dis = v1.dot(normal);
        auto d_hat_sqrt = dHat;
        if (dis < 0) {
          normal = -normal;
          dis = -dis;
        }
        auto pos2 = eb0 + normal * (d_hat_sqrt - dis);
        auto pos3 = eb1 + normal * (d_hat_sqrt - dis);
        if (d_hat_sqrt - dis < 0)
          printf("FUCKING WRONG EEHESS! dhat - dis = %f (which < 0)\n", d_hat_sqrt - dis);

        auto u0 = v0;
        auto u1 = pos2 - ea0;
        auto u2 = pos3 - ea0;
        mat3 Dm{u0[0], u1[0], u2[0], u0[1], u1[1], u2[1], u0[2], u1[2], u2[2]};
        auto DmInv = inverse(Dm);
        auto F = Ds * DmInv;
        auto I5 = (F * normal).l2NormSqr();
        // T I5 = normal.dot(F.transpose() * F * normal);

        zs::vec<T, 9> flatten_pk1{};
        {
          auto tmp = flatten(F * dyadic_prod(normal, normal));
          flatten_pk1 = -2 * kappa * (activeGap2 * activeGap2 * (I5 - 1) * (1 + 2 * zs::log(I5) - 1 / I5)) * tmp;
        }

        zs::vec<T, 9, 12> PFPx = dFdXMatrix(DmInv, wrapv<3>{});

        auto grad = -PFPx.transpose() * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, ee[0]), grad(d));
          atomic_add(exec_cuda, &vtemp(gTag, d, ee[1]), grad(3 + d));
          atomic_add(exec_cuda, &vtemp(gTag, d, ee[2]), grad(6 + d));
          atomic_add(exec_cuda, &vtemp(gTag, d, ee[3]), grad(9 + d));
        }

        if (!includeHessian) return;
        T lambda0 = kappa * (2 * activeGap2 * activeGap2 *(6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));

        if (lambda0 < 0)
          printf("FUCKING WRONG EEHESS! lambda0 = %e, I5 = %e\n", lambda0, I5);

        auto nn = dyadic_prod(normal, normal);
        auto fnn = F * nn;
        auto q0 = flatten(fnn) / zs::sqrt(I5);
        auto eeHess = PFPx.transpose() * (lambda0 * dyadic_prod(q0, q0)) * PFPx;
#endif
            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", ee[i]);
                BCorder[i] = vtemp("BCorder", ee[i]);
                BCfixed[i] = vtemp("BCfixed", ee[i]);
            }
            rotate_hessian(eeHess, BCbasis, BCorder, BCfixed, projectDBC);
            // ee[0], ee[1], ee[2], ee[3]
            tempEE.tuple<144>("H", eei) = eeHess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, ee[vi]), eeHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });

    if (enableMollification) {
        auto get_mollifier = [] ZS_LAMBDA(const auto &ea0Rest, const auto &ea1Rest, const auto &eb0Rest,
                                          const auto &eb1Rest, const auto &ea0, const auto &ea1, const auto &eb0,
                                          const auto &eb1) {
            T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
            return zs::make_tuple(mollifier_ee(ea0, ea1, eb0, eb1, epsX), mollifier_grad_ee(ea0, ea1, eb0, eb1, epsX),
                                  mollifier_hess_ee(ea0, ea1, eb0, eb1, epsX));
        };
        auto numEEM = nEEM.getVal();
        pol(range(numEEM), [vtemp = proxy<space>({}, vtemp), tempEEM = proxy<space>({}, tempEEM),
                            EEM = proxy<space>(EEM), gTag, xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa,
                            projectDBC = projectDBC, includeHessian, get_mollifier] __device__(int eemi) mutable {
            auto eem = EEM[eemi]; // <x, y, z, w>
            auto ea0Rest = vtemp.pack<3>("x0", eem[0]);
            auto ea1Rest = vtemp.pack<3>("x0", eem[1]);
            auto eb0Rest = vtemp.pack<3>("x0", eem[2]);
            auto eb1Rest = vtemp.pack<3>("x0", eem[3]);
            auto ea0 = vtemp.pack<3>("xn", eem[0]);
            auto ea1 = vtemp.pack<3>("xn", eem[1]);
            auto eb0 = vtemp.pack<3>("xn", eem[2]);
            auto eb1 = vtemp.pack<3>("xn", eem[3]);
#if 0
            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, kappa);
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, kappa);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] =
                get_mollifier(ea0Rest, ea1Rest, eb0Rest, eb1Rest, ea0, ea1, eb0, eb1);

            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledEEGrad = mollifierEE * barrierDistGrad * eeGrad;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[0]), -(scaledMollifierGrad(0, d) + scaledEEGrad(0, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[1]), -(scaledMollifierGrad(1, d) + scaledEEGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[2]), -(scaledMollifierGrad(2, d) + scaledEEGrad(2, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[3]), -(scaledMollifierGrad(3, d) + scaledEEGrad(3, d)));
            }

            if (!includeHessian)
                return;
            // hessian
            auto eeGrad_ = Vec12View{eeGrad.data()};
            auto eemHess = barrierDist2 * mollifierHessEE +
                           barrierDistGrad * (dyadic_prod(Vec12View{mollifierGradEE.data()}, eeGrad_) +
                                              dyadic_prod(eeGrad_, Vec12View{mollifierGradEE.data()}));

            auto eeHess = dist_hess_ee(ea0, ea1, eb0, eb1);
            eeHess = (barrierDistHess * dyadic_prod(eeGrad_, eeGrad_) + barrierDistGrad * eeHess);
            eemHess += mollifierEE * eeHess;
            // make pd
            make_pd(eemHess);
#else
        auto v0 = ea1 - ea0;
        auto v1 = eb1 - eb0;
        auto c = v0.cross(v1).norm();
        auto I1 = c * c;
        if (I1 == 0) {
          if (includeHessian) 
            tempEEM.template tuple<144>("H", eemi) = mat12::zeros();
          return;
        }
        auto PFPx = pFpx_pee(ea0, ea1, eb0, eb1, dHat);
        auto dis = dist2_ee(ea0, ea1, eb0, eb1);
        auto I2 = dis / activeGap2;
        dis = zs::sqrt(dis);
        auto F = mat3::zeros();
        F(0, 0) = 1;
        F(1, 1) = c;
        F(2, 2) = dis / dHat;
        constexpr auto n1 = vec3{0, 1, 0};
        constexpr auto n1n1 = dyadic_prod(n1, n1);
        constexpr auto n2 = vec3{0, 0, 1};
        constexpr auto n2n2 = dyadic_prod(n2, n2);

        auto eps_x = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);

        auto flatten_g1 = flatten(F * n1n1);
        auto flatten_g2 = flatten(F * n2n2);

        T p1 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
        T p2 = kappa * 2 * (I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) / (I2 * eps_x * eps_x);

        auto flatten_pk1 = flatten_g1 * p1 + flatten_g2 * p2;
        auto grad = -PFPx * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, eem[0]), grad[d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, eem[1]), grad[3 + d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, eem[2]), grad[6 + d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, eem[3]), grad[9 + d]);
        }

        if (!includeHessian)
          return;

        // hessian
        T lambda10 = kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);  // p1
        T lambda11 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
        T lambda12 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);

        auto fnn = F * n1n1;
        auto q10 = flatten(fnn);
        q10 /= c;

        mat3 Tx{0, 0, 0, 0, 0, 1, 0, -1, 0};
        mat3 Ty{0, 0, -1, 0, 0, 0, 1, 0, 0};
        mat3 Tz{0, 1, 0, -1, 0, 0, 0, 0, 0};
        constexpr auto ratio = (T)1 / (T)g_sqrt2;
        Tx *= ratio;
        Ty *= ratio;  // ?
        Tz *= ratio;

        auto q11 = flatten(Tx * fnn).normalized();
        auto q12 = flatten(Tz * fnn).normalized();

        auto M9_temp = dyadic_prod(q11, q11) * lambda11;
        auto projectedH = M9_temp;
        projectedH += dyadic_prod(q12, q12) * lambda12;

        T lambda20 = -kappa * (2 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (6 * I2 + 2 * I2 * zs::log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * zs::log(I2) + 1)) / (I2 * eps_x * eps_x);

        fnn = F * n2n2;
        auto q20 = flatten(fnn) / (dis / dHat); // sqrt(I2)

        T lambdag1g = kappa * 4 * c * F(2, 2) * ((2 * activeGap2 * activeGap2 * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) / (I2 * eps_x * eps_x));
        mat2 tmp{lambda10, lambdag1g, lambdag1g, lambda20};
        auto [eivals, eivecs] = zs::eigen_decomposition(tmp);
        for (int i = 0; i != 2; ++i) {
          if (eivals(i) > 0) {
            auto eimat = mat3::zeros();
            auto ci = col(eivecs, i);
            eimat(1, 1) = ci[0];
            eimat(2, 2) = ci[1];
            auto eiv = flatten(eimat);
            projectedH += dyadic_prod(eiv, eiv) * eivals[i];
          }
        }
        auto eemHess = PFPx * projectedH * PFPx.transpose();
#endif
            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", eem[i]);
                BCorder[i] = vtemp("BCorder", eem[i]);
                BCfixed[i] = vtemp("BCfixed", eem[i]);
            }
            rotate_hessian(eemHess, BCbasis, BCorder, BCfixed, projectDBC);
            // ee[0], ee[1], ee[2], ee[3]
            tempEEM.tuple<144>("H", eemi) = eemHess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, eem[vi]), eemHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
        auto numPPM = nPPM.getVal();
        pol(range(numPPM), [vtemp = proxy<space>({}, vtemp), tempPPM = proxy<space>({}, tempPPM),
                            PPM = proxy<space>(PPM), gTag, xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa,
                            projectDBC = projectDBC, includeHessian, get_mollifier] __device__(int ppmi) mutable {
            auto ppm = PPM[ppmi]; // <x, z, y, w>, <0, 2, 1, 3>
            auto ea0Rest = vtemp.pack<3>("x0", ppm[0]);
            auto ea1Rest = vtemp.pack<3>("x0", ppm[1]);
            auto eb0Rest = vtemp.pack<3>("x0", ppm[2]);
            auto eb1Rest = vtemp.pack<3>("x0", ppm[3]);
            auto ea0 = vtemp.pack<3>("xn", ppm[0]);
            auto ea1 = vtemp.pack<3>("xn", ppm[1]);
            auto eb0 = vtemp.pack<3>("xn", ppm[2]);
            auto eb1 = vtemp.pack<3>("xn", ppm[3]);
#if 0
            auto ppGrad = dist_grad_pp(ea0, eb0);
            auto dist2 = dist2_pp(ea0, eb0);
            if (dist2 < xi2) {
                printf("dist already smaller than xi!\n");
            }
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, kappa);
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, kappa);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] =
                get_mollifier(ea0Rest, ea1Rest, eb0Rest, eb1Rest, ea0, ea1, eb0, eb1);

            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledPPGrad = mollifierEE * barrierDistGrad * ppGrad;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[0]), -(scaledMollifierGrad(0, d) + scaledPPGrad(0, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[1]), -(scaledMollifierGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[2]), -(scaledMollifierGrad(2, d) + scaledPPGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[3]), -(scaledMollifierGrad(3, d)));
            }

            if (!includeHessian)
                return;

            // hessian
            using GradT = zs::vec<T, 12>;
            auto extendedPPGrad = GradT::zeros();
            for (int d = 0; d != 3; ++d) {
                extendedPPGrad(d) = barrierDistGrad * ppGrad(0, d);
                extendedPPGrad(6 + d) = barrierDistGrad * ppGrad(1, d);
            }
            auto ppmHess = barrierDist2 * mollifierHessEE +
                           dyadic_prod(Vec12View{mollifierGradEE.data()}, extendedPPGrad) +
                           dyadic_prod(extendedPPGrad, Vec12View{mollifierGradEE.data()});

            auto ppHess = dist_hess_pp(ea0, eb0);
            auto ppGrad_ = Vec6View{ppGrad.data()};

            ppHess = (barrierDistHess * dyadic_prod(ppGrad_, ppGrad_) + barrierDistGrad * ppHess);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) {
                    ppmHess(0 + i, 0 + j) += mollifierEE * ppHess(0 + i, 0 + j);
                    ppmHess(0 + i, 6 + j) += mollifierEE * ppHess(0 + i, 3 + j);
                    ppmHess(6 + i, 0 + j) += mollifierEE * ppHess(3 + i, 0 + j);
                    ppmHess(6 + i, 6 + j) += mollifierEE * ppHess(3 + i, 3 + j);
                }
            // make pd
            make_pd(ppmHess);
#else
        auto v0 = ea1 - ea0;
        auto v1 = eb1 - eb0;
        auto c = v0.cross(v1).norm();
        auto I1 = c * c;
        if (I1 == 0) {
          if (includeHessian) 
            tempPPM.template tuple<144>("H", ppmi) = mat12::zeros();
          return;
        }
        auto PFPx = pFpx_ppp(ea0, eb0, ea1, eb1, dHat);
        auto dis = dist2_pp(ea0, eb0);
        auto I2 = dis / activeGap2;
        dis = zs::sqrt(dis);
        auto F = mat3::zeros();
        F(0, 0) = 1;
        F(1, 1) = c;
        F(2, 2) = dis / dHat;
        constexpr auto n1 = vec3{0, 1, 0};
        constexpr auto n1n1 = dyadic_prod(n1, n1);
        constexpr auto n2 = vec3{0, 0, 1};
        constexpr auto n2n2 = dyadic_prod(n2, n2);

        auto eps_x = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);

        auto flatten_g1 = flatten(F * n1n1);
        auto flatten_g2 = flatten(F * n2n2);

        T p1 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
        T p2 = kappa * 2 * (I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) / (I2 * eps_x * eps_x);

        auto flatten_pk1 = flatten_g1 * p1 + flatten_g2 * p2;
        auto grad = -PFPx * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, ppm[0]), grad[d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, ppm[1]), grad[3 + d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, ppm[2]), grad[6 + d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, ppm[3]), grad[9 + d]);
        }

        if (!includeHessian)
          return;

        // hessian
        T lambda10 = kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);  // p1
        T lambda11 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
        T lambda12 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);

        auto fnn = F * n1n1;
        auto q10 = flatten(fnn);
        q10 /= c;

        mat3 Tx{0, 0, 0, 0, 0, 1, 0, -1, 0};
        mat3 Ty{0, 0, -1, 0, 0, 0, 1, 0, 0};
        mat3 Tz{0, 1, 0, -1, 0, 0, 0, 0, 0};
        constexpr auto ratio = (T)1 / (T)g_sqrt2;
        Tx *= ratio;
        Ty *= ratio;  // ?
        Tz *= ratio;

        auto q11 = flatten(Tx * fnn).normalized();
        auto q12 = flatten(Tz * fnn).normalized();

        auto M9_temp = dyadic_prod(q11, q11) * lambda11;
        auto projectedH = M9_temp;
        projectedH += dyadic_prod(q12, q12) * lambda12;

        T lambda20 = -kappa * (2 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (6 * I2 + 2 * I2 * zs::log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * zs::log(I2) + 1)) / (I2 * eps_x * eps_x);

        fnn = F * n2n2;
        auto q20 = flatten(fnn) / (dis / dHat); // sqrt(I2)

        T lambdag1g = kappa * 4 * c * F(2, 2) * ((2 * activeGap2 * activeGap2 * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) / (I2 * eps_x * eps_x));
        mat2 tmp{lambda10, lambdag1g, lambdag1g, lambda20};
        auto [eivals, eivecs] = zs::eigen_decomposition(tmp);
        for (int i = 0; i != 2; ++i) {
          if (eivals(i) > 0) {
            auto eimat = mat3::zeros();
            auto ci = col(eivecs, i);
            eimat(1, 1) = ci[0];
            eimat(2, 2) = ci[1];
            auto eiv = flatten(eimat);
            projectedH += dyadic_prod(eiv, eiv) * eivals[i];
          }
        }
        auto ppmHess = PFPx * projectedH * PFPx.transpose();
#endif
            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", ppm[i]);
                BCorder[i] = vtemp("BCorder", ppm[i]);
                BCfixed[i] = vtemp("BCfixed", ppm[i]);
            }
            rotate_hessian(ppmHess, BCbasis, BCorder, BCfixed, projectDBC);
            // ee[0], ee[1], ee[2], ee[3]
            tempPPM.tuple<144>("H", ppmi) = ppmHess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, ppm[vi]), ppmHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
        auto numPEM = nPEM.getVal();
        pol(range(numPEM), [vtemp = proxy<space>({}, vtemp), tempPEM = proxy<space>({}, tempPEM),
                            PEM = proxy<space>(PEM), gTag, xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa,
                            projectDBC = projectDBC, includeHessian, get_mollifier] __device__(int pemi) mutable {
            auto pem = PEM[pemi]; // <x, w, y, z>, <0, 2, 3, 1>
            auto ea0Rest = vtemp.pack<3>("x0", pem[0]);
            auto ea1Rest = vtemp.pack<3>("x0", pem[1]);
            auto eb0Rest = vtemp.pack<3>("x0", pem[2]);
            auto eb1Rest = vtemp.pack<3>("x0", pem[3]);
            auto ea0 = vtemp.pack<3>("xn", pem[0]);
            auto ea1 = vtemp.pack<3>("xn", pem[1]);
            auto eb0 = vtemp.pack<3>("xn", pem[2]);
            auto eb1 = vtemp.pack<3>("xn", pem[3]);
#if 0
            auto peGrad = dist_grad_pe(ea0, eb0, eb1);
            auto dist2 = dist2_pe(ea0, eb0, eb1);
            if (dist2 < xi2) {
                printf("dist already smaller than xi!\n");
            }
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, kappa);
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, kappa);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] =
                get_mollifier(ea0Rest, ea1Rest, eb0Rest, eb1Rest, ea0, ea1, eb0, eb1);

            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledPEGrad = mollifierEE * barrierDistGrad * peGrad;

            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[0]), -(scaledMollifierGrad(0, d) + scaledPEGrad(0, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[1]), -(scaledMollifierGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[2]), -(scaledMollifierGrad(2, d) + scaledPEGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[3]), -(scaledMollifierGrad(3, d) + scaledPEGrad(2, d)));
            }

            if (!includeHessian)
                return;

            // hessian
            using GradT = zs::vec<T, 12>;
            auto extendedPEGrad = GradT::zeros();
            for (int d = 0; d != 3; ++d) {
                extendedPEGrad(d) = barrierDistGrad * peGrad(0, d);
                extendedPEGrad(6 + d) = barrierDistGrad * peGrad(1, d);
                extendedPEGrad(9 + d) = barrierDistGrad * peGrad(2, d);
            }
            auto pemHess = barrierDist2 * mollifierHessEE +
                           dyadic_prod(Vec12View{mollifierGradEE.data()}, extendedPEGrad) +
                           dyadic_prod(extendedPEGrad, Vec12View{mollifierGradEE.data()});

            auto peHess = dist_hess_pe(ea0, eb0, eb1);
            auto peGrad_ = Vec9View{peGrad.data()};

            peHess = (barrierDistHess * dyadic_prod(peGrad_, peGrad_) + barrierDistGrad * peHess);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) {
                    pemHess(0 + i, 0 + j) += mollifierEE * peHess(0 + i, 0 + j);
                    //
                    pemHess(0 + i, 6 + j) += mollifierEE * peHess(0 + i, 3 + j);
                    pemHess(0 + i, 9 + j) += mollifierEE * peHess(0 + i, 6 + j);
                    //
                    pemHess(6 + i, 0 + j) += mollifierEE * peHess(3 + i, 0 + j);
                    pemHess(9 + i, 0 + j) += mollifierEE * peHess(6 + i, 0 + j);
                    //
                    pemHess(6 + i, 6 + j) += mollifierEE * peHess(3 + i, 3 + j);
                    pemHess(6 + i, 9 + j) += mollifierEE * peHess(3 + i, 6 + j);
                    pemHess(9 + i, 6 + j) += mollifierEE * peHess(6 + i, 3 + j);
                    pemHess(9 + i, 9 + j) += mollifierEE * peHess(6 + i, 6 + j);
                }

            // make pd
            make_pd(pemHess);
#else
        auto v0 = ea1 - ea0;
        auto v1 = eb1 - eb0;
        auto c = v0.cross(v1).norm();
        auto I1 = c * c;
        if (I1 == 0) {
          if (includeHessian) 
            tempPEM.template tuple<144>("H", pemi) = mat12::zeros();
          return;
        }
        auto PFPx = pFpx_ppe(ea0, eb0, eb1, ea1, dHat);
        auto dis = dist2_pe(ea0, eb0, eb1);
        auto I2 = dis / activeGap2;
        dis = zs::sqrt(dis);
        auto F = mat3::zeros();
        F(0, 0) = 1;
        F(1, 1) = c;
        F(2, 2) = dis / dHat;
        constexpr auto n1 = vec3{0, 1, 0};
        constexpr auto n1n1 = dyadic_prod(n1, n1);
        constexpr auto n2 = vec3{0, 0, 1};
        constexpr auto n2n2 = dyadic_prod(n2, n2);

        auto eps_x = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);

        auto flatten_g1 = flatten(F * n1n1);
        auto flatten_g2 = flatten(F * n2n2);

        T p1 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
        T p2 = kappa * 2 * (I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) / (I2 * eps_x * eps_x);

        auto flatten_pk1 = flatten_g1 * p1 + flatten_g2 * p2;
        auto grad = -PFPx * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, pem[0]), grad[d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, pem[1]), grad[3 + d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, pem[2]), grad[6 + d]);
          atomic_add(exec_cuda, &vtemp(gTag, d, pem[3]), grad[9 + d]);
        }

        if (!includeHessian)
          return;

        // hessian
        T lambda10 = kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);  // p1
        T lambda11 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
        T lambda12 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);

        auto fnn = F * n1n1;
        auto q10 = flatten(fnn);
        q10 /= c;

        mat3 Tx{0, 0, 0, 0, 0, 1, 0, -1, 0};
        mat3 Ty{0, 0, -1, 0, 0, 0, 1, 0, 0};
        mat3 Tz{0, 1, 0, -1, 0, 0, 0, 0, 0};
        constexpr auto ratio = (T)1 / (T)g_sqrt2;
        Tx *= ratio;
        Ty *= ratio;  // ?
        Tz *= ratio;

        auto q11 = flatten(Tx * fnn).normalized();
        auto q12 = flatten(Tz * fnn).normalized();

        auto M9_temp = dyadic_prod(q11, q11) * lambda11;
        auto projectedH = M9_temp;
        projectedH += dyadic_prod(q12, q12) * lambda12;

        T lambda20 = -kappa * (2 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (6 * I2 + 2 * I2 * zs::log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * zs::log(I2) + 1)) / (I2 * eps_x * eps_x);

        fnn = F * n2n2;
        auto q20 = flatten(fnn) / (dis / dHat); // sqrt(I2)

        T lambdag1g = kappa * 4 * c * F(2, 2) * ((2 * activeGap2 * activeGap2 * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) / (I2 * eps_x * eps_x));
        mat2 tmp{lambda10, lambdag1g, lambdag1g, lambda20};
        auto [eivals, eivecs] = zs::eigen_decomposition(tmp);
        for (int i = 0; i != 2; ++i) {
          if (eivals(i) > 0) {
            auto eimat = mat3::zeros();
            auto ci = col(eivecs, i);
            eimat(1, 1) = ci[0];
            eimat(2, 2) = ci[1];
            auto eiv = flatten(eimat);
            projectedH += dyadic_prod(eiv, eiv) * eivals[i];
          }
        }
        auto pemHess = PFPx * projectedH * PFPx.transpose();
#endif
            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", pem[i]);
                BCorder[i] = vtemp("BCorder", pem[i]);
                BCfixed[i] = vtemp("BCfixed", pem[i]);
            }
            rotate_hessian(pemHess, BCbasis, BCorder, BCfixed, projectDBC);
            // ee[0], ee[1], ee[2], ee[3]
            tempPEM.tuple<144>("H", pemi) = pemHess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, pem[vi]), pemHess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    }
    return;
}

void IPCSystem::computeFrictionBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag,
                                                         bool includeHessian) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
    auto numFPP = nFPP.getVal();
    pol(range(numFPP),
        [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP), FPP = proxy<space>(FPP), gTag,
         epsvh = epsv * dt, fricMu = fricMu, projectDBC = projectDBC, includeHessian] __device__(int fppi) mutable {
            auto fpp = FPP[fppi];
            auto p0 = vtemp.pack<3>("xn", fpp[0]) - vtemp.pack<3>("xhat", fpp[0]);
            auto p1 = vtemp.pack<3>("xn", fpp[1]) - vtemp.pack<3>("xhat", fpp[1]);
            auto basis = fricPP.pack<3, 2>("basis", fppi);
            auto fn = fricPP("fn", fppi);
            auto relDX3D = point_point_rel_dx(p0, p1);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -point_point_rel_dx_tan_to_mesh(relDX, basis);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, fpp[0]), TTTDX(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fpp[1]), TTTDX(1, d));
            }
            // hessian
            if (!includeHessian)
                return;
            relDX = basis.transpose() * relDX3D;
            auto TT = point_point_TT(basis); // 2x6
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 6, 6>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(TT.transpose() * ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar), ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }
            // rotate and project
            mat3 BCbasis[2];
            int BCorder[2];
            int BCfixed[2];
            for (int i = 0; i != 2; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fpp[i]);
                BCorder[i] = vtemp("BCorder", fpp[i]);
                BCfixed[i] = vtemp("BCfixed", fpp[i]);
            }
            rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
            // pp[0], pp[1]
            fricPP.tuple<36>("H", fppi) = hess;
            /// construct P
            for (int vi = 0; vi != 2; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fpp[vi]), hess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    auto numFPE = nFPE.getVal();
    pol(range(numFPE),
        [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE), FPE = proxy<space>(FPE), gTag,
         epsvh = epsv * dt, fricMu = fricMu, projectDBC = projectDBC, includeHessian] __device__(int fpei) mutable {
            auto fpe = FPE[fpei];
            auto p = vtemp.pack<3>("xn", fpe[0]) - vtemp.pack<3>("xhat", fpe[0]);
            auto e0 = vtemp.pack<3>("xn", fpe[1]) - vtemp.pack<3>("xhat", fpe[1]);
            auto e1 = vtemp.pack<3>("xn", fpe[2]) - vtemp.pack<3>("xhat", fpe[2]);
            auto basis = fricPE.pack<3, 2>("basis", fpei);
            auto fn = fricPE("fn", fpei);
            auto yita = fricPE("yita", fpei);
            auto relDX3D = point_edge_rel_dx(p, e0, e1, yita);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -point_edge_rel_dx_tan_to_mesh(relDX, basis, yita);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, fpe[0]), TTTDX(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fpe[1]), TTTDX(1, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fpe[2]), TTTDX(2, d));
            }
            // hessian
            if (!includeHessian)
                return;
            relDX = basis.transpose() * relDX3D;
            auto TT = point_edge_TT(basis, yita); // 2x9
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 9, 9>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(TT.transpose() * ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar), ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }
            // rotate and project
            mat3 BCbasis[3];
            int BCorder[3];
            int BCfixed[3];
            for (int i = 0; i != 3; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fpe[i]);
                BCorder[i] = vtemp("BCorder", fpe[i]);
                BCfixed[i] = vtemp("BCfixed", fpe[i]);
            }
            rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
            // pe[0], pe[1], pe[2]
            fricPE.tuple<81>("H", fpei) = hess;
            /// construct P
            for (int vi = 0; vi != 3; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fpe[vi]), hess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    auto numFPT = nFPT.getVal();
    pol(range(numFPT),
        [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT), FPT = proxy<space>(FPT), gTag,
         epsvh = epsv * dt, fricMu = fricMu, projectDBC = projectDBC, includeHessian] __device__(int fpti) mutable {
            auto fpt = FPT[fpti];
            auto p = vtemp.pack<3>("xn", fpt[0]) - vtemp.pack<3>("xhat", fpt[0]);
            auto v0 = vtemp.pack<3>("xn", fpt[1]) - vtemp.pack<3>("xhat", fpt[1]);
            auto v1 = vtemp.pack<3>("xn", fpt[2]) - vtemp.pack<3>("xhat", fpt[2]);
            auto v2 = vtemp.pack<3>("xn", fpt[3]) - vtemp.pack<3>("xhat", fpt[3]);
            auto basis = fricPT.pack<3, 2>("basis", fpti);
            auto fn = fricPT("fn", fpti);
            auto betas = fricPT.pack<2>("beta", fpti);
            auto relDX3D = point_triangle_rel_dx(p, v0, v1, v2, betas[0], betas[1]);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -point_triangle_rel_dx_tan_to_mesh(relDX, basis, betas[0], betas[1]);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, fpt[0]), TTTDX(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fpt[1]), TTTDX(1, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fpt[2]), TTTDX(2, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fpt[3]), TTTDX(3, d));
            }
            // hessian
            if (!includeHessian)
                return;
            relDX = basis.transpose() * relDX3D;
            auto TT = point_triangle_TT(basis, betas[0], betas[1]); // 2x12
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 12, 12>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(TT.transpose() * ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar), ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }
            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fpt[i]);
                BCorder[i] = vtemp("BCorder", fpt[i]);
                BCfixed[i] = vtemp("BCfixed", fpt[i]);
            }
            rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
            // pt[0], pt[1], pt[2], pt[3]
            fricPT.tuple<144>("H", fpti) = hess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fpt[vi]), hess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    auto numFEE = nFEE.getVal();
    pol(range(numFEE),
        [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE), FEE = proxy<space>(FEE), gTag,
         epsvh = epsv * dt, fricMu = fricMu, projectDBC = projectDBC, includeHessian] __device__(int feei) mutable {
            auto fee = FEE[feei];
            auto e0 = vtemp.pack<3>("xn", fee[0]) - vtemp.pack<3>("xhat", fee[0]);
            auto e1 = vtemp.pack<3>("xn", fee[1]) - vtemp.pack<3>("xhat", fee[1]);
            auto e2 = vtemp.pack<3>("xn", fee[2]) - vtemp.pack<3>("xhat", fee[2]);
            auto e3 = vtemp.pack<3>("xn", fee[3]) - vtemp.pack<3>("xhat", fee[3]);
            auto basis = fricEE.pack<3, 2>("basis", feei);
            auto fn = fricEE("fn", feei);
            auto gammas = fricEE.pack<2>("gamma", feei);
            auto relDX3D = edge_edge_rel_dx(e0, e1, e2, e3, gammas[0], gammas[1]);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -edge_edge_rel_dx_tan_to_mesh(relDX, basis, gammas[0], gammas[1]);
            // gradient
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, fee[0]), TTTDX(0, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fee[1]), TTTDX(1, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fee[2]), TTTDX(2, d));
                atomic_add(exec_cuda, &vtemp(gTag, d, fee[3]), TTTDX(3, d));
            }
            // hessian
            if (!includeHessian)
                return;
            relDX = basis.transpose() * relDX3D;
            auto TT = edge_edge_TT(basis, gammas[0], gammas[1]); // 2x12
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 12, 12>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(TT.transpose() * ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar), ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }

            // rotate and project
            mat3 BCbasis[4];
            int BCorder[4];
            int BCfixed[4];
            for (int i = 0; i != 4; ++i) {
                BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fee[i]);
                BCorder[i] = vtemp("BCorder", fee[i]);
                BCfixed[i] = vtemp("BCfixed", fee[i]);
            }
            rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
            // ee[0], ee[1], ee[2], ee[3]
            fricEE.tuple<144>("H", feei) = hess;
            /// construct P
            for (int vi = 0; vi != 4; ++vi) {
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fee[vi]), hess(vi * 3 + i, vi * 3 + j));
                    }
            }
        });
    return;
}

} // namespace zeno