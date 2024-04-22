#include "UnifiedSolver.cuh"
#include "Utils.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/Friction.hpp"
#include "zensim/geometry/SpatialQuery.hpp"

///
/// @ref https://github.com/KemengHuang/GPU_IPC
/// @author Kemeng Huang
///
namespace zeno {

#define RANK 2
#define gassThreshold ((T)0.1)

typename UnifiedIPCSystem::T UnifiedIPCSystem::collisionEnergy(zs::CudaExecutionPolicy &pol,
                                                               const zs::SmallString tag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> &es = temp;

    std::vector<T> Es(0);
    auto activeGap2 = dHat * dHat + 2 * xi * dHat;
    auto numPP = PP.getCount();
    es.resize(count_warps(numPP));
    es.reset(0);
    pol(range(numPP), [vtemp = proxy<space>({}, vtemp), PP = PP.port(), es = proxy<space>(es), xi2 = xi * xi,
                       dHat = dHat, activeGap2, n = numPP] __device__(int ppi) mutable {
        auto pp = PP[ppi];
        auto x0 = vtemp.pack<3>("xn", pp[0]);
        auto x1 = vtemp.pack<3>("xn", pp[1]);
        auto dist2 = dist2_pp(x0, x1);
        if (dist2 < xi2)
            printf("dist already smaller than xi!\n");
        // atomic_add(exec_cuda, &res[0],
        //           zs::barrier(dist2 - xi2, activeGap2, kappa));
        // es[ppi] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

        auto I5 = dist2 / activeGap2;
        auto lenE = (dist2 - activeGap2);
#if (RANK == 1)
        auto E = -lenE * lenE * zs::log(I5);
#elif (RANK == 2)
        auto E = lenE * lenE * zs::sqr(zs::log(I5));
#elif (RANK == 4)
        auto E = lenE * lenE * zs::sqr(zs::sqr(zs::log(I5)));
#endif
        reduce_to(ppi, n, E, es[ppi / 32]);
    });
    Es.push_back(reduce(pol, es) * kappa);

    auto numPE = PE.getCount();
    es.resize(count_warps(numPE));
    es.reset(0);
    pol(range(numPE), [vtemp = proxy<space>({}, vtemp), PE = PE.port(), es = proxy<space>(es), xi2 = xi * xi,
                       dHat = dHat, activeGap2, n = numPE] __device__(int pei) mutable {
        auto pe = PE[pei];
        auto p = vtemp.pack<3>("xn", pe[0]);
        auto e0 = vtemp.pack<3>("xn", pe[1]);
        auto e1 = vtemp.pack<3>("xn", pe[2]);

        auto dist2 = dist2_pe(p, e0, e1);
        if (dist2 < xi2)
            printf("dist already smaller than xi!\n");
        // atomic_add(exec_cuda, &res[0],
        //           zs::barrier(dist2 - xi2, activeGap2, kappa));
        // es[pei] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

        auto I5 = dist2 / activeGap2;
        auto lenE = (dist2 - activeGap2);
#if (RANK == 1)
        auto E = -lenE * lenE * zs::log(I5);
#elif (RANK == 2)
        auto E = lenE * lenE * zs::sqr(zs::log(I5));
#elif (RANK == 4)
        auto E = lenE * lenE * zs::sqr(zs::sqr(zs::log(I5)));
#endif
        reduce_to(pei, n, E, es[pei / 32]);
    });
    Es.push_back(reduce(pol, es) * kappa);

    auto numPT = PT.getCount();
    es.resize(count_warps(numPT));
    es.reset(0);
    pol(range(numPT), [vtemp = proxy<space>({}, vtemp), PT = PT.port(), es = proxy<space>(es), xi2 = xi * xi,
                       dHat = dHat, activeGap2, n = numPT] __device__(int pti) mutable {
        auto pt = PT[pti];
        auto p = vtemp.pack<3>("xn", pt[0]);
        auto t0 = vtemp.pack<3>("xn", pt[1]);
        auto t1 = vtemp.pack<3>("xn", pt[2]);
        auto t2 = vtemp.pack<3>("xn", pt[3]);

        auto dist2 = dist2_pt(p, t0, t1, t2);
        if (dist2 < xi2)
            printf("dist already smaller than xi!\n");
        // atomic_add(exec_cuda, &res[0],
        //           zs::barrier(dist2 - xi2, activeGap2, kappa));
        // es[pti] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

        auto I5 = dist2 / activeGap2;
        auto lenE = (dist2 - activeGap2);
#if (RANK == 1)
        auto E = -lenE * lenE * zs::log(I5);
#elif (RANK == 2)
        auto E = lenE * lenE * zs::sqr(zs::log(I5));
#elif (RANK == 4)
        auto E = lenE * lenE * zs::sqr(zs::sqr(zs::log(I5)));
#endif
        reduce_to(pti, n, E, es[pti / 32]);
    });
    Es.push_back(reduce(pol, es) * kappa);

    auto numEE = EE.getCount();
    es.resize(count_warps(numEE));
    es.reset(0);
    pol(range(numEE), [vtemp = proxy<space>({}, vtemp), EE = EE.port(), es = proxy<space>(es), xi2 = xi * xi,
                       dHat = dHat, activeGap2, n = numEE] __device__(int eei) mutable {
        auto ee = EE[eei];
        auto ea0 = vtemp.pack<3>("xn", ee[0]);
        auto ea1 = vtemp.pack<3>("xn", ee[1]);
        auto eb0 = vtemp.pack<3>("xn", ee[2]);
        auto eb1 = vtemp.pack<3>("xn", ee[3]);

        auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
        if (dist2 < xi2)
            printf("dist already smaller than xi!\n");
        // atomic_add(exec_cuda, &res[0],
        //           zs::barrier(dist2 - xi2, activeGap2, kappa));
        // es[eei] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

        auto I5 = dist2 / activeGap2;
        auto lenE = (dist2 - activeGap2);
#if (RANK == 1)
        auto E = -lenE * lenE * zs::log(I5);
#elif (RANK == 2)
        auto E = lenE * lenE * zs::sqr(zs::log(I5));
#elif (RANK == 4)
        auto E = lenE * lenE * zs::sqr(zs::sqr(zs::log(I5)));
#endif
        reduce_to(eei, n, E, es[eei / 32]);
    });
    Es.push_back(reduce(pol, es) * kappa);

    if (enableMollification) {
        auto numEEM = EEM.getCount();
        es.resize(count_warps(numEEM));
        es.reset(0);
        pol(range(numEEM), [vtemp = proxy<space>({}, vtemp), EEM = EEM.port(), es = proxy<space>(es), xi2 = xi * xi,
                            dHat = dHat, activeGap2, n = numEEM] __device__(int eemi) mutable {
            auto eem = EEM[eemi];
            auto ea0 = vtemp.pack<3>("xn", eem[0]);
            auto ea1 = vtemp.pack<3>("xn", eem[1]);
            auto eb0 = vtemp.pack<3>("xn", eem[2]);
            auto eb1 = vtemp.pack<3>("xn", eem[3]);

            auto v0 = ea1 - ea0;
            auto v1 = eb1 - eb0;
            auto c = v0.cross(v1).norm();
            auto I1 = c * c;
            T E = 0;
            if (I1 != 0) {
                auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");
                auto I2 = dist2 / activeGap2;

                auto rv0 = vtemp.pack<3>("x0", eem[0]);
                auto rv1 = vtemp.pack<3>("x0", eem[1]);
                auto rv2 = vtemp.pack<3>("x0", eem[2]);
                auto rv3 = vtemp.pack<3>("x0", eem[3]);
                T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
#if (RANK == 1)
                E = (2 - I1 / epsX) * (I1 / epsX) * -zs::sqr(activeGap2 - activeGap2 * I2) * zs::log(I2);
#elif (RANK == 2)
                E = (-(1 / (epsX * epsX)) * I1 * I1 + (2 / epsX) * I1) * zs::sqr(activeGap2 - activeGap2 * I2) * zs::sqr(zs::log(I2));
#elif (RANK == 4)
                E = (-(1 / (epsX * epsX)) * I1 * I1 + (2 / epsX) * I1) *
                    zs::sqr((activeGap2 - activeGap2 * I2) * zs::log(I2) * zs::log(I2));
#endif
            }
            reduce_to(eemi, n, E, es[eemi / 32]);
        });
        Es.push_back(reduce(pol, es) * kappa);

        auto numPPM = PPM.getCount();
        es.resize(count_warps(numPPM));
        es.reset(0);
        pol(range(numPPM), [vtemp = proxy<space>({}, vtemp), PPM = PPM.port(), es = proxy<space>(es), xi2 = xi * xi,
                            dHat = dHat, activeGap2, n = numPPM] __device__(int ppmi) mutable {
            auto ppm = PPM[ppmi];

            auto v0 = vtemp.pack<3>("xn", ppm[1]) - vtemp.pack<3>("xn", ppm[0]);
            auto v1 = vtemp.pack<3>("xn", ppm[3]) - vtemp.pack<3>("xn", ppm[2]);
            auto c = v0.cross(v1).norm();
            auto I1 = c * c;
            T E = 0;
            if (I1 != 0) {
                auto dist2 = dist2_pp(vtemp.pack<3>("xn", ppm[0]), vtemp.pack<3>("xn", ppm[2]));
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");
                auto I2 = dist2 / activeGap2;

                auto rv0 = vtemp.pack<3>("x0", ppm[0]);
                auto rv1 = vtemp.pack<3>("x0", ppm[1]);
                auto rv2 = vtemp.pack<3>("x0", ppm[2]);
                auto rv3 = vtemp.pack<3>("x0", ppm[3]);
                T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
#if (RANK == 1)
                E = (2 - I1 / epsX) * (I1 / epsX) * -zs::sqr(activeGap2 - activeGap2 * I2) * zs::log(I2);
#elif (RANK == 2)
                E = (-(1 / (epsX * epsX)) * I1 * I1 + (2 / epsX) * I1) * zs::sqr(activeGap2 - activeGap2 * I2) * zs::sqr(zs::log(I2));
#elif (RANK == 4)
                E = (-(1 / (epsX * epsX)) * I1 * I1 + (2 / epsX) * I1) *
                    zs::sqr((activeGap2 - activeGap2 * I2) * zs::log(I2) * zs::log(I2));
#endif
            }
            reduce_to(ppmi, n, E, es[ppmi / 32]);
        });
        Es.push_back(reduce(pol, es) * kappa);

        auto numPEM = PEM.getCount();
        es.resize(count_warps(numPEM));
        es.reset(0);
        pol(range(numPEM), [vtemp = proxy<space>({}, vtemp), PEM = PEM.port(), es = proxy<space>(es), xi2 = xi * xi,
                            dHat = dHat, activeGap2, n = numPEM] __device__(int pemi) mutable {
            auto pem = PEM[pemi];

            auto p = vtemp.pack<3>("xn", pem[0]);
            auto e0 = vtemp.pack<3>("xn", pem[2]);
            auto e1 = vtemp.pack<3>("xn", pem[3]);
            auto v0 = vtemp.pack<3>("xn", pem[1]) - p;
            auto v1 = e1 - e0;
            auto c = v0.cross(v1).norm();
            auto I1 = c * c;
            T E = 0;
            if (I1 != 0) {
                auto dist2 = dist2_pe(p, e0, e1);
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");
                auto I2 = dist2 / activeGap2;

                auto rv0 = vtemp.pack<3>("x0", pem[0]);
                auto rv1 = vtemp.pack<3>("x0", pem[1]);
                auto rv2 = vtemp.pack<3>("x0", pem[2]);
                auto rv3 = vtemp.pack<3>("x0", pem[3]);
                T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
#if (RANK == 1)
                E = (2 - I1 / epsX) * (I1 / epsX) * -zs::sqr(activeGap2 - activeGap2 * I2) * zs::log(I2);
#elif (RANK == 2)
                E = (-(1 / (epsX * epsX)) * I1 * I1 + (2 / epsX) * I1) * zs::sqr(activeGap2 - activeGap2 * I2) * zs::sqr(zs::log(I2));
#elif (RANK == 4)
                E = (-(1 / (epsX * epsX)) * I1 * I1 + (2 / epsX) * I1) *
                    zs::sqr((activeGap2 - activeGap2 * I2) * zs::log(I2) * zs::log(I2));
#endif
            }
            reduce_to(pemi, n, E, es[pemi / 32]);
        });
        Es.push_back(reduce(pol, es) * kappa);
    } // mollification

    if (s_enableFriction) {
        if (fricMu != 0) {
            if (s_enableSelfFriction) {
                auto numFPP = FPP.getCount();
                es.resize(count_warps(numFPP));
                es.reset(0);
                pol(range(numFPP),
                    [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP), FPP = FPP.port(),
                     es = proxy<space>(es), epsvh = epsv * dt, n = numFPP] __device__(int fppi) mutable {
                        auto fpp = FPP[fppi];
                        auto p0 = vtemp.pack<3>("xn", fpp[0]) - vtemp.pack<3>("xhat", fpp[0]);
                        auto p1 = vtemp.pack<3>("xn", fpp[1]) - vtemp.pack<3>("xhat", fpp[1]);
                        auto basis = fricPP.template pack<3, 2>("basis", fppi);
                        auto fn = fricPP("fn", fppi);
                        auto relDX3D = point_point_rel_dx(p0, p1);
                        auto relDX = basis.transpose() * relDX3D;
                        auto relDXNorm2 = relDX.l2NormSqr();
                        auto E = f0_SF(relDXNorm2, epsvh) * fn;
                        reduce_to(fppi, n, E, es[fppi / 32]);
                    });
                Es.push_back(reduce(pol, es) * fricMu);

                auto numFPE = FPE.getCount();
                es.resize(count_warps(numFPE));
                es.reset(0);
                pol(range(numFPE),
                    [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE), FPE = FPE.port(),
                     es = proxy<space>(es), epsvh = epsv * dt, n = numFPE] __device__(int fpei) mutable {
                        auto fpe = FPE[fpei];
                        auto p = vtemp.pack<3>("xn", fpe[0]) - vtemp.pack<3>("xhat", fpe[0]);
                        auto e0 = vtemp.pack<3>("xn", fpe[1]) - vtemp.pack<3>("xhat", fpe[1]);
                        auto e1 = vtemp.pack<3>("xn", fpe[2]) - vtemp.pack<3>("xhat", fpe[2]);
                        auto basis = fricPE.template pack<3, 2>("basis", fpei);
                        auto fn = fricPE("fn", fpei);
                        auto yita = fricPE("yita", fpei);
                        auto relDX3D = point_edge_rel_dx(p, e0, e1, yita);
                        auto relDX = basis.transpose() * relDX3D;
                        auto relDXNorm2 = relDX.l2NormSqr();
                        auto E = f0_SF(relDXNorm2, epsvh) * fn;
                        reduce_to(fpei, n, E, es[fpei / 32]);
                    });
                Es.push_back(reduce(pol, es) * fricMu);

                auto numFPT = FPT.getCount();
                es.resize(count_warps(numFPT));
                es.reset(0);
                pol(range(numFPT),
                    [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT), FPT = FPT.port(),
                     es = proxy<space>(es), epsvh = epsv * dt, n = numFPT] __device__(int fpti) mutable {
                        auto fpt = FPT[fpti];
                        auto p = vtemp.pack<3>("xn", fpt[0]) - vtemp.pack<3>("xhat", fpt[0]);
                        auto v0 = vtemp.pack<3>("xn", fpt[1]) - vtemp.pack<3>("xhat", fpt[1]);
                        auto v1 = vtemp.pack<3>("xn", fpt[2]) - vtemp.pack<3>("xhat", fpt[2]);
                        auto v2 = vtemp.pack<3>("xn", fpt[3]) - vtemp.pack<3>("xhat", fpt[3]);
                        auto basis = fricPT.template pack<3, 2>("basis", fpti);
                        auto fn = fricPT("fn", fpti);
                        auto betas = fricPT.pack(dim_c<2>, "beta", fpti);
                        auto relDX3D = point_triangle_rel_dx(p, v0, v1, v2, betas[0], betas[1]);
                        auto relDX = basis.transpose() * relDX3D;
                        auto relDXNorm2 = relDX.l2NormSqr();
                        auto E = f0_SF(relDXNorm2, epsvh) * fn;
                        reduce_to(fpti, n, E, es[fpti / 32]);
                    });
                Es.push_back(reduce(pol, es) * fricMu);

                auto numFEE = FEE.getCount();
                es.resize(count_warps(numFEE));
                es.reset(0);
                pol(range(numFEE),
                    [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE), FEE = FEE.port(),
                     es = proxy<space>(es), epsvh = epsv * dt, n = numFEE] __device__(int feei) mutable {
                        auto fee = FEE[feei];
                        auto e0 = vtemp.pack<3>("xn", fee[0]) - vtemp.pack<3>("xhat", fee[0]);
                        auto e1 = vtemp.pack<3>("xn", fee[1]) - vtemp.pack<3>("xhat", fee[1]);
                        auto e2 = vtemp.pack<3>("xn", fee[2]) - vtemp.pack<3>("xhat", fee[2]);
                        auto e3 = vtemp.pack<3>("xn", fee[3]) - vtemp.pack<3>("xhat", fee[3]);
                        auto basis = fricEE.template pack<3, 2>("basis", feei);
                        auto fn = fricEE("fn", feei);
                        auto gammas = fricEE.pack(dim_c<2>, "gamma", feei);
                        auto relDX3D = edge_edge_rel_dx(e0, e1, e2, e3, gammas[0], gammas[1]);
                        auto relDX = basis.transpose() * relDX3D;
                        auto relDXNorm2 = relDX.l2NormSqr();
                        auto E = f0_SF(relDXNorm2, epsvh) * fn;
                        reduce_to(feei, n, E, es[feei / 32]);
                    });
                Es.push_back(reduce(pol, es) * fricMu);
            }
        }
    } // fric

    std::sort(Es.begin(), Es.end());
    T E = 0;
    for (auto e : Es)
        E += e;
    return E;
}

void UnifiedIPCSystem::computeBarrierGradient(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    using mat12 = zs::vec<T, 12, 12>;
    using mat3 = zs::vec<T, 3, 3>;
    using Vec12View = zs::vec_view<T, zs::integer_sequence<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_sequence<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_sequence<int, 6>>;
    auto numPP = PP.getCount();
    pol(range(numPP), [vtemp = proxy<space>({}, vtemp), PP = PP.port(), gTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                       kappa = kappa] __device__(int ppi) mutable {
        auto pp = PP[ppi];
        auto x0 = vtemp.pack<3>("xn", pp[0]);
        auto x1 = vtemp.pack<3>("xn", pp[1]);
#if 1
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
#else
#endif
    });
    auto numPE = PE.getCount();
    pol(range(numPE), [vtemp = proxy<space>({}, vtemp), PE = PE.port(), gTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                       kappa = kappa] __device__(int pei) mutable {
        auto pe = PE[pei];
        auto p = vtemp.pack<3>("xn", pe[0]);
        auto e0 = vtemp.pack<3>("xn", pe[1]);
        auto e1 = vtemp.pack<3>("xn", pe[2]);
#if 1
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
#else
#endif
    });
    auto numPT = PT.getCount();
    pol(range(numPT), [vtemp = proxy<space>({}, vtemp), PT = PT.port(), gTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                       kappa = kappa] __device__(int pti) mutable {
        auto pt = PT[pti];
        auto p = vtemp.pack<3>("xn", pt[0]);
        auto t0 = vtemp.pack<3>("xn", pt[1]);
        auto t1 = vtemp.pack<3>("xn", pt[2]);
        auto t2 = vtemp.pack<3>("xn", pt[3]);
#if 1
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
#else
#endif
    });
    auto numEE = EE.getCount();
    pol(range(numEE), [vtemp = proxy<space>({}, vtemp), EE = EE.port(), gTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                       kappa = kappa] __device__(int eei) mutable {
        auto ee = EE[eei];
        auto ea0 = vtemp.pack<3>("xn", ee[0]);
        auto ea1 = vtemp.pack<3>("xn", ee[1]);
        auto eb0 = vtemp.pack<3>("xn", ee[2]);
        auto eb1 = vtemp.pack<3>("xn", ee[3]);
#if 1
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
#else
#endif
    });

    if (enableMollification) {
        auto get_mollifier = [] ZS_LAMBDA(const auto &ea0Rest, const auto &ea1Rest, const auto &eb0Rest,
                                          const auto &eb1Rest, const auto &ea0, const auto &ea1, const auto &eb0,
                                          const auto &eb1) {
            T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
            return zs::make_tuple(mollifier_ee(ea0, ea1, eb0, eb1, epsX), mollifier_grad_ee(ea0, ea1, eb0, eb1, epsX),
                                  mollifier_hess_ee(ea0, ea1, eb0, eb1, epsX));
        };
        auto numEEM = EEM.getCount();
        pol(range(numEEM), [vtemp = proxy<space>({}, vtemp), EEM = EEM.port(), gTag, xi2 = xi * xi, dHat = dHat,
                            activeGap2, kappa = kappa, get_mollifier] __device__(int eemi) mutable {
            auto eem = EEM[eemi]; // <x, y, z, w>
            auto ea0Rest = vtemp.pack<3>("x0", eem[0]);
            auto ea1Rest = vtemp.pack<3>("x0", eem[1]);
            auto eb0Rest = vtemp.pack<3>("x0", eem[2]);
            auto eb1Rest = vtemp.pack<3>("x0", eem[3]);
            auto ea0 = vtemp.pack<3>("xn", eem[0]);
            auto ea1 = vtemp.pack<3>("xn", eem[1]);
            auto eb0 = vtemp.pack<3>("xn", eem[2]);
            auto eb1 = vtemp.pack<3>("xn", eem[3]);
#if 1
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
#else
#endif
        });
        auto numPPM = PPM.getCount();
        pol(range(numPPM), [vtemp = proxy<space>({}, vtemp), PPM = PPM.port(), gTag, xi2 = xi * xi, dHat = dHat,
                            activeGap2, kappa = kappa, get_mollifier] __device__(int ppmi) mutable {
            auto ppm = PPM[ppmi]; // <x, z, y, w>, <0, 2, 1, 3>
            auto ea0Rest = vtemp.pack<3>("x0", ppm[0]);
            auto ea1Rest = vtemp.pack<3>("x0", ppm[1]);
            auto eb0Rest = vtemp.pack<3>("x0", ppm[2]);
            auto eb1Rest = vtemp.pack<3>("x0", ppm[3]);
            auto ea0 = vtemp.pack<3>("xn", ppm[0]);
            auto ea1 = vtemp.pack<3>("xn", ppm[1]);
            auto eb0 = vtemp.pack<3>("xn", ppm[2]);
            auto eb1 = vtemp.pack<3>("xn", ppm[3]);
#if 1
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
#else
#endif
        });
        auto numPEM = PEM.getCount();
        pol(range(numPEM), [vtemp = proxy<space>({}, vtemp), PEM = PEM.port(), gTag, xi2 = xi * xi, dHat = dHat,
                            activeGap2, kappa = kappa, get_mollifier] __device__(int pemi) mutable {
            auto pem = PEM[pemi]; // <x, w, y, z>, <0, 2, 3, 1>
            auto ea0Rest = vtemp.pack<3>("x0", pem[0]);
            auto ea1Rest = vtemp.pack<3>("x0", pem[1]);
            auto eb0Rest = vtemp.pack<3>("x0", pem[2]);
            auto eb1Rest = vtemp.pack<3>("x0", pem[3]);
            auto ea0 = vtemp.pack<3>("xn", pem[0]);
            auto ea1 = vtemp.pack<3>("xn", pem[1]);
            auto eb0 = vtemp.pack<3>("xn", pem[2]);
            auto eb1 = vtemp.pack<3>("xn", pem[3]);
#if 1
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
#else
#endif
        });
    }
    return;
}

void UnifiedIPCSystem::computeFrictionBarrierGradient(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    using Vec12View = zs::vec_view<T, zs::integer_sequence<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_sequence<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_sequence<int, 6>>;
    auto numFPP = FPP.getCount();
    pol(range(numFPP), [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP), FPP = FPP.port(), gTag,
                        epsvh = epsv * dt, fricMu = fricMu] __device__(int fppi) mutable {
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
    });
    auto numFPE = FPE.getCount();
    pol(range(numFPE), [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE), FPE = FPE.port(), gTag,
                        epsvh = epsv * dt, fricMu = fricMu] __device__(int fpei) mutable {
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
    });
    auto numFPT = FPT.getCount();
    pol(range(numFPT), [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT), FPT = FPT.port(), gTag,
                        epsvh = epsv * dt, fricMu = fricMu] __device__(int fpti) mutable {
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
    });
    auto numFEE = FEE.getCount();
    pol(range(numFEE), [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE), FEE = FEE.port(), gTag,
                        epsvh = epsv * dt, fricMu = fricMu] __device__(int feei) mutable {
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
    });
    return;
}

template <typename LinSysT, typename VecTM, typename VecTI,
          zs::enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTI::dim == 1, VecTI::extent * 3 == VecTM::template range_t<0>::value> = 0>
__forceinline__ __device__ void add_hessian_slow(LinSysT &sys, const VecTI &inds, const VecTM &hess) {
    using namespace zs;
    constexpr int codim = VecTI::extent;
    using mat3 = typename LinSysT::mat3;
    using pair_t = typename LinSysT::pair_t;
    using dyn_hess_t = typename LinSysT::dyn_hess_t;
    auto &spmat = sys.spmat;
    const auto nnz = spmat.nnz();
    auto &dynHess = sys.dynHess;
#pragma unroll
    for (int i = 0; i != codim; ++i) {
        auto subOffset = i * 3;
        auto row = inds[i];
        // diagonal
        auto loc = spmat._ptrs[row];
        auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
        for (int r = 0; r != 3; ++r)
            for (int c = 0; c != 3; ++c) {
                atomic_add(exec_cuda, &mat(r, c), hess(subOffset + r, subOffset + c));
            }
        // non-diagonal
        for (int j = i + 1; j < codim; ++j) {
            mat3 subBlock;
            for (int r = 0; r != 3; ++r)
                for (int c = 0; c != 3; ++c)
                    subBlock(r, c) = hess(subOffset + r, j * 3 + c);

            auto col = inds[j];
            if (row < col) {
                if (auto loc = spmat.locate(row, col, zs::true_c); loc >= nnz) {
                    // not exist in spmat
                    dynHess.try_push(zs::make_tuple(pair_t{row, col}, subBlock));
                } else {
                    // exist in spmat
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    for (int r = 0; r != 3; ++r)
                        for (int c = 0; c != 3; ++c) {
                            atomic_add(exec_cuda, &mat(r, c), subBlock(r, c));
                        }
                }
            } else {
                if (auto loc = spmat.locate(col, row, zs::true_c); loc >= nnz) {
                    // not exist in spmat
                    dynHess.try_push(zs::make_tuple(pair_t{row, col}, subBlock));
                } else {
                    // exist in spmat
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    for (int c = 0; c != 3; ++c)
                        for (int r = 0; r != 3; ++r) {
                            atomic_add(exec_cuda, &mat(c, r), subBlock(r, c));
                        }
                }
            }
        }
    }
}
/// tile version
template <typename LinSysT, typename VecTM, typename VecTI,
          zs::enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTI::dim == 1, VecTI::extent * 3 == VecTM::template range_t<0>::value> = 0>
__forceinline__ __device__ void
add_hessian(cooperative_groups::thread_block_tile<8, cooperative_groups::thread_block> &tile, LinSysT &sys,
            const VecTI &inds, const VecTM &hess) {
    using namespace zs;
    constexpr int codim = VecTI::extent;
    using mat3 = typename LinSysT::mat3;
    using pair_t = typename LinSysT::pair_t;
    using dyn_hess_t = typename LinSysT::dyn_hess_t;
    auto &spmat = sys.spmat;
    const auto nnz = spmat.nnz();
    auto &dynHess = sys.dynHess;
    const int cap = __popc(tile.ballot(1)); // assume active pattern 0...001111 [15, 14, ..., 0]
    auto laneId = tile.thread_rank();
#pragma unroll
    for (int i = 0; i != codim; ++i) {
        auto subOffsetI = i * 3;
        auto row = inds[i];
        // diagonal
        auto loc = spmat._ptrs[row];
        auto &mat = const_cast<mat3 &>(spmat._vals[loc]);

        for (int d = laneId; d < 9; d += cap) {
            atomic_add(exec_cuda, &mat(d / 3, d % 3), hess(subOffsetI + d / 3, subOffsetI + d % 3));
        }
        // non-diagonal
        for (int j = i + 1; j < codim; ++j) {
            auto subOffsetJ = j * 3;
#if 0
            mat3 subBlock;
            for (int r = 0; r != 3; ++r)
                for (int c = 0; c != 3; ++c)
                    subBlock(r, c) = hess(subOffsetI + r, subOffsetJ + c);
#endif

            auto col = inds[j];
            if (row < col) {
                if (auto loc = spmat.locate(row, col, zs::true_c); loc >= nnz) {
                    // not exist in spmat
                    auto no = dynHess.next_index(tile);
                    auto &[inds, mat] = dynHess[no];
                    for (int d = laneId; d < 9; d += cap)
                        // mat.val(d) = subBlock.val(d);
                        mat.val(d) = hess(subOffsetI + d / 3, subOffsetJ + d % 3);
                    if (laneId == 0)
                        inds = pair_t{row, col};
                } else {
                    // exist in spmat
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    for (int d = laneId; d < 9; d += cap)
                        // atomic_add(exec_cuda, &mat.val(d), subBlock(d / 3, d % 3));
                        atomic_add(exec_cuda, &mat.val(d), hess(subOffsetI + d / 3, subOffsetJ + d % 3));
                }
            } else {
                if (auto loc = spmat.locate(col, row, zs::true_c); loc >= nnz) {
                    // not exist in spmat
                    auto no = dynHess.next_index(tile);
                    auto &[inds, mat] = dynHess[no];
                    for (int d = laneId; d < 9; d += cap)
                        // mat.val(d) = subBlock.val(d);
                        mat.val(d) = hess(subOffsetI + d / 3, subOffsetJ + d % 3);
                    if (laneId == 0)
                        inds = pair_t{row, col};
                } else {
                    // exist in spmat
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    for (int d = laneId; d < 9; d += cap)
                        // atomic_add(exec_cuda, &mat.val(d), subBlock(d % 3, d / 3));
                        atomic_add(exec_cuda, &mat.val(d), hess(subOffsetI + d % 3, subOffsetJ + d / 3));
                }
            }
        }
    }
}
template <typename T, zs::enable_if_t<std::is_fundamental_v<T>> = 0>
__forceinline__ __device__ T tile_shfl(cooperative_groups::thread_block_tile<8, cooperative_groups::thread_block> &tile,
                                       T var, int srcLane) {
    return tile.shfl(var, srcLane);
}
template <typename VecT, zs::enable_if_t<zs::is_vec<VecT>::value> = 0>
__forceinline__ __device__ VecT tile_shfl(
    cooperative_groups::thread_block_tile<8, cooperative_groups::thread_block> &tile, const VecT &var, int srcLane) {
    VecT ret{};
    for (typename VecT::index_type i = 0; i != VecT::extent; ++i)
        ret.val(i) = tile_shfl(tile, var.val(i), srcLane);
    return ret;
}
template <typename LinSysT, typename VecTM, typename VecTI,
          zs::enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTI::dim == 1, VecTI::extent * 3 == VecTM::template range_t<0>::value> = 0>
__forceinline__ __device__ void add_hessian(LinSysT &sys, const VecTI &inds, const VecTM &hess) {
    using namespace zs;
    using mat3 = typename LinSysT::mat3;
    using pair_t = typename LinSysT::pair_t;
    using dyn_hess_t = typename LinSysT::dyn_hess_t;
    auto &spmat = sys.spmat;
    auto tile = cg::tiled_partition<8>(cg::this_thread_block());

    bool has_work = true; // is this visible to rest threads in tile cuz of __forceinline__ ??

    u32 work_queue = tile.ballot(has_work);
    while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_work = tile_shfl(tile, hess, cur_rank);
        auto cur_index = tile.shfl(inds, cur_rank); // gather index as well
        add_hessian(tile, sys, cur_index, cur_work);
        // add_hessian0(sys, cur_index, cur_work);

        if (tile.thread_rank() == cur_rank)
            has_work = false;
        work_queue = tile.ballot(has_work);
    }
    return;
}

void UnifiedIPCSystem::updateBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    using mat12 = zs::vec<T, 12, 12>;
    using mat3 = zs::vec<T, 3, 3>;
    using Vec12View = zs::vec_view<T, zs::integer_sequence<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_sequence<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_sequence<int, 6>>;

    auto &dynHess = linsys.dynHess;

    auto numPP = PP.getCount();
    dynHess.reserveFor(numPP);
    pol(range(numPP), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), PP = PP.port(), gTag,
                       xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa] __device__(int ppi) mutable {
        auto pp = PP[ppi];
        auto x0 = vtemp.pack<3>("xn", pp[0]);
        auto x1 = vtemp.pack<3>("xn", pp[1]);
#if 0
        auto ppGrad = dist_grad_pp(x0, x1);
        auto dist2 = dist2_pp(x0, x1);
        if (dist2 < xi2)
            printf("dist already smaller than xi!\n");
        auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, kappa);
        /// gradient
        auto grad = ppGrad * (-barrierDistGrad);
        // gradient
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pp[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pp[1]), grad(1, d));
        }

        /// hessian
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
        auto DmInv = 1 / u0;
        auto F = Ds * DmInv;
        T I5 = F.dot(F);

        auto tmp = F * 2;
#if (RANK == 1)
        vec3 flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (I5 + 2 * I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 2)
        vec3 flatten_pk1 = (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * (I5 - 1) * (I5 + I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 4)
        vec3 flatten_pk1 = (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * zs::log(I5) * (I5 - 1) * (2 * I5 + I5 * zs::log(I5) - 2)) / I5 * tmp;
#endif

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

#if (RANK == 1)
        T lambda0 =
            kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
#elif (RANK == 2)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * I5 + zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 6 * I5 * zs::log(I5) - 2 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 7 * I5 * I5 * zs::log(I5) - 2)) / I5;
        if (dis * dis < gassThreshold * activeGap2)
            lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * gassThreshold + zs::log(gassThreshold) - 3 * gassThreshold * gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) + 6 * gassThreshold * zs::log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) - 7 * gassThreshold * gassThreshold * zs::log(gassThreshold) - 2)) / gassThreshold;
#elif (RANK == 4)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * (24 * I5 + 2 * zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 12 * I5 * zs::log(I5) - 12 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 14 * I5 * I5 * zs::log(I5) - 12)) / I5;
#endif
        auto Q0 = F / zs::sqrt(I5);
        auto H = lambda0 * dyadic_prod(Q0, Q0);
        auto ppHess = PFPx.transpose() * H * PFPx;
#endif
        // pp[0], pp[1]
        add_hessian(sysHess, pp, ppHess);
    });
    auto numPE = PE.getCount();
    // auto peOffset = hess3.increaseCount(numPE);
    // , hess3 = proxy<space>(hess3)
    dynHess.reserveFor(numPE * 3);
    pol(range(numPE), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), PE = PE.port(), gTag,
                       xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa] __device__(int pei) mutable {
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
        /// gradient
        auto grad = peGrad * (-barrierDistGrad);
        // gradient
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[2]), grad(2, d));
        }

        /// hessian
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
        } else {
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
#if (RANK == 1)
        flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (I5 + 2 * I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 2)
        flatten_pk1 = (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * (I5 - 1) * (I5 + I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 4)
        flatten_pk1 = (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * zs::log(I5) * (I5 - 1) * (2 * I5 + I5 * zs::log(I5) - 2)) / I5 * tmp;
#endif

        zs::vec<T, 6, 9> PFPx = dFdXMatrix(DmInv, wrapv<3>{});

        auto grad = -PFPx.transpose() * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[0]), grad(d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[1]), grad(3 + d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[2]), grad(6 + d));
        }

#if (RANK == 1)
        T lambda0 =
            kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
#elif (RANK == 2)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * I5 + zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 6 * I5 * zs::log(I5) - 2 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 7 * I5 * I5 * zs::log(I5) - 2)) / I5;
        if (dis * dis < gassThreshold * activeGap2) {
            lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * gassThreshold + zs::log(gassThreshold) - 3 * gassThreshold * gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) + 6 * gassThreshold * zs::log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) - 7 * gassThreshold * gassThreshold * zs::log(gassThreshold) - 2)) / gassThreshold;
        }
#elif (RANK == 4)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * (24 * I5 + 2 * zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 12 * I5 * zs::log(I5) - 12 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 14 * I5 * I5 * zs::log(I5) - 12)) / I5;
#endif
        auto q0 = flatten(fnn) / zs::sqrt(I5);
        auto H = lambda0 * dyadic_prod(q0, q0);
        auto peHess = PFPx.transpose() * H * PFPx;
#endif
        // pe[0], pe[1], pe[2]
        add_hessian(sysHess, pe, peHess);
    });
    auto numPT = PT.getCount();
    dynHess.reserveFor(numPT * 6);
    pol(range(numPT), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), PT = PT.port(), gTag,
                       xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa] __device__(int pti) mutable {
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
        /// gradient
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[3]), grad(3, d));
        }

        /// hessian
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
#if (RANK == 1)
            flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (I5 + 2 * I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 2)
            flatten_pk1 = (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * (I5 - 1) * (I5 + I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 4)
            flatten_pk1 = (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * zs::log(I5) * (I5 - 1) * (2 * I5 + I5 * zs::log(I5) - 2)) / I5 * tmp;
#endif
        }

        auto PFPx = dFdXMatrix(DmInv, wrapv<3>{});

        auto grad = -PFPx.transpose() * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[0]), grad(d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[1]), grad(3 + d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[2]), grad(6 + d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[3]), grad(9 + d));
        }
#if (RANK == 1)
        T lambda0 =
            kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
#elif (RANK == 2)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * I5 + zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 6 * I5 * zs::log(I5) - 2 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 7 * I5 * I5 * zs::log(I5) - 2)) / I5;
        if (dis * dis < gassThreshold * dHat) {
            lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * gassThreshold + zs::log(gassThreshold) - 3 * gassThreshold * gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) + 6 * gassThreshold * zs::log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) - 7 * gassThreshold * gassThreshold * zs::log(gassThreshold) - 2)) / gassThreshold;
        }
#elif (RANK == 4)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * (24 * I5 + 2 * zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 12 * I5 * zs::log(I5) - 12 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 14 * I5 * I5 * zs::log(I5) - 12)) / I5;
#endif
        auto q0 = flatten(F * dyadic_prod(normal, normal)) / zs::sqrt(I5);
        auto ptHess = PFPx.transpose() * (lambda0 * dyadic_prod(q0, q0)) * PFPx;
#endif
        // pt[0], pt[1], pt[2], pt[3]
        add_hessian(sysHess, pt, ptHess);
    });
    auto numEE = EE.getCount();
    dynHess.reserveFor(numEE * 6);
    pol(range(numEE), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), EE = EE.port(), gTag,
                       xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa] __device__(int eei) mutable {
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
        /// gradient
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[3]), grad(3, d));
        }
        /// hessian
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
#if (RANK == 1)
            flatten_pk1 =
                -2 * kappa * (activeGap2 * activeGap2 * (I5 - 1) * (I5 + 2 * I5 * zs::log(I5) - 1) / I5) * tmp;
#elif (RANK == 2)
            flatten_pk1 = 2 * (2 * kappa * activeGap2 * activeGap2 * zs::log(I5) * (I5 - 1) * (I5 + I5 * zs::log(I5) - 1)) / I5 * tmp;
#elif (RANK == 4)
            flatten_pk1 = 4 * kappa *
                          (activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * zs::log(I5) * (I5 - 1) *
                           (2 * I5 + I5 * zs::log(I5) - 2)) /
                          I5 * tmp;
#endif
        }

        zs::vec<T, 9, 12> PFPx = dFdXMatrix(DmInv, wrapv<3>{});

        auto grad = -PFPx.transpose() * flatten_pk1;
        for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[0]), grad(d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[1]), grad(3 + d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[2]), grad(6 + d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[3]), grad(9 + d));
        }

#if (RANK == 1)
        T lambda0 =
            kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
#elif (RANK == 2)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * I5 + zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 6 * I5 * zs::log(I5) - 2 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 7 * I5 * I5 * zs::log(I5) - 2)) / I5;
        if (dis * dis < gassThreshold * activeGap2)
            lambda0 = -(4 * kappa * activeGap2 * activeGap2 * (4 * gassThreshold + zs::log(gassThreshold) - 3 * gassThreshold * gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) + 6 * gassThreshold * zs::log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * zs::log(gassThreshold) * zs::log(gassThreshold) - 7 * gassThreshold * gassThreshold * zs::log(gassThreshold) - 2)) / gassThreshold;
#elif (RANK == 4)
        T lambda0 = -(4 * kappa * activeGap2 * activeGap2 * zs::log(I5) * zs::log(I5) * (24 * I5 + 2 * zs::log(I5) - 3 * I5 * I5 * zs::log(I5) * zs::log(I5) + 12 * I5 * zs::log(I5) - 12 * I5 * I5 + I5 * zs::log(I5) * zs::log(I5) - 14 * I5 * I5 * zs::log(I5) - 12)) / I5;
#endif

        if (lambda0 < 0)
            printf("FUCKING WRONG EEHESS! lambda0 = %e, I5 = %e\n", lambda0, I5);

        auto nn = dyadic_prod(normal, normal);
        auto fnn = F * nn;
        auto q0 = flatten(fnn) / zs::sqrt(I5);
        auto eeHess = PFPx.transpose() * (lambda0 * dyadic_prod(q0, q0)) * PFPx;
#endif
        // ee[0], ee[1], ee[2], ee[3]
        add_hessian(sysHess, ee, eeHess);
    });

    if (enableMollification) {
        auto get_mollifier = [] ZS_LAMBDA(const auto &ea0Rest, const auto &ea1Rest, const auto &eb0Rest,
                                          const auto &eb1Rest, const auto &ea0, const auto &ea1, const auto &eb0,
                                          const auto &eb1) {
            T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
            return zs::make_tuple(mollifier_ee(ea0, ea1, eb0, eb1, epsX), mollifier_grad_ee(ea0, ea1, eb0, eb1, epsX),
                                  mollifier_hess_ee(ea0, ea1, eb0, eb1, epsX));
        };
        auto numEEM = EEM.getCount();
        dynHess.reserveFor(numEEM * 6);
        pol(range(numEEM), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), EEM = EEM.port(), gTag,
                            xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa,
                            get_mollifier] __device__(int eemi) mutable {
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

            /// gradient
            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledEEGrad = mollifierEE * barrierDistGrad * eeGrad;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[0]), -(scaledMollifierGrad(0, d) + scaledEEGrad(0, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[1]), -(scaledMollifierGrad(1, d) + scaledEEGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[2]), -(scaledMollifierGrad(2, d) + scaledEEGrad(2, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[3]), -(scaledMollifierGrad(3, d) + scaledEEGrad(3, d)));
            }

            /// hessian
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

#if (RANK == 1)
            T p1 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                   (eps_x * eps_x);
            T p2 = kappa * 2 *
                   (I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) /
                   (I2 * eps_x * eps_x);
#elif (RANK == 2)
            T p1 = -kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T p2 = -kappa * 2 * (2 * I1 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * zs::log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T p1 = -kappa * 2 * (2 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T p2 = -kappa * 2 * (2 * I1 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#endif

            auto flatten_pk1 = flatten_g1 * p1 + flatten_g2 * p2;
            auto grad = -PFPx * flatten_pk1;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[0]), grad[d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[1]), grad[3 + d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[2]), grad[6 + d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, eem[3]), grad[9 + d]);
            }

            // hessian
#if (RANK == 1)
            T lambda10 = kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                         (eps_x * eps_x); // p1
            T lambda11 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                         (eps_x * eps_x);
            T lambda12 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                         (eps_x * eps_x);
#elif (RANK == 2)
            T lambda10 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            T lambda11 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T lambda12 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 4)
            T lambda10 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            T lambda11 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T lambda12 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#endif

            auto fnn = F * n1n1;
            auto q10 = flatten(fnn);
            q10 /= c;

            mat3 Tx{0, 0, 0, 0, 0, 1, 0, -1, 0};
            mat3 Ty{0, 0, -1, 0, 0, 0, 1, 0, 0};
            mat3 Tz{0, 1, 0, -1, 0, 0, 0, 0, 0};
            constexpr auto ratio = (T)1 / (T)g_sqrt2;
            Tx *= ratio;
            Ty *= ratio; // ?
            Tz *= ratio;

            auto q11 = flatten(Tx * fnn).normalized();
            auto q12 = flatten(Tz * fnn).normalized();

            auto M9_temp = dyadic_prod(q11, q11) * lambda11;
            auto projectedH = M9_temp;
            projectedH += dyadic_prod(q12, q12) * lambda12;

#if (RANK == 1)
            T lambda20 = -kappa *
                         (2 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) *
                          (6 * I2 + 2 * I2 * zs::log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * zs::log(I2) + 1)) /
                         (I2 * eps_x * eps_x);
#elif (RANK == 2)
            T lambda20 = kappa * (4 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (4 * I2 + zs::log(I2) - 3 * I2 * I2 * zs::log(I2) * zs::log(I2) + 6 * I2 * zs::log(I2) - 2 * I2 * I2 + I2 * zs::log(I2) * zs::log(I2) - 7 * I2 * I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T lambda20 = kappa * (4 * I1 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - 2 * eps_x) * (24 * I2 + 2 * zs::log(I2) - 3 * I2 * I2 * zs::log(I2) * zs::log(I2) + 12 * I2 * zs::log(I2) - 12 * I2 * I2 + I2 * zs::log(I2) * zs::log(I2) - 14 * I2 * I2 * zs::log(I2) - 12)) / (I2 * (eps_x * eps_x));
#endif

            fnn = F * n2n2;
            auto q20 = flatten(fnn) / (dis / dHat); // sqrt(I2)

#if (RANK == 1)
            T lambdag1g = kappa * 4 * c * F(2, 2) *
                          ((2 * activeGap2 * activeGap2 * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) /
                           (I2 * eps_x * eps_x));
#elif (RANK == 2)
            T lambdag1g = -kappa * 4 * c * F(2, 2) * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * zs::log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T lambdag1g = -kappa * 4 * c * F(2, 2) * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#endif
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
            // ee[0], ee[1], ee[2], ee[3]
            add_hessian(sysHess, eem, eemHess);
        });
        auto numPPM = PPM.getCount();
        dynHess.reserveFor(numPPM * 6);
        pol(range(numPPM), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), PPM = PPM.port(), gTag,
                            xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa,
                            get_mollifier] __device__(int ppmi) mutable {
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

            /// gradient
            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledPPGrad = mollifierEE * barrierDistGrad * ppGrad;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[0]), -(scaledMollifierGrad(0, d) + scaledPPGrad(0, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[1]), -(scaledMollifierGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[2]), -(scaledMollifierGrad(2, d) + scaledPPGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[3]), -(scaledMollifierGrad(3, d)));
            }

            /// hessian
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

#if (RANK == 1)
            T p1 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                   (eps_x * eps_x);
            T p2 = kappa * 2 *
                   (I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) /
                   (I2 * eps_x * eps_x);
#elif (RANK == 2)
            T p1 = -kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T p2 = -kappa * 2 * (2 * I1 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * zs::log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T p1 = -kappa * 2 * (2 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T p2 = -kappa * 2 * (2 * I1 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#endif

            auto flatten_pk1 = flatten_g1 * p1 + flatten_g2 * p2;
            auto grad = -PFPx * flatten_pk1;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[0]), grad[d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[1]), grad[3 + d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[2]), grad[6 + d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, ppm[3]), grad[9 + d]);
            }

            // hessian
#if (RANK == 1)
            T lambda10 = kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                         (eps_x * eps_x); // p1
            T lambda11 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                         (eps_x * eps_x);
            T lambda12 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                         (eps_x * eps_x);
#elif (RANK == 2)
            T lambda10 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            T lambda11 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T lambda12 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 4)
            T lambda10 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            T lambda11 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T lambda12 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#endif

            auto fnn = F * n1n1;
            auto q10 = flatten(fnn);
            q10 /= c;

            mat3 Tx{0, 0, 0, 0, 0, 1, 0, -1, 0};
            mat3 Ty{0, 0, -1, 0, 0, 0, 1, 0, 0};
            mat3 Tz{0, 1, 0, -1, 0, 0, 0, 0, 0};
            constexpr auto ratio = (T)1 / (T)g_sqrt2;
            Tx *= ratio;
            Ty *= ratio; // ?
            Tz *= ratio;

            auto q11 = flatten(Tx * fnn).normalized();
            auto q12 = flatten(Tz * fnn).normalized();

            auto M9_temp = dyadic_prod(q11, q11) * lambda11;
            auto projectedH = M9_temp;
            projectedH += dyadic_prod(q12, q12) * lambda12;

#if (RANK == 1)
            T lambda20 = -kappa *
                         (2 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) *
                          (6 * I2 + 2 * I2 * zs::log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * zs::log(I2) + 1)) /
                         (I2 * eps_x * eps_x);
#elif (RANK == 2)
            T lambda20 = kappa * (4 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (4 * I2 + zs::log(I2) - 3 * I2 * I2 * zs::log(I2) * zs::log(I2) + 6 * I2 * zs::log(I2) - 2 * I2 * I2 + I2 * zs::log(I2) * zs::log(I2) - 7 * I2 * I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T lambda20 = kappa * (4 * I1 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - 2 * eps_x) * (24 * I2 + 2 * zs::log(I2) - 3 * I2 * I2 * zs::log(I2) * zs::log(I2) + 12 * I2 * zs::log(I2) - 12 * I2 * I2 + I2 * zs::log(I2) * zs::log(I2) - 14 * I2 * I2 * zs::log(I2) - 12)) / (I2 * (eps_x * eps_x));
#endif
            fnn = F * n2n2;
            auto q20 = flatten(fnn) / (dis / dHat); // sqrt(I2)

#if (RANK == 1)
            T lambdag1g = kappa * 4 * c * F(2, 2) *
                          ((2 * activeGap2 * activeGap2 * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) /
                           (I2 * eps_x * eps_x));
#elif (RANK == 2)
            T lambdag1g = -kappa * 4 * c * F(2, 2) * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * zs::log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T lambdag1g = -kappa * 4 * c * F(2, 2) * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#endif
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
            // ee[0], ee[1], ee[2], ee[3]
            add_hessian(sysHess, ppm, ppmHess);
        });
        auto numPEM = PEM.getCount();
        dynHess.reserveFor(numPEM * 6);
        pol(range(numPEM), [vtemp = proxy<space>({}, vtemp), sysHess = port<space>(linsys), PEM = PEM.port(), gTag,
                            xi2 = xi * xi, dHat = dHat, activeGap2, kappa = kappa,
                            get_mollifier] __device__(int pemi) mutable {
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

            /// gradient
            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledPEGrad = mollifierEE * barrierDistGrad * peGrad;

            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[0]), -(scaledMollifierGrad(0, d) + scaledPEGrad(0, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[1]), -(scaledMollifierGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[2]), -(scaledMollifierGrad(2, d) + scaledPEGrad(1, d)));
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[3]), -(scaledMollifierGrad(3, d) + scaledPEGrad(2, d)));
            }

            /// hessian
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

#if (RANK == 1)
            T p1 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                   (eps_x * eps_x);
            T p2 = kappa * 2 *
                   (I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) /
                   (I2 * eps_x * eps_x);
#elif (RANK == 2)
            T p1 = -kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T p2 = -kappa * 2 * (2 * I1 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * zs::log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T p1 = -kappa * 2 * (2 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T p2 = -kappa * 2 * (2 * I1 * activeGap2 * activeGap2 * zs::pow(log(I2), (T)3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#endif

            auto flatten_pk1 = flatten_g1 * p1 + flatten_g2 * p2;
            auto grad = -PFPx * flatten_pk1;
            for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[0]), grad[d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[1]), grad[3 + d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[2]), grad[6 + d]);
                atomic_add(exec_cuda, &vtemp(gTag, d, pem[3]), grad[9 + d]);
            }

// hessian
#if (RANK == 1)
            T lambda10 = kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                         (eps_x * eps_x); // p1
            T lambda11 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                         (eps_x * eps_x);
            T lambda12 = kappa * 2 * (2 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                         (eps_x * eps_x);
#elif (RANK == 2)
            T lambda10 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            T lambda11 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T lambda12 = -kappa * (4 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 4)
            T lambda10 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            T lambda11 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            T lambda12 = -kappa * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#endif

            auto fnn = F * n1n1;
            auto q10 = flatten(fnn);
            q10 /= c;

            mat3 Tx{0, 0, 0, 0, 0, 1, 0, -1, 0};
            mat3 Ty{0, 0, -1, 0, 0, 0, 1, 0, 0};
            mat3 Tz{0, 1, 0, -1, 0, 0, 0, 0, 0};
            constexpr auto ratio = (T)1 / (T)g_sqrt2;
            Tx *= ratio;
            Ty *= ratio; // ?
            Tz *= ratio;

            auto q11 = flatten(Tx * fnn).normalized();
            auto q12 = flatten(Tz * fnn).normalized();

            auto M9_temp = dyadic_prod(q11, q11) * lambda11;
            auto projectedH = M9_temp;
            projectedH += dyadic_prod(q12, q12) * lambda12;

#if (RANK == 1)
            T lambda20 = -kappa *
                         (2 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) *
                          (6 * I2 + 2 * I2 * zs::log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * zs::log(I2) + 1)) /
                         (I2 * eps_x * eps_x);
#elif (RANK == 2)
            T lambda20 = kappa * (4 * I1 * activeGap2 * activeGap2 * (I1 - 2 * eps_x) * (4 * I2 + zs::log(I2) - 3 * I2 * I2 * zs::log(I2) * zs::log(I2) + 6 * I2 * zs::log(I2) - 2 * I2 * I2 + I2 * zs::log(I2) * zs::log(I2) - 7 * I2 * I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T lambda20 = kappa * (4 * I1 * activeGap2 * activeGap2 * zs::log(I2) * zs::log(I2) * (I1 - 2 * eps_x) * (24 * I2 + 2 * zs::log(I2) - 3 * I2 * I2 * zs::log(I2) * zs::log(I2) + 12 * I2 * zs::log(I2) - 12 * I2 * I2 + I2 * zs::log(I2) * zs::log(I2) - 14 * I2 * I2 * zs::log(I2) - 12)) / (I2 * (eps_x * eps_x));
#endif

            fnn = F * n2n2;
            auto q20 = flatten(fnn) / (dis / dHat); // sqrt(I2)

#if (RANK == 1)
            T lambdag1g = kappa * 4 * c * F(2, 2) *
                          ((2 * activeGap2 * activeGap2 * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * zs::log(I2) - 1)) /
                           (I2 * eps_x * eps_x));
#elif (RANK == 2)
            T lambdag1g = -kappa * 4 * c * F(2, 2) * (4 * activeGap2 * activeGap2 * zs::log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * zs::log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            T lambdag1g = -kappa * 4 * c * F(2, 2) * (4 * activeGap2 * activeGap2 * zs::pow(zs::log(I2), (T)3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * zs::log(I2) - 2)) / (I2 * (eps_x * eps_x));
#endif
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
            // ee[0], ee[1], ee[2], ee[3]
            add_hessian(sysHess, pem, pemHess);
        });
    }
    return;
}

void UnifiedIPCSystem::updateFrictionBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol,
                                                               const zs::SmallString &gTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    using Vec12View = zs::vec_view<T, zs::integer_sequence<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_sequence<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_sequence<int, 6>>;

    auto &dynHess = linsys.dynHess;

    auto numFPP = FPP.getCount();
    dynHess.reserveFor(numFPP);
    pol(range(numFPP),
        [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP), sysHess = port<space>(linsys),
         FPP = FPP.port(), gTag, epsvh = epsv * dt, fricMu = fricMu] __device__(int fppi) mutable {
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
            // pp[0], pp[1]
            add_hessian(sysHess, fpp, hess);
        });
    auto numFPE = FPE.getCount();
    dynHess.reserveFor(numFPE * 3);
    pol(range(numFPE),
        [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE), sysHess = port<space>(linsys),
         FPE = FPE.port(), gTag, epsvh = epsv * dt, fricMu = fricMu] __device__(int fpei) mutable {
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
            // pe[0], pe[1], pe[2]
            add_hessian(sysHess, fpe, hess);
        });
    auto numFPT = FPT.getCount();
    dynHess.reserveFor(numFPT * 6);
    pol(range(numFPT),
        [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT), sysHess = port<space>(linsys),
         FPT = FPT.port(), gTag, epsvh = epsv * dt, fricMu = fricMu] __device__(int fpti) mutable {
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
            // pt[0], pt[1], pt[2], pt[3]
            add_hessian(sysHess, fpt, hess);
        });
    auto numFEE = FEE.getCount();
    dynHess.reserveFor(numFEE * 6);
    pol(range(numFEE),
        [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE), sysHess = port<space>(linsys),
         FEE = FEE.port(), gTag, epsvh = epsv * dt, fricMu = fricMu] __device__(int feei) mutable {
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

            // ee[0], ee[1], ee[2], ee[3]
            add_hessian(sysHess, fee, hess);
        });
    return;
}

} // namespace zeno