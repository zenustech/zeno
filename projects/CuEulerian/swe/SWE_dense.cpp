#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include "../scheme.hpp"

namespace zeno {

#define SW_OPEN_BOUNDARY 1

struct SolveShallowWaterHeight : INode {
    void height_flux(float *flx, float *flz, float *h, float *u, float *w, int nx, int nz, int halo) {
        auto pol = zs::omp_exec();
        pol(zs::Collapse{nz + 1, nx + 1}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };
            i += halo / 2;
            j += halo / 2;

            float u_adv = u[idx(i, j)];
            float w_adv = w[idx(i, j)];

            if (u_adv < 0) {
                flx[idx(i, j)] = u_adv * scheme::TVD_MUSCL3(h[idx(i - 1, j)], h[idx(i, j)], h[idx(i + 1, j)]);
            } else {
                flx[idx(i, j)] = u_adv * scheme::TVD_MUSCL3(h[idx(i, j)], h[idx(i - 1, j)], h[idx(i - 2, j)]);
            }

            if (w_adv < 0) {
                flz[idx(i, j)] = w_adv * scheme::TVD_MUSCL3(h[idx(i, j - 1)], h[idx(i, j)], h[idx(i, j + 1)]);
            } else {
                flz[idx(i, j)] = w_adv * scheme::TVD_MUSCL3(h[idx(i, j)], h[idx(i, j - 1)], h[idx(i, j - 2)]);
            }
        });
    }

    void height_integral(float *h_new, float *h_old, float *h_n, float *flx, float *flz, int nx, int nz, int halo,
                         float dx, float dt, float c0, float c1) {
        auto pol = zs::omp_exec();
        pol(zs::Collapse{nz, nx}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };
            i += halo / 2;
            j += halo / 2;

            h_new[idx(i, j)] =
                c0 * h_n[idx(i, j)] +
                c1 * (h_old[idx(i, j)] +
                      (flx[idx(i, j)] - flx[idx(i + 1, j)] + flz[idx(i, j)] - flz[idx(i, j + 1)]) / dx * dt);
        });
    }

    void boundary_height(float *h, int nx, int nz, int halo) {
        auto pol = zs::omp_exec();

        // x boundary
        pol(zs::Collapse{nz + halo, halo}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };

            if (i < halo / 2) {
                // left
                h[idx(i, j)] = h[idx(halo - 1 - i, j)];
            } else {
                // right
                i += nx;
                h[idx(i, j)] = h[idx(2 * nx + halo - 1 - i, j)];
            }
        });

        // z boundary
        pol(zs::Collapse{halo, nx + halo}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };

            if (j < halo / 2) {
                // front
                h[idx(i, j)] = h[idx(i, halo - 1 - j)];
            } else {
                // back
                j += nz;
                h[idx(i, j)] = h[idx(i, 2 * nz + halo - 1 - j)];
            }
        });
    }

    void apply() override {
        auto grid = get_input<PrimitiveObject>("SWGrid");
        auto &ud = grid->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")) || (!ud.has<int>("halo")) || (!ud.has<float>("dx")))
            zeno::log_error("no such UserData named '{}', '{}', '{}' or '{}'.", "nx", "nz", "halo", "dx");
        int nx = ud.get2<int>("nx");
        int nz = ud.get2<int>("nz");
        int halo = ud.get2<int>("halo");
        float dx = ud.get2<float>("dx");
        auto dt = get_input2<float>("dt");

        const unsigned int nc = (nx + halo) * (nz + halo);

        auto height_attr = get_input2<std::string>("height_attr");
        auto u_attr = get_input2<std::string>("u_attr");
        auto w_attr = get_input2<std::string>("w_attr");

        auto &h_old = grid->verts.attr<float>(height_attr);
        auto &u = grid->verts.attr<float>(u_attr);
        auto &w = grid->verts.attr<float>(w_attr);
        std::vector<float> h_new(nc), h_rk(nc), flx(nc), flz(nc);

        // 3rd-order 3-stage TVD Runge-Kutta method
        // 1st stage h_old --> h_new
        height_flux(flx.data(), flz.data(), h_old.data(), u.data(), w.data(), nx, nz, halo);
        height_integral(h_new.data(), h_old.data(), h_old.data(), flx.data(), flz.data(), nx, nz, halo, dx, dt, 0.f,
                        1.f);
        boundary_height(h_new.data(), nx, nz, halo);

        // 2nd stage h_new --> h_rk
        height_flux(flx.data(), flz.data(), h_new.data(), u.data(), w.data(), nx, nz, halo);
        height_integral(h_rk.data(), h_new.data(), h_old.data(), flx.data(), flz.data(), nx, nz, halo, dx, dt,
                        3.f / 4.f, 1.f / 4.f);
        boundary_height(h_rk.data(), nx, nz, halo);

        // 3rd stage h_rk --> h_new
        height_flux(flx.data(), flz.data(), h_rk.data(), u.data(), w.data(), nx, nz, halo);
        height_integral(h_new.data(), h_rk.data(), h_old.data(), flx.data(), flz.data(), nx, nz, halo, dx, dt,
                        1.f / 3.f, 2.f / 3.f);
        boundary_height(h_new.data(), nx, nz, halo);

        std::swap(h_old, h_new);

        set_output("SWGrid", std::move(grid));
    }
};

ZENDEFNODE(SolveShallowWaterHeight, {/* inputs: */
                                     {
                                         "SWGrid",
                                         {"float", "dt", "0.04"},
                                         {"string", "height_attr", "height"},
                                         {"string", "u_attr", "u"},
                                         {"string", "w_attr", "w"},
                                     },
                                     /* outputs: */
                                     {
                                         "SWGrid",
                                     },
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});

struct SolveShallowWaterMomentum : INode {
    void boundary_velocity(float *u, float *w, int nx, int nz, int halo) {
        auto pol = zs::omp_exec();

        // x boundary
        pol(zs::Collapse{nz + halo, halo}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };

            if (i < halo / 2) {
                // left
#if SW_OPEN_BOUNDARY
                u[idx(i, j)] = -std::abs(u[idx(halo - 1 - i, j)]);
                if (i == halo / 2 - 1) {
                    u[idx(i + 1, j)] = -std::abs(u[idx(i + 2, j)]);
                }
#else
                u[idx(i, j)] = -u[idx(halo - i, j)];
                if (i == halo / 2 - 1) {
                    u[idx(i + 1, j)] = 0;
                }
#endif
                w[idx(i, j)] = w[idx(halo - 1 - i, j)];
            } else {
                // right
                i += nx;
#if SW_OPEN_BOUNDARY
                u[idx(i, j)] = std::abs(u[idx(2 * nx + halo - 1 - i, j)]);
#else
                if (i == nx + halo / 2) {
                    u[idx(i, j)] = 0;
                } else {
                    u[idx(i, j)] = -u[idx(2 * nx + halo - i, j)];
                }
#endif
                w[idx(i, j)] = w[idx(2 * nx + halo - 1 - i, j)];
            }
        });

        // z boundary
        pol(zs::Collapse{halo, nx + halo}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };

            if (j < halo / 2) {
                // front
                u[idx(i, j)] = u[idx(i, halo - 1 - j)];
#if SW_OPEN_BOUNDARY
                w[idx(i, j)] = -std::abs(w[idx(i, halo - 1 - j)]);
                if (j == halo / 2 - 1) {
                    w[idx(i, j + 1)] = -std::abs(w[idx(i, j + 2)]);
                }
#else
                w[idx(i, j)] = -w[idx(i, halo - j)];
                if (j == halo / 2 - 1) {
                    w[idx(i, j + 1)] = 0;
                }
#endif
            } else {
                // back
                j += nz;
                u[idx(i, j)] = u[idx(i, 2 * nz + halo - 1 - j)];
#if SW_OPEN_BOUNDARY
                w[idx(i, j)] = std::abs(w[idx(i, 2 * nz + halo - 1 - j)]);
#else
                if (j == nz + halo / 2) {
                    w[idx(i, j)] = 0;
                } else {
                    w[idx(i, j)] = -w[idx(i, 2 * nz + halo - j)];
                }
#endif
            }
        });
    }

    void momentum_stencil(float *u_new, float *w_new, float *u_old, float *w_old, float *u_n, float *w_n, float *h,
                          float *B, float gravity, int nx, int nz, int halo, float dx, float dt, float c0, float c1) {
        auto pol = zs::omp_exec();
        pol(zs::Collapse{nz, nx}, [&](int j, int i) {
            auto idx = [=](auto i, auto j) { return j * (nx + halo) + i; };
            i += halo / 2;
            j += halo / 2;

            float u_adv, w_adv, adv_term, grad_term;
            int upwind;

            // update u
            u_adv = u_old[idx(i, j)];
            w_adv = 0.25f * (w_old[idx(i - 1, j)] + w_old[idx(i, j)] + w_old[idx(i - 1, j + 1)] + w_old[idx(i, j + 1)]);

            adv_term = 0;
            upwind = u_adv < 0 ? 1 : -1;
            adv_term += u_adv * scheme::HJ_WENO3(u_old[idx(i - upwind, j)], u_old[idx(i, j)], u_old[idx(i + upwind, j)],
                                                 u_old[idx(i + 2 * upwind, j)], u_adv, dx);
            upwind = w_adv < 0 ? 1 : -1;
            adv_term += w_adv * scheme::HJ_WENO3(u_old[idx(i, j - upwind)], u_old[idx(i, j)], u_old[idx(i, j + upwind)],
                                                 u_old[idx(i, j + 2 * upwind)], w_adv, dx);
            grad_term = gravity * 0.5f * (h[idx(i, j)] + h[idx(i - 1, j)]) *
                        ((h[idx(i, j)] - h[idx(i - 1, j)]) / dx + (B[idx(i, j)] - B[idx(i - 1, j)]) / dx);

            u_new[idx(i, j)] = c0 * u_n[idx(i, j)] + c1 * (u_old[idx(i, j)] - (adv_term + grad_term) * dt);

            // update w
            u_adv = 0.25f * (u_old[idx(i, j - 1)] + u_old[idx(i, j)] + u_old[idx(i + 1, j - 1)] + u_old[idx(i + 1, j)]);
            w_adv = w_old[idx(i, j)];

            adv_term = 0;
            upwind = u_adv < 0 ? 1 : -1;
            adv_term += u_adv * scheme::HJ_WENO3(w_old[idx(i - upwind, j)], w_old[idx(i, j)], w_old[idx(i + upwind, j)],
                                                 w_old[idx(i + 2 * upwind, j)], u_adv, dx);
            upwind = w_adv < 0 ? 1 : -1;
            adv_term += w_adv * scheme::HJ_WENO3(w_old[idx(i, j - upwind)], w_old[idx(i, j)], w_old[idx(i, j + upwind)],
                                                 w_old[idx(i, j + 2 * upwind)], w_adv, dx);
            grad_term = gravity * 0.5f * (h[idx(i, j)] + h[idx(i, j - 1)]) *
                        ((h[idx(i, j)] - h[idx(i, j - 1)]) / dx + (B[idx(i, j)] - B[idx(i, j - 1)]) / dx);

            w_new[idx(i, j)] = c0 * w_n[idx(i, j)] + c1 * (w_old[idx(i, j)] - (adv_term + grad_term) * dt);
        });
    };

    void apply() override {
        auto grid = get_input<PrimitiveObject>("SWGrid");
        auto &ud = grid->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")) || (!ud.has<int>("halo")) || (!ud.has<float>("dx")))
            zeno::log_error("no such UserData named '{}', '{}', '{}' or '{}'.", "nx", "nz", "halo", "dx");
        int nx = ud.get2<int>("nx");
        int nz = ud.get2<int>("nz");
        int halo = ud.get2<int>("halo");
        float dx = ud.get2<float>("dx");
        auto dt = get_input2<float>("dt");
        auto gravity = get_input2<float>("gravity");

        const unsigned int nc = (nx + halo) * (nz + halo);

        auto terrain_attr = get_input2<std::string>("terrain_attr");
        auto height_attr = get_input2<std::string>("height_attr");
        auto u_attr = get_input2<std::string>("u_attr");
        auto w_attr = get_input2<std::string>("w_attr");

        auto &B = grid->verts.attr<float>(terrain_attr);
        auto &h = grid->verts.attr<float>(height_attr);
        auto &u_old = grid->verts.attr<float>(u_attr);
        auto &w_old = grid->verts.attr<float>(w_attr);
        std::vector<float> u_new(nc), w_new(nc), u_rk(nc), w_rk(nc);

        // 3rd-order 3-stage TVD Runge-Kutta method
        // 1st stage u_old, w_old --> u_new, w_new
        momentum_stencil(u_new.data(), w_new.data(), u_old.data(), w_old.data(), u_old.data(), w_old.data(), h.data(),
                         B.data(), gravity, nx, nz, halo, dx, dt, 0.f, 1.f);
        boundary_velocity(u_new.data(), w_new.data(), nx, nz, halo);

        // 2nd stage u_new, w_new --> u_rk, w_rk
        momentum_stencil(u_rk.data(), w_rk.data(), u_new.data(), w_new.data(), u_old.data(), w_old.data(), h.data(),
                         B.data(), gravity, nx, nz, halo, dx, dt, 3.f / 4.f, 1.f / 4.f);
        boundary_velocity(u_rk.data(), w_rk.data(), nx, nz, halo);

        // 3rd stage u_rk, w_rk --> u_new, w_new
        momentum_stencil(u_new.data(), w_new.data(), u_rk.data(), w_rk.data(), u_old.data(), w_old.data(), h.data(),
                         B.data(), gravity, nx, nz, halo, dx, dt, 1.f / 3.f, 2.f / 3.f);
        boundary_velocity(u_new.data(), w_new.data(), nx, nz, halo);

        std::swap(u_old, u_new);
        std::swap(w_old, w_new);

        set_output("SWGrid", std::move(grid));
    }
};

ZENDEFNODE(SolveShallowWaterMomentum, {/* inputs: */
                                       {
                                           "SWGrid",
                                           {"float", "dt", "0.04"},
                                           {"float", "gravity", "9.8"},
                                           {"string", "terrain_attr", "terrain"},
                                           {"string", "height_attr", "height"},
                                           {"string", "u_attr", "u"},
                                           {"string", "w_attr", "w"},
                                       },
                                       /* outputs: */
                                       {
                                           "SWGrid",
                                       },
                                       /* params: */
                                       {},
                                       /* category: */
                                       {"Eulerian"}});

} // namespace zeno