#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ZSParticleToZSGrid : INode {
  void p2g_momentum(zs::CudaExecutionPolicy &cudaPol,
                    const typename ZenoParticles::particles_t &pars,
                    const typename ZenoPartition::table_t &partition,
                    typename ZenoGrid::grid_t &grid,
                    bool isAffineAugmented = true) {
    using namespace zs;
    cudaPol(range(pars.size()),
            [pars = proxy<execspace_e::cuda>({}, pars),
             table = proxy<execspace_e::cuda>(partition),
             grid = proxy<execspace_e::cuda>({}, grid), dxinv = 1.f / grid.dx,
             isAffineAugmented] __device__(size_t pi) mutable {
              using grid_t = RM_CVREF_T(grid);
              using mat3 = zs::vec<float, 3, 3>;
              const auto Dinv = 4.f * dxinv * dxinv;
              auto pos = pars.pack<3>("x", pi);
              auto vel = pars.pack<3>("v", pi);
              auto mass = pars("m", pi);
              auto C =
                  isAffineAugmented ? pars.pack<3, 3>("C", pi) : mat3::zeros();

              auto arena = make_local_arena(grid.dx, pos);

              for (auto loc : arena.range()) {
                auto coord = arena.coord(loc);
                auto localIndex = coord & (grid_t::side_length - 1);
                auto blockno = table.query(coord - localIndex);
                if (blockno < 0)
                  printf("THE HELL!");
                auto block = grid.block(blockno);

                auto xixp = arena.diff(loc);
                auto W = arena.weight(loc);
                const auto cellid = grid_t::coord_to_cellid(localIndex);
                atomic_add(exec_cuda, &block("m", cellid), mass * W);
                auto Cxixp = C * xixp;
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &block("v", d, cellid),
                             W * mass * (vel[d] + Cxixp[d]));
              }
            });
  }
  void p2g_curve_force(zs::CudaExecutionPolicy &cudaPol,
                       const zs::StvkWithHencky<float> &model,
                       const typename ZenoParticles::particles_t &verts,
                       const typename ZenoParticles::particles_t &eles,
                       const typename ZenoPartition::table_t &partition,
                       const float dt, typename ZenoGrid::grid_t &grid,
                       bool isFlipStyle = false,
                       bool isAffineAugmented = true) {
    using namespace zs;
    bool materialParamOverride =
        eles.hasProperty("mu") && eles.hasProperty("lam");
    SmallString vtag = isFlipStyle ? "vstar" : "v";
    cudaPol(range(eles.size()),
            [verts = proxy<execspace_e::cuda>({}, verts),
             eles = proxy<execspace_e::cuda>({}, eles),
             table = proxy<execspace_e::cuda>(partition),
             grid = proxy<execspace_e::cuda>({}, grid), model = model,
             materialParamOverride, dt, dxinv = 1.f / grid.dx, vtag,
             isAffineAugmented] __device__(size_t pi) mutable {
              using grid_t = RM_CVREF_T(grid);
              using mat2 = zs::vec<float, 2, 2>;
              using mat3 = zs::vec<float, 3, 3>;
              const auto Dinv = 4.f * dxinv * dxinv;
              auto pos = eles.pack<3>("x", pi);
              auto vel = eles.pack<3>("v", pi);
              auto mass = eles("m", pi);
              auto vol = eles("vol", pi);
              auto C =
                  isAffineAugmented ? eles.pack<3, 3>("C", pi) : mat3::zeros();
              auto F = eles.pack<3, 3>("F", pi);
              auto d_ = eles.pack<3, 3>("d", pi);

              // hard coded P compute
              constexpr auto gamma = 10.f;
              constexpr auto k = 2000.f;
              auto [Q, R] = math::gram_schmidt(F);
              mat2 R2{R(1, 1), R(1, 2), 0, R(2, 2)};
              if (materialParamOverride) {
                model.mu = eles("mu", pi);
                model.lam = eles("lam", pi);
              }
              auto P2 = model.first_piola(R2); // use as F
              auto P = mat3::zeros();
#if 1
              P(0, 0) = k * (R(0, 0) - 1);
#else
              if (R(0, 0) < 1) // bent (compressed)
                P(0, 0) = 0.01f * zs::max(R(0, 0) - 1, -0.3f);
#endif
              P(1, 1) = P2(0, 0);
              P(1, 2) = P2(0, 1);
              P(2, 1) = P2(1, 0);
              P(2, 2) = P2(1, 1);
              P = Q * P;

              float rr = R(0, 1) * R(0, 1) + R(0, 2) * R(0, 2); // shearing
              float gp = gamma;

              auto A = mat3::zeros();
              A(0, 0) = rr * gp;
              A(0, 1) = (R(1, 1) * R(0, 1) + R(1, 2) * R(0, 2)) * gp;
              A(0, 2) = R(2, 2) * R(0, 2) * gp;
              A(1, 0) = A(0, 1);
              A(2, 0) = A(0, 2);
              // P += Q * A * inverse(R).transpose();
              auto invJ = R(0, 0) * R(1, 1) * R(2, 2);
              if (invJ != 0.f)
                invJ = 1.f / invJ;
              mat3 Rit{};
              Rit(0, 0) = R(1, 1) * R(2, 2);
              Rit(0, 1) = -R(1, 0) * R(2, 2);
              Rit(0, 2) = R(1, 0) * R(2, 1) - R(1, 1) * R(2, 0);
              Rit(1, 0) = 0;
              Rit(1, 1) = R(0, 0) * R(2, 2);
              Rit(1, 2) = -R(0, 0) * R(2, 1);
              Rit(2, 0) = Rit(2, 1) = 0;
              Rit(2, 2) = R(0, 0) * R(1, 1);
              P += Q * A * invJ * Rit;

              //
              zs::vec<float, 3, 2> P_c23{P(0, 1), P(0, 2), P(1, 1),
                                         P(1, 2), P(2, 1), P(2, 2)};
              zs::vec<float, 2, 3> d_c23T{d_(0, 1), d_(1, 1), d_(2, 1),
                                          d_(0, 2), d_(1, 2), d_(2, 2)};

              auto arena =
                  make_local_arena<grid_e::collocated, kernel_e::quadratic, 1>(
                      grid.dx, pos);
              // compression
              for (auto loc : arena.range()) {
                auto coord = arena.coord(loc);
                auto localIndex = coord & (grid_t::side_length - 1);
                auto blockno = table.query(coord - localIndex);
                if (blockno < 0)
                  printf("THE HELL!");
                auto block = grid.block(blockno);

                auto Wmass = arena.weight(loc) * mass;
                auto xixp = arena.diff(loc);
                auto Wgrad = arena.weightGradients(loc) * dxinv;
                const auto cellid = grid_t::coord_to_cellid(localIndex);

                auto vft = P_c23 * (d_c23T * Wgrad) * (-vol * dt);
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &block(vtag, d, cellid), (float)vft(d));

                atomic_add(exec_cuda, &block("m", cellid), Wmass);
                auto Cxixp = C * xixp;
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &block("v", d, cellid),
                             Wmass * (vel[d] + Cxixp[d]));
              }

              // type (ii)
              auto transfer = [P_c0 = col(P, 0), &grid, &table,
                               &vtag](const auto &pos, auto D_beta_eps,
                                      const auto coeff) {
                auto vft = coeff * P_c0 * D_beta_eps;
                auto arena = make_local_arena(grid.dx, pos);

                for (auto loc : arena.range()) {
                  auto coord = arena.coord(loc);
                  auto localIndex = coord & (grid_t::side_length - 1);
                  auto blockno = table.query(coord - localIndex);
                  if (blockno < 0)
                    printf("THE HELL!");
                  auto block = grid.block(blockno);

                  auto W = arena.weight(loc);
                  const auto cellid = grid_t::coord_to_cellid(localIndex);
                  for (int d = 0; d != 3; ++d) {
                    atomic_add(exec_cuda, &block(vtag, d, cellid),
                               (float)(W * vft[d]));
                  }
                }
              };
              auto Dminv00 = eles("DmInv", pi);
              auto inds =
                  eles.pack<2>("inds", pi).template reinterpret_bits<int>();
              transfer(verts.pack<3>("x", inds[0]), Dminv00, vol * dt);
              transfer(verts.pack<3>("x", inds[1]), Dminv00, -vol * dt);
            });
  }
  void p2g_vert_bending_force(zs::CudaExecutionPolicy &cudaPol,
                              const typename ZenoParticles::particles_t &verts,
                              const typename ZenoParticles::particles_t &eles,
                              const typename ZenoPartition::table_t &partition,
                              const float dt, typename ZenoGrid::grid_t &grid,
                              float thickness, bool isFlipStyle = false) {
    using namespace zs;
    SmallString vtag = isFlipStyle ? "vstar" : "v";
    cudaPol(range(eles.size()), [verts = proxy<execspace_e::cuda>({}, verts),
                                 eles = proxy<execspace_e::cuda>({}, eles),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid),
                                 vtag, dxinv = 1.f / grid.dx, thickness,
                                 dt] __device__(size_t ei) mutable {
      using grid_t = RM_CVREF_T(grid);
      using vec3 = zs::vec<float, 3>;
      auto k = eles("k", ei);
      auto restLength = eles("rl", ei);
      auto vinds = eles.pack<2>("vinds", ei).reinterpret_bits<int>();

      auto x0 = verts.pack<3>("x", vinds[0]);
      auto x1 = verts.pack<3>("x", vinds[1]);
      auto dir = x1 - x0;
      auto len = dir.norm();
      dir /= len;
      auto vfdt = len < restLength ? (thickness * thickness * len) * k *
                                         (1.f - len / restLength) * dt * dir
                                   : vec3::zeros();

      auto transfer = [&grid, &table, &vtag](const auto &pos, const auto &vft) {
        auto arena = make_local_arena(grid.dx, pos);

        for (auto loc : arena.range()) {
          auto coord = arena.coord(loc);
          auto localIndex = coord & (grid_t::side_length - 1);
          auto blockno = table.query(coord - localIndex);
          if (blockno < 0)
            printf("THE HELL!");
          auto block = grid.block(blockno);
          auto W = arena.weight(loc);
          const auto cellid = grid_t::coord_to_cellid(localIndex);
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &block(vtag, d, cellid), (float)(W * vft[d]));
          }
        }
      };
      transfer(x0, -vfdt);
      transfer(x1, vfdt);
    });
  }
  void
  p2g_element_bending_force(zs::CudaExecutionPolicy &cudaPol,
                            const typename ZenoParticles::particles_t &tris,
                            const typename ZenoParticles::particles_t &eles,
                            const typename ZenoPartition::table_t &partition,
                            const float dt, typename ZenoGrid::grid_t &grid,
                            float thickness, bool isFlipStyle = false) {
    using namespace zs;
    SmallString vtag = isFlipStyle ? "vstar" : "v";
    cudaPol(range(eles.size()), [tris = proxy<execspace_e::cuda>({}, tris),
                                 eles = proxy<execspace_e::cuda>({}, eles),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid),
                                 vtag, dxinv = 1.f / grid.dx, thickness,
                                 dt] __device__(size_t ei) mutable {
      using grid_t = RM_CVREF_T(grid);
      using vec3 = zs::vec<float, 3>;
      auto k = eles("k", ei);
      auto restLength = eles("rl", ei);
      auto einds = eles.pack<2>("einds", ei).reinterpret_bits<int>();

      auto x0 = tris.pack<3>("x", einds[0]);
      auto x1 = tris.pack<3>("x", einds[1]);
      auto dir = x1 - x0;
      auto len = dir.norm();
      dir /= len;
      auto vfdt = len < restLength ? (thickness * thickness * len) * k *
                                         (1.f - len / restLength) * dt * dir
                                   : vec3::zeros();

      auto transfer = [&grid, &table, &vtag](const auto &pos, const auto &vft) {
        auto arena = make_local_arena(grid.dx, pos);

        for (auto loc : arena.range()) {
          auto coord = arena.coord(loc);
          auto localIndex = coord & (grid_t::side_length - 1);
          auto blockno = table.query(coord - localIndex);
          if (blockno < 0)
            printf("THE HELL!");
          auto block = grid.block(blockno);
          auto W = arena.weight(loc);
          const auto cellid = grid_t::coord_to_cellid(localIndex);
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &block(vtag, d, cellid), (float)(W * vft[d]));
          }
        }
      };
      transfer(x0, -vfdt);
      transfer(x1, vfdt);
    });
  }
  void p2g_angle_bending_force(zs::CudaExecutionPolicy &cudaPol,
                               const typename ZenoParticles::particles_t &verts,
                               const typename ZenoParticles::particles_t &eles,
                               const typename ZenoPartition::table_t &partition,
                               const float dt, typename ZenoGrid::grid_t &grid,
                               bool isFlipStyle = false) {
    using namespace zs;
    SmallString vtag = isFlipStyle ? "vstar" : "v";
    cudaPol(range(eles.size()), [verts = proxy<execspace_e::cuda>({}, verts),
                                 eles = proxy<execspace_e::cuda>({}, eles),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx,
                                 vtag] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      using mat2 = zs::vec<float, 2, 2>;
      using mat3 = zs::vec<float, 3, 3>;
      using vec3 = zs::vec<float, 3>;
      const auto Dinv = 4.f * dxinv * dxinv;
      auto k = eles("k", pi);
      auto restAngle = eles("ra", pi);
      auto vinds = eles.pack<4>("vinds", pi).reinterpret_bits<int>();
      // here, v0->v1 is the hinge
      auto v0 = verts.pack<3>("x", vinds[0]);
      auto v1 = verts.pack<3>("x", vinds[1]);
      auto v2 = verts.pack<3>("x", vinds[2]);
      auto v3 = verts.pack<3>("x", vinds[3]);
      float angle{}, edgeNrm{}, h{};
      {
        auto n1 = (v0 - v2).cross(v1 - v2);
        auto n2 = (v1 - v3).cross(v0 - v3); // <v2, v1, v3>
        auto angle = zs::acos(
            zs::max(-1.f, zs::min(1.f, n1.dot(n2) / zs::sqrt(n1.l2NormSqr() *
                                                             n2.l2NormSqr()))));
        if (n2.cross(n1).dot(v0 - v1) < 0) // towards "closing"
          angle = -angle;
        edgeNrm = (v1 - v0).norm();
        h = (n1.norm() + n2.norm()) / (edgeNrm * 6);
      }
      auto coeff = 2 * (angle - restAngle) * k * edgeNrm / h * dt;
      /**
       *              v1 - e4 - v3
       *             /  \ [n2] /
       *           e3   e0   e2
       *           / [n1] \  /
       *         v2 - e1 - v0
       */
      auto e0 = v1 - v0;
      auto e1 = v2 - v0;
      auto e2 = v3 - v0;
      auto e3 = v2 - v1;
      auto e4 = v3 - v1;
      auto n1 = e0.cross(e1);
      auto n2 = e2.cross(e0);
      auto n1SqNorm = n1.l2NormSqr();
      auto n2SqNorm = n2.l2NormSqr();
      auto e0norm = e0.norm();
      auto da_dv2 = -e0norm / n1SqNorm * n1;
      auto da_dv0 = -e0.dot(e3) / (e0norm * n1SqNorm) * n1 -
                    e0.dot(e4) / (e0norm * n2SqNorm) * n2;
      auto da_dv1 = e0.dot(e1) / (e0norm * n1SqNorm) * n1 +
                    e0.dot(e2) / (e0norm * n2SqNorm) * n2;
      auto da_dv3 = -e0norm / n2SqNorm * n2;

      auto transfer = [&grid, &table, &vtag](const auto &pos, const auto &vft) {
        auto arena = make_local_arena(grid.dx, pos);

        for (auto loc : arena.range()) {
          auto coord = arena.coord(loc);
          auto localIndex = coord & (grid_t::side_length - 1);
          auto blockno = table.query(coord - localIndex);
          if (blockno < 0)
            printf("THE HELL!");
          auto block = grid.block(blockno);
          auto W = arena.weight(loc);
          const auto cellid = grid_t::coord_to_cellid(localIndex);
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &block(vtag, d, cellid), (float)(W * vft[d]));
          }
        }
      };
      transfer(verts.pack<3>("x", vinds[0]), coeff * da_dv0);
      transfer(verts.pack<3>("x", vinds[1]), coeff * da_dv1);
      transfer(verts.pack<3>("x", vinds[2]), coeff * da_dv2);
      transfer(verts.pack<3>("x", vinds[3]), coeff * da_dv3);
    });
  }
  template <typename Model>
  void p2g_surface_force(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                         const typename ZenoParticles::particles_t &verts,
                         const typename ZenoParticles::particles_t &eles,
                         const typename ZenoPartition::table_t &partition,
                         const float dt, typename ZenoGrid::grid_t &grid,
                         bool isFlipStyle = false,
                         bool isAffineAugmented = true) {
    using namespace zs;
    bool materialParamOverride =
        eles.hasProperty("mu") && eles.hasProperty("lam");
    SmallString vtag = isFlipStyle ? "vstar" : "v";
    cudaPol(
        range(eles.size()),
        [verts = proxy<execspace_e::cuda>({}, verts),
         eles = proxy<execspace_e::cuda>({}, eles),
         table = proxy<execspace_e::cuda>(partition),
         grid = proxy<execspace_e::cuda>({}, grid), model = model,
         materialParamOverride, dt, dxinv = 1.f / grid.dx, vtag,
         isAffineAugmented] __device__(size_t pi) mutable {
          using grid_t = RM_CVREF_T(grid);
          using mat3 = zs::vec<float, 3, 3>;
          const auto Dinv = 4.f * dxinv * dxinv;
          auto pos = eles.pack<3>("x", pi);
          auto vel = eles.pack<3>("v", pi);
          auto mass = eles("m", pi);
          auto vol = eles("vol", pi);
          auto C = isAffineAugmented ? eles.pack<3, 3>("C", pi) : mat3::zeros();
          auto F = eles.pack<3, 3>("F", pi);
          auto d_ = eles.pack<3, 3>("d", pi);

          // hard coded P compute
          using mat2 = zs::vec<float, 2, 2>;
          constexpr auto gamma = 0.f;
          constexpr auto k = 40000.f;
          auto [Q, R] = math::gram_schmidt(F);
          mat2 R2{R(0, 0), R(0, 1), R(1, 0), R(1, 1)};
          if (materialParamOverride) {
            model.mu = eles("mu", pi);
            model.lam = eles("lam", pi);
          }
          auto P2 = model.first_piola(R2); // use as F
          auto Pplane = mat3::zeros();
          Pplane(0, 0) = P2(0, 0);
          Pplane(0, 1) = P2(0, 1);
          Pplane(1, 0) = P2(1, 0);
          Pplane(1, 1) = P2(1, 1);
          Pplane = Q * Pplane; // inplane

          float rr = R(0, 2) * R(0, 2) + R(1, 2) * R(1, 2); // shearing
          float gg = gamma;

          float gf = 0.f;
          if (R(2, 2) < 1) { // compression
            const auto v = 1.f - R(2, 2);
            gf = -k * v * v;
          }

          auto A = mat3::zeros();
          A(0, 0) = gg * R(0, 2) * R(0, 2);
          A(0, 1) = gg * R(0, 2) * R(1, 2);
          A(0, 2) = gg * R(0, 2) * R(2, 2);
          A(1, 1) = gg * R(1, 2) * R(1, 2);
          A(1, 2) = gg * R(1, 2) * R(2, 2);
          A(2, 2) = gf * R(2, 2);
          A(1, 0) = A(0, 1);
          A(2, 0) = A(0, 2);
          A(2, 1) = A(1, 2);
          auto P = Pplane + Q * A * inverse(R).transpose();

          //
          auto P_c3 = col(P, 2);
          auto d_c3 = col(d_, 2);

          auto arena =
              make_local_arena<grid_e::collocated, kernel_e::quadratic, 1>(
                  grid.dx, pos);
          // compression
          for (auto loc : arena.range()) {
            auto coord = arena.coord(loc);
            auto localIndex = coord & (grid_t::side_length - 1);
            auto blockno = table.query(coord - localIndex);
            if (blockno < 0)
              printf("THE HELL!");
            auto block = grid.block(blockno);

            auto Wmass = arena.weight(loc) * mass;
            auto xixp = arena.diff(loc);
            auto Wgrad = arena.weightGradients(loc) * dxinv;
            const auto cellid = grid_t::coord_to_cellid(localIndex);

            auto vft = P_c3 * Wgrad.dot(d_c3) * (-vol * dt);
            for (int d = 0; d != 3; ++d)
              atomic_add(exec_cuda, &block(vtag, d, cellid), (float)vft(d));

            atomic_add(exec_cuda, &block("m", cellid), Wmass);
            auto Cxixp = C * xixp;
            for (int d = 0; d != 3; ++d)
              atomic_add(exec_cuda, &block("v", d, cellid),
                         Wmass * (vel[d] + Cxixp[d]));
          }

          // type (ii)
          auto transfer = [&P, &grid, &table, &vtag](const auto &pos,
                                                     const auto &Dinv_r,
                                                     const auto coeff) {
            auto vft = coeff * zs::vec<float, 3>{
                                   P(0, 0) * Dinv_r(0) + P(0, 1) * Dinv_r(1),
                                   P(1, 0) * Dinv_r(0) + P(1, 1) * Dinv_r(1),
                                   P(2, 0) * Dinv_r(0) + P(2, 1) * Dinv_r(1)};
            auto arena = make_local_arena(grid.dx, pos);

            for (auto loc : arena.range()) {
              auto coord = arena.coord(loc);
              auto localIndex = coord & (grid_t::side_length - 1);
              auto blockno = table.query(coord - localIndex);
              if (blockno < 0)
                printf("THE HELL!");
              auto block = grid.block(blockno);

              auto W = arena.weight(loc);
              const auto cellid = grid_t::coord_to_cellid(localIndex);
              for (int d = 0; d != 3; ++d) {
                atomic_add(exec_cuda, &block(vtag, d, cellid),
                           (float)(W * vft[d]));
              }
            }
          };
          auto Dminv = eles.pack<3, 3>("DmInv", pi);
          auto inds = eles.pack<3>("inds", pi).template reinterpret_bits<int>();
          // auto vol0 = verts("vol", ind0);
          auto p0 = verts.pack<3>("x", inds[0]);
          {
            zs::vec<float, 3> Dminv_r[2] = {row(Dminv, 0), row(Dminv, 1)};
            transfer(p0, Dminv_r[0] + Dminv_r[1], vol * dt);
            for (int i = 1; i != 3; ++i) {
              // auto Dinv_ri = row(Dminv, i - 1);
              auto Dinv_ri = Dminv_r[i - 1];
              auto p_i = verts.pack<3>("x", inds[i]);
              transfer(p_i, Dinv_ri, -vol * dt);
              // transfer(p0, Dinv_ri, vol * dt);
            }
          }
        });
  }
  template <typename Model, typename AnisoModel>
  void p2g_apic(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                const AnisoModel &anisoModel,
                const typename ZenoParticles::particles_t &pars,
                const typename ZenoPartition::table_t &partition,
                const float dt, typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    bool materialParamOverride =
        pars.hasProperty("mu") && pars.hasProperty("lam");
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx, model = model,
                                 materialParamOverride,
                                 anisoModel] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      const auto Dinv = 4.f * dxinv * dxinv;
      auto localPos = pars.pack<3>("x", pi);
      auto vel = pars.pack<3>("v", pi);
      auto mass = pars("m", pi);
      auto vol = pars("vol", pi);
      auto C = pars.pack<3, 3>("C", pi);
      auto F = pars.pack<3, 3>("F", pi);
      if (materialParamOverride) {
        model.mu = pars("mu", pi);
        model.lam = pars("lam", pi);
      }
      auto P = model.first_piola(F);
      if constexpr (is_same_v<RM_CVREF_T(anisoModel), AnisotropicArap<float>>)
        P += anisoModel.first_piola(F, pars.pack<3>("a", pi));

      auto contrib = -dt * Dinv * vol * P * F.transpose();
      auto arena = make_local_arena(grid.dx, localPos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0)
          printf("THE HELL!");
        auto block = grid.block(blockno);

        auto xixp = arena.diff(loc);
        auto W = arena.weight(loc);
        const auto cellid = grid_t::coord_to_cellid(localIndex);
        atomic_add(exec_cuda, &block("m", cellid), mass * W);
        auto Cxixp = C * xixp;
        auto fdt = contrib * xixp;
        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &block("v", d, cellid),
                     W * (mass * (vel[d] + Cxixp[d]) + fdt[d]));
      }
    });
  }
  template <typename Model, typename AnisoModel>
  void p2g_flip(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                const AnisoModel &anisoModel,
                const typename ZenoParticles::particles_t &pars,
                const typename ZenoPartition::table_t &partition,
                const float dt, typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx, model,
                                 anisoModel] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      auto localPos = pars.pack<3>("x", pi);
      auto vel = pars.pack<3>("v", pi);
      auto mass = pars("m", pi);
      auto vol = pars("vol", pi);
      auto F = pars.pack<3, 3>("F", pi);
      auto P = model.first_piola(F);
      if constexpr (is_same_v<RM_CVREF_T(anisoModel), AnisotropicArap<float>>)
        P += anisoModel.first_piola(F, pars.pack<3>("a", pi));

      auto contrib = -dt * vol * P * F.transpose();
      auto arena = make_local_arena<grid_e::collocated, kernel_e::quadratic, 1>(
          grid.dx, localPos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0)
          printf("THE HELL!");
        auto block = grid.block(blockno);

        auto W = arena.weight(loc);
        auto massW = W * mass;
        auto Wgrad = arena.weightGradients(loc) * dxinv;
        const auto cellid = grid_t::coord_to_cellid(localIndex);

        atomic_add(exec_cuda, &block("m", cellid), massW);
        auto fdt = contrib * Wgrad;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &block("v", d, cellid), massW * vel[d]);
          atomic_add(exec_cuda, &block("vstar", d, cellid), W * fdt[d]);
        }
      }
    });
  }
  template <typename Model, typename AnisoModel>
  void p2g_aflip(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                 const AnisoModel &anisoModel,
                 const typename ZenoParticles::particles_t &pars,
                 const typename ZenoPartition::table_t &partition,
                 const float dt, typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    bool materialParamOverride =
        pars.hasProperty("mu") && pars.hasProperty("lam");
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx, model = model,
                                 materialParamOverride,
                                 anisoModel] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      const auto Dinv = 4.f * dxinv * dxinv;
      auto localPos = pars.pack<3>("x", pi);
      auto vel = pars.pack<3>("v", pi);
      auto mass = pars("m", pi);
      auto vol = pars("vol", pi);
      auto C = pars.pack<3, 3>("C", pi);
      auto F = pars.pack<3, 3>("F", pi);
      if (materialParamOverride) {
        model.mu = pars("mu", pi);
        model.lam = pars("lam", pi);
      }
      auto P = model.first_piola(F);
      if constexpr (is_same_v<RM_CVREF_T(anisoModel), AnisotropicArap<float>>)
        P += anisoModel.first_piola(F, pars.pack<3>("a", pi));

      auto contrib = -dt * Dinv * vol * P * F.transpose();
      auto arena = make_local_arena(grid.dx, localPos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0)
          printf("THE HELL!");
        auto block = grid.block(blockno);

        auto xixp = arena.diff(loc);
        auto W = arena.weight(loc);
        const auto cellid = grid_t::coord_to_cellid(localIndex);
        atomic_add(exec_cuda, &block("m", cellid), mass * W);
        auto Cxixp = C * xixp;
        auto fdt = contrib * xixp;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &block("v", d, cellid),
                     W * (mass * (vel[d] + Cxixp[d])));
          atomic_add(exec_cuda, &block("vstar", d, cellid), W * fdt[d]);
        }
      }
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSParticleToZSGrid\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto stepDt = get_input<zeno::NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

#if 0
    cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid),
             table = proxy<execspace_e::cuda>(
                 partition)] __device__(auto bi, auto ci) mutable {
              auto block = grid.block(bi);
              auto mass = block("m", ci);
                auto vel = block.pack<3>("v", ci);
                if (mass != 0.f || vel(1) != 0.f) {
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;
                  printf("(%f, %f, %f) mass: %f, vel: %f, %f, %f\n", pos[0],
                         pos[1], pos[2], mass, vel[0], vel[1], vel[2]);
                }
            });
    puts("before p2g grid check");
    getchar();
#endif

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      auto &model = parObjPtr->getModel();

      std::size_t numE = 0;
      if (parObjPtr->isMeshPrimitive())
        numE = parObjPtr->getQuadraturePoints().size();
      else if (parObjPtr->isBendingString())
        numE = parObjPtr->getQuadraturePoints().size();
      fmt::print("[p2g] dx: {}, dt: {}, npars: {}, neles: {}\n", grid.dx,
                 stepDt, pars.size(), numE);

      if (parObjPtr->category == ZenoParticles::mpm)
        match([&](auto &elasticModel, auto &anisoElasticModel) {
          if (zsgrid->transferScheme == "apic")
            p2g_apic(cudaPol, elasticModel, anisoElasticModel, pars, partition,
                     stepDt, grid);
          else if (zsgrid->transferScheme == "flip")
            p2g_flip(cudaPol, elasticModel, anisoElasticModel, pars, partition,
                     stepDt, grid);
          else if (zsgrid->transferScheme == "aflip")
            p2g_aflip(cudaPol, elasticModel, anisoElasticModel, pars, partition,
                      stepDt, grid);
        })(model.getElasticModel(), model.getAnisoElasticModel());
      else if (parObjPtr->isLagrangianParticles()) {
        bool isAffineAugmented = zsgrid->isAffineAugmented();
        bool isFlipStyle = zsgrid->isFlipStyle();
        auto &eles = parObjPtr->getQuadraturePoints();
        p2g_momentum(cudaPol, pars, partition, grid, isAffineAugmented);
        // p2g_momentum(cudaPol, eles, partition, grid, isAffineAugmented);
        match([&](auto &elasticModel) {
          if (parObjPtr->category == ZenoParticles::surface) {
            p2g_surface_force(cudaPol, elasticModel, pars, eles, partition,
                              stepDt, grid, isFlipStyle, isAffineAugmented);
          } else if (parObjPtr->category == ZenoParticles::curve) {
            if constexpr (is_same_v<RM_CVREF_T(elasticModel),
                                    StvkWithHencky<float>>)
              p2g_curve_force(cudaPol, elasticModel, pars, eles, partition,
                              stepDt, grid, isFlipStyle, isAffineAugmented);
            else
              throw std::runtime_error(
                  "curve should use StvkWithHencky elastic constitutive model "
                  "for friction handling!");
          }
        })(model.getElasticModel());
      } else if (parObjPtr->category == ZenoParticles::vert_bending_spring) {
        bool isFlipStyle = zsgrid->isFlipStyle();
        auto &eles = parObjPtr->getQuadraturePoints();
        p2g_vert_bending_force(
            cudaPol, pars, eles, partition, stepDt, grid, model.dx,
            isFlipStyle); // here pars refers to surface vertices
      } else if (parObjPtr->category == ZenoParticles::tri_bending_spring) {
        bool isFlipStyle = zsgrid->isFlipStyle();
        auto &eles = parObjPtr->getQuadraturePoints();
        p2g_element_bending_force(
            cudaPol, pars, eles, partition, stepDt, grid, model.dx,
            isFlipStyle); // here pars refers to surface elements
      } else if (parObjPtr->category == ZenoParticles::bending) {
        bool isFlipStyle = zsgrid->isFlipStyle();
        auto &eles = parObjPtr->getQuadraturePoints();
        p2g_angle_bending_force(cudaPol, pars, eles, partition, stepDt, grid,
                                isFlipStyle);
      } else if (parObjPtr->category != ZenoParticles::tracker) {
        // not implemented yet
      }
    }

    fmt::print(fg(fmt::color::cyan), "done executing ZSParticleToZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ZSParticleToZSGrid,
           {
               {"ZSParticles", "ZSPartition", "ZSGrid", "dt"},
               {"ZSGrid"},
               {},
               {"MPM"},
           });

struct ZSGridToZSParticle : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSGridToZSParticle\n");
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");

    auto stepDt = get_input<NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    using grid_t = RM_CVREF_T(grid);
    if (table->hasTags()) {
      using Ti = typename ZenoPartition::Ti;
      table->clearTags();
      cudaPol(Collapse{table->numBoundaryEntries(), grid_t::block_size},
              [table = proxy<execspace_e::cuda>(partition),
               boundaryIndices =
                   proxy<execspace_e::cuda>(table->getBoundaryIndices()),
               tags = proxy<execspace_e::cuda>(table->getTags()),
               grid = proxy<execspace_e::cuda>(
                   {}, grid)] __device__(auto boundaryNo, auto ci) mutable {
                using grid_t = RM_CVREF_T(grid);
                using table_t = RM_CVREF_T(table);
                using key_t = typename table_t::key_t;
                auto bi = boundaryIndices[boundaryNo];
                if (grid("m", bi, ci) != 0.f && tags[boundaryNo] == 0) {
                  if (atomic_cas(exec_cuda, &tags[boundaryNo], 0, 1) == 0) {
#if 0
                    auto bcoord = table._activeKeys[bi];
                    constexpr auto side_length = grid_t::side_length;
                    int isBoundary =
                        (table.query(bcoord + key_t{-side_length, 0, 0}) ==
                         table_t::sentinel_v)
                            << 0 |
                        (table.query(bcoord + key_t{side_length, 0, 0}) ==
                         table_t::sentinel_v)
                            << 1 |
                        (table.query(bcoord + key_t{0, -side_length, 0}) ==
                         table_t::sentinel_v)
                            << 2 |
                        (table.query(bcoord + key_t{0, side_length, 0}) ==
                         table_t::sentinel_v)
                            << 3 |
                        (table.query(bcoord + key_t{0, 0, -side_length}) ==
                         table_t::sentinel_v)
                            << 4 |
                        (table.query(bcoord + key_t{0, 0, side_length}) ==
                         table_t::sentinel_v)
                            << 5;
                    printf("grid (%d, %d) [bou tag: %d] mass: %f\n", bi, ci,
                           isBoundary, grid("m", bi, ci));
#endif
                  }
                }
              });
      fmt::print("checking {} boundary blocks out of {} blocks in total.",
                 table->numBoundaryEntries(), table->numEntries());
    }

    for (auto &&parObjPtr : parObjPtrs) {
      if (parObjPtr->asBoundary)
        continue;
      fmt::print("g2p iterating par: {}\n", (void *)parObjPtr);
      auto &pars = parObjPtr->getParticles();

      if (parObjPtr->category == ZenoParticles::mpm) {
        if (zsgrid->isPicStyle())
          cudaPol(range(pars.size()),
                  [pars = proxy<execspace_e::cuda>({}, pars),
                   table = proxy<execspace_e::cuda>(partition),
                   grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                   dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                    using grid_t = RM_CVREF_T(grid);
                    const auto Dinv = 4.f * dxinv * dxinv;
                    auto pos = pars.pack<3>("x", pi);
                    auto vel = zs::vec<float, 3>::zeros();
                    auto C = zs::vec<float, 3, 3>::zeros();

                    auto arena = make_local_arena(grid.dx, pos);
                    for (auto loc : arena.range()) {
                      auto coord = arena.coord(loc);
                      auto localIndex = coord & (grid_t::side_length - 1);
                      auto blockno = table.query(coord - localIndex);
                      if (blockno < 0)
                        printf("THE HELL!");
                      auto block = grid.block(blockno);
                      auto xixp = arena.diff(loc);
                      auto W = arena.weight(loc);
                      auto vi = block.pack<3>(
                          "v", grid_t::coord_to_cellid(localIndex));

                      vel += vi * W;
                      C += W * Dinv * dyadic_prod(vi, xixp);
                    }
                    pars.tuple<3>("v", pi) = vel;
                    pars.tuple<3 * 3>("C", pi) = C;
                    pos += vel * dt;
                    pars.tuple<3>("x", pi) = pos;

                    auto F = pars.pack<3, 3>("F", pi);
                    auto tmp = zs::vec<float, 3, 3>::identity() + C * dt;
                    F = tmp * F;
                    pars.tuple<3 * 3>("F", pi) = F;
                  });
        else if (zsgrid->isFlipStyle()) {
          if (zsgrid->isAffineAugmented())
            cudaPol(range(pars.size()),
                    [pars = proxy<execspace_e::cuda>({}, pars),
                     table = proxy<execspace_e::cuda>(partition),
                     grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                     dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                      using grid_t = RM_CVREF_T(grid);
                      const auto Dinv = 4.f * dxinv * dxinv;
                      auto pos = pars.pack<3>("x", pi);
                      auto vel = zs::vec<float, 3>::zeros();
                      auto vstar = zs::vec<float, 3>::zeros();
                      auto C = zs::vec<float, 3, 3>::zeros();

                      auto arena = make_local_arena(grid.dx, pos);
                      for (auto loc : arena.range()) {
                        auto coord = arena.coord(loc);
                        auto localIndex = coord & (grid_t::side_length - 1);
                        auto blockno = table.query(coord - localIndex);
                        if (blockno < 0)
                          printf("THE HELL!");
                        auto block = grid.block(blockno);
                        auto xixp = arena.diff(loc);
                        auto W = arena.weight(loc);
                        auto vi = block.pack<3>(
                            "v", grid_t::coord_to_cellid(localIndex));
                        vel += vi * W;
                        auto vs = block.pack<3>(
                            "vstar", grid_t::coord_to_cellid(localIndex));
                        vstar += vs * W;
                        C += W * Dinv * dyadic_prod(vs, xixp);
                      }
                      constexpr float flip = 0.99f;
                      const auto beta = pars("beta", pi);
                      auto vp0 = pars.pack<3>("v", pi);
                      pars.tuple<3>("v", pi) = vstar + flip * (vp0 - vel);
                      pars.tuple<3 * 3>("C", pi) = C;
                      // pos += vstar * dt;
                      // beta=0->pure aflip
                      pos += dt * (vstar + beta * flip * (vp0 - vel));
                      pars.tuple<3>("x", pi) = pos;

                      auto F = pars.pack<3, 3>("F", pi);
                      auto tmp = zs::vec<float, 3, 3>::identity() + C * dt;
                      F = tmp * F;
                      pars.tuple<3 * 3>("F", pi) = F;
                    });
          else
            cudaPol(range(pars.size()),
                    [pars = proxy<execspace_e::cuda>({}, pars),
                     table = proxy<execspace_e::cuda>(partition),
                     grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                     dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                      using grid_t = RM_CVREF_T(grid);
                      auto pos = pars.pack<3>("x", pi);
                      auto v = zs::vec<float, 3>::zeros();
                      auto vstar = zs::vec<float, 3>::zeros();
                      auto vGrad = zs::vec<float, 3, 3>::zeros();

                      auto arena = make_local_arena<grid_e::collocated,
                                                    kernel_e::quadratic, 1>(
                          grid.dx, pos);
                      for (auto loc : arena.range()) {
                        auto coord = arena.coord(loc);
                        auto localIndex = coord & (grid_t::side_length - 1);
                        auto blockno = table.query(coord - localIndex);
                        if (blockno < 0)
                          printf("THE HELL!");
                        auto block = grid.block(blockno);
                        auto W = arena.weight(loc);
                        auto Wgrad = arena.weightGradients(loc) * dxinv;

                        auto vi = block.pack<3>(
                            "v", grid_t::coord_to_cellid(localIndex));
                        auto vs = block.pack<3>(
                            "vstar", grid_t::coord_to_cellid(localIndex));
                        v += vi * W;
                        vstar += vs * W;
                        vGrad += dyadic_prod(vs, Wgrad);
                      }
                      constexpr float flip = 0.99f;
                      auto vp0 = pars.pack<3>("v", pi);
                      auto vel = vstar + flip * (vp0 - v);
                      pars.tuple<3>("v", pi) = vel;
                      pos += vstar * dt;
                      pars.tuple<3>("x", pi) = pos;

                      auto F = pars.pack<3, 3>("F", pi);
                      auto tmp = zs::vec<float, 3, 3>::identity() + vGrad * dt;
                      F = tmp * F;
                      pars.tuple<3 * 3>("F", pi) = F;
                    });
        }
      } else if (parObjPtr->category == ZenoParticles::tracker) {
        cudaPol(range(pars.size()),
                [pars = proxy<execspace_e::cuda>({}, pars),
                 table = proxy<execspace_e::cuda>(partition),
                 grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                 dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                  using grid_t = RM_CVREF_T(grid);
                  auto pos = pars.pack<3>("x", pi);
                  auto vel = zs::vec<float, 3>::zeros();

                  auto arena = make_local_arena(grid.dx, pos);
                  for (auto loc : arena.range()) {
                    auto coord = arena.coord(loc);
                    auto localIndex = coord & (grid_t::side_length - 1);
                    auto blockno = table.query(coord - localIndex);
                    if (blockno < 0)
                      printf("THE HELL!");
                    auto block = grid.block(blockno);
                    auto W = arena.weight(loc);
                    auto vi =
                        block.pack<3>("v", grid_t::coord_to_cellid(localIndex));

                    vel += vi * W;
                  }
                  // vel
                  pars.tuple<3>("v", pi) = vel;
                  // pos
                  pos += vel * dt;
                  pars.tuple<3>("x", pi) = pos;
                });
      } else if (parObjPtr->isLagrangianParticles()) {
        bool isFlipStyle = zsgrid->isFlipStyle();
        if (isFlipStyle)
          cudaPol(range(pars.size()),
                  [pars = proxy<execspace_e::cuda>({}, pars),
                   table = proxy<execspace_e::cuda>(partition),
                   grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                   dxinv = 1.f / grid.dx] __device__(int pi) mutable {
                    using grid_t = RM_CVREF_T(grid);
                    const auto Dinv = 4.f * dxinv * dxinv;
                    auto pos = pars.pack<3>("x", pi);
                    auto vel = zs::vec<float, 3>::zeros();
                    auto vstar = zs::vec<float, 3>::zeros();
                    auto C = zs::vec<float, 3, 3>::zeros();

                    auto arena = make_local_arena(grid.dx, pos);
                    for (auto loc : arena.range()) {
                      auto coord = arena.coord(loc);
                      auto localIndex = coord & (grid_t::side_length - 1);
                      auto blockno = table.query(coord - localIndex);
                      if (blockno < 0)
                        printf("THE HELL!");
                      auto block = grid.block(blockno);
                      auto xixp = arena.diff(loc);
                      auto W = arena.weight(loc);
                      // auto Wgrad = arena.weightGradients(loc) * dxinv;
                      auto vi = block.pack<3>(
                          "v", grid_t::coord_to_cellid(localIndex));
                      vel += vi * W;
                      auto vs = block.pack<3>(
                          "vstar", grid_t::coord_to_cellid(localIndex));
                      vstar += vs * W;
                      C += W * Dinv * dyadic_prod(vs, xixp);
                    }
                    // vel
                    constexpr float flip = 0.99f;
                    const auto beta = pars("beta", pi);
                    auto vp0 = pars.pack<3>("v", pi);
                    pars.tuple<3>("v", pi) = vstar + flip * (vp0 - vel);
                    // C
                    auto skew = 0.5f * (C - C.transpose());
                    auto sym = 0.5f * (C + C.transpose());
                    C = skew + sym;
                    pars.tuple<9>("C", pi) = C;
                    // pos += vstar * dt;  // (a)flip
                    pos += dt * (vstar + beta * flip * (vp0 - vel));
                    pars.tuple<3>("x", pi) = pos;
                  });
        else if (zsgrid->isPicStyle())
          cudaPol(range(pars.size()),
                  [pars = proxy<execspace_e::cuda>({}, pars),
                   table = proxy<execspace_e::cuda>(partition),
                   grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                   dxinv = 1.f / grid.dx] __device__(int pi) mutable {
                    using grid_t = RM_CVREF_T(grid);
                    const auto Dinv = 4.f * dxinv * dxinv;
                    auto pos = pars.pack<3>("x", pi);
                    auto vel = zs::vec<float, 3>::zeros();
                    auto C = zs::vec<float, 3, 3>::zeros();

                    auto arena = make_local_arena(grid.dx, pos);
                    for (auto loc : arena.range()) {
                      auto coord = arena.coord(loc);
                      auto localIndex = coord & (grid_t::side_length - 1);
                      auto blockno = table.query(coord - localIndex);
                      if (blockno < 0)
                        printf("THE HELL!");
                      auto block = grid.block(blockno);
                      auto xixp = arena.diff(loc);
                      auto W = arena.weight(loc);
                      // auto Wgrad = arena.weightGradients(loc) * dxinv;
                      auto vi = block.pack<3>(
                          "v", grid_t::coord_to_cellid(localIndex));

                      vel += vi * W;
                      C += W * Dinv * dyadic_prod(vi, xixp);
                    }
                    // vel
                    pars.tuple<3>("v", pi) = vel;
                    // C
                    auto skew = 0.5f * (C - C.transpose());
                    auto sym = 0.5f * (C + C.transpose());
                    C = skew + sym;
                    pars.tuple<9>("C", pi) = C;
                    // pos
                    pos += vel * dt;
                    pars.tuple<3>("x", pi) = pos;
                  });
        auto &eles = parObjPtr->getQuadraturePoints();
        const zs::SmallString vtag = isFlipStyle ? "vstar" : "v";
        if (parObjPtr->category == ZenoParticles::surface) {
          cudaPol(
              range(eles.size()),
              [verts = proxy<execspace_e::cuda>({}, pars),
               eles = proxy<execspace_e::cuda>({}, eles),
               table = proxy<execspace_e::cuda>(partition),
               grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
               dxinv = 1.f / grid.dx, vtag] __device__(size_t pi) mutable {
                using mat2 = zs::vec<float, 2, 2>;
                using mat3 = zs::vec<float, 3, 3>;
                using grid_t = RM_CVREF_T(grid);
                const auto Dinv = 4.f * dxinv * dxinv;
                auto pos = eles.pack<3>("x", pi);
                auto C = zs::vec<float, 3, 3>::zeros();

                auto arena = make_local_arena(grid.dx, pos);
                for (auto loc : arena.range()) {
                  auto coord = arena.coord(loc);
                  auto localIndex = coord & (grid_t::side_length - 1);
                  auto blockno = table.query(coord - localIndex);
                  if (blockno < 0)
                    printf("THE HELL!");
                  auto block = grid.block(blockno);
                  auto xixp = arena.diff(loc);
                  auto W = arena.weight(loc);
                  auto vi =
                      block.pack<3>(vtag, grid_t::coord_to_cellid(localIndex));

                  C += W * Dinv * dyadic_prod(vi, xixp);
                }
                auto skew = 0.5f * (C - C.transpose());
                auto sym = 0.5f * (C + C.transpose());
                C = skew + sym;
                eles.tuple<9>("C", pi) = C;

                // section 4.3
                auto is =
                    eles.pack<3>("inds", pi).template reinterpret_bits<int>();

                auto p0 = verts.pack<3>("x", is[0]);
                auto p1 = verts.pack<3>("x", is[1]);
                auto p2 = verts.pack<3>("x", is[2]);
                // pos
                eles.tuple<3>("x", pi) = (p0 + p1 + p2) / 3;
                // vel
                eles.tuple<3>("v", pi) =
                    (verts.pack<3>("v", is[0]) + verts.pack<3>("v", is[1]) +
                     verts.pack<3>("v", is[2])) /
                    3;

                // d
                auto d_c1 = p1 - p0;
                auto d_c2 = p2 - p0;
                auto d_c3 = col(eles.pack<3, 3>("d", pi), 2);
// d_c3 += dt * (vGrad * d_c3);
#if 0
                    d_c3 += dt * (C * d_c3);
#else
                // ref: Yun Fei, libwetcloth;
                // This file is part of the libWetCloth open source project
                //
                // Copyright 2018 Yun (Raymond) Fei, Christopher Batty, Eitan
                // Grinspun, and Changxi Zheng
                //
                // This Source Code Form is subject to the terms of the Mozilla
                // Public License, v. 2.0. If a copy of the MPL was not
                // distributed with this file, You can obtain one at
                // http://mozilla.org/MPL/2.0/.
                d_c3 += (dt * C + 0.5 * dt * dt * C * C) * d_c3;
#endif

                mat3 d{d_c1[0], d_c2[0], d_c3[0], d_c1[1], d_c2[1],
                       d_c3[1], d_c1[2], d_c2[2], d_c3[2]};
                eles.tuple<9>("d", pi) = d;
                // F
                eles.tuple<9>("F", pi) = d * eles.pack<3, 3>("DmInv", pi);
              });
        } // case: surface
        else if (parObjPtr->category == ZenoParticles::curve) {
          cudaPol(
              range(eles.size()),
              [verts = proxy<execspace_e::cuda>({}, pars),
               eles = proxy<execspace_e::cuda>({}, eles),
               table = proxy<execspace_e::cuda>(partition),
               grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
               dxinv = 1.f / grid.dx, vtag] __device__(size_t pi) mutable {
                using mat2 = zs::vec<float, 2, 2>;
                using mat3 = zs::vec<float, 3, 3>;
                using grid_t = RM_CVREF_T(grid);
                const auto Dinv = 4.f * dxinv * dxinv;
                auto pos = eles.pack<3>("x", pi);
                auto C = zs::vec<float, 3, 3>::zeros();

                auto arena = make_local_arena(grid.dx, pos);
                for (auto loc : arena.range()) {
                  auto coord = arena.coord(loc);
                  auto localIndex = coord & (grid_t::side_length - 1);
                  auto blockno = table.query(coord - localIndex);
                  if (blockno < 0)
                    printf("THE HELL!");
                  auto block = grid.block(blockno);
                  auto xixp = arena.diff(loc);
                  auto W = arena.weight(loc);
                  auto vi =
                      block.pack<3>(vtag, grid_t::coord_to_cellid(localIndex));

                  C += W * Dinv * dyadic_prod(vi, xixp);
                }
                auto skew = 0.5f * (C - C.transpose());
                auto sym = 0.5f * (C + C.transpose());
                C = skew + sym;
                eles.tuple<9>("C", pi) = C;

                // section 4.3
                auto is =
                    eles.pack<2>("inds", pi).template reinterpret_bits<int>();

                auto p0 = verts.pack<3>("x", is[0]);
                auto p1 = verts.pack<3>("x", is[1]);
                // pos
                eles.tuple<3>("x", pi) = (p0 + p1) / 2;
                // vel
                eles.tuple<3>("v", pi) =
                    (verts.pack<3>("v", is[0]) + verts.pack<3>("v", is[1])) / 2;

                // d
                auto d_c1 = p1 - p0;
                zs::vec<float, 3, 2> d_c23{};
                {
                  auto d = eles.pack<3, 3>("d", pi);
                  for (int r = 0; r != 3; ++r)
                    for (int c = 0; c != 2; ++c)
                      d_c23(r, c) = d(r, c + 1);
                }
#if 0
                d_c23 += dt * (C * d_c23);
#else
                // ref: Yun Fei, libwetcloth;
                // This file is part of the libWetCloth open source project
                //
                // Copyright 2018 Yun (Raymond) Fei, Christopher Batty, Eitan
                // Grinspun, and Changxi Zheng
                //
                // This Source Code Form is subject to the terms of the Mozilla
                // Public License, v. 2.0. If a copy of the MPL was not
                // distributed with this file, You can obtain one at
                // http://mozilla.org/MPL/2.0/.
                d_c23 += (dt * C + 0.5 * dt * dt * C * C) * d_c23;
#endif

                mat3 d{d_c1[0], d_c23(0, 0), d_c23(0, 1),
                       d_c1[1], d_c23(1, 0), d_c23(1, 1),
                       d_c1[2], d_c23(2, 0), d_c23(2, 1)};
                eles.tuple<9>("d", pi) = d;
                // F
                eles.tuple<9>("F", pi) = d * eles.pack<3, 3>("DmInv", pi);
              });
        } // case: curve
      }   // end mesh particle g2p
    }
    fmt::print(fg(fmt::color::cyan), "done executing ZSGridToZSParticle\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ZSGridToZSParticle,
           {
               {"ZSGrid", "ZSPartition", "ZSParticles", "dt"},
               {"ZSParticles"},
               {},
               {"MPM"},
           });

struct ZSBoundaryPrimitiveToZSGrid : INode {
  void p2g_momentum(zs::CudaExecutionPolicy &cudaPol, std::size_t offset,
                    std::size_t size,
                    const typename ZenoParticles::particles_t &pars,
                    ZenoPartition &zspartition, typename ZenoGrid::grid_t &grid,
                    bool primitiveAsBoundary, bool includeNormal = false) {
    using namespace zs;
    const typename ZenoPartition::table_t &partition = zspartition.get();

    Vector<int> flag{pars.get_allocator(), 1};
    flag.setVal(0);
    cudaPol(range(size), [pars = proxy<execspace_e::cuda>({}, pars),
                          table = proxy<execspace_e::cuda>(partition),
                          grid = proxy<execspace_e::cuda>({}, grid),
                          dxinv = 1.f / grid.dx, offset, includeNormal,
                          flag = proxy<execspace_e::cuda>(
                              flag)] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      const auto Dinv = 4.f * dxinv * dxinv;
      pi += offset;
      auto pos = pars.pack<3>("x", pi);
      auto vel = pars.pack<3>("v", pi);
      auto mass = pars("m", pi);
      auto nrm = pars.pack<3>("nrm", pi);

      auto arena =
          make_local_arena<grid_e::collocated, kernel_e::linear>(grid.dx, pos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0) {
          // printf("THE HELL!");
          if (flag[0] == 0)
            flag[0] = 1;
          continue;
        }
        auto block = grid.block(blockno);
        auto W = arena.weight(loc);
        const auto cellid = grid_t::coord_to_cellid(localIndex);
        atomic_add(exec_cuda, &block("m", cellid), mass * W);
        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &block("v", d, cellid), W * mass * vel[d]);
        if (includeNormal)
          for (int d = 0; d != 3; ++d)
            atomic_add(exec_cuda, &block("nrm", d, cellid), W * mass * nrm[d]);
      }
    });
    if (flag.getVal() != 0) {
      if (!primitiveAsBoundary) {
        zspartition.requestRebuild = true;
        fmt::print(fg(fmt::color::red),
                   "encountering boundary particles breaking CFL\n");
      }
    }
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSBoundaryPrimitiveToZSGrid\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto zspartition = get_input<ZenoPartition>("ZSPartition");
    auto &partition = zspartition->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    if (zsgrid->transferScheme != "boundary")
      throw std::runtime_error("grid is not of boundary type!");

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      auto &eles = parObjPtr->getQuadraturePoints();
      if (!pars.hasProperty("nrm") || !eles.hasProperty("nrm"))
        throw std::runtime_error(
            "boundary primitive does not have normal channel!");
      auto sprayedOffset = parObjPtr->sprayedOffset;
      auto sprayedSize = pars.size() - sprayedOffset;
      auto size = sprayedOffset;
      if (sprayedSize == 0) {
        p2g_momentum(cudaPol, 0, pars.size(), pars, *zspartition, grid,
                     parObjPtr->asBoundary, false);
        p2g_momentum(cudaPol, 0, eles.size(), eles, *zspartition, grid,
                     parObjPtr->asBoundary, true);
      } else {
        p2g_momentum(cudaPol, sprayedOffset, sprayedSize, pars, *zspartition,
                     grid, parObjPtr->asBoundary, true);
      }
      fmt::print("[boundary particle p2g] dx: {}, npars: {}, neles: {}\n",
                 grid.dx, pars.size(), eles.size());
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing ZSBoundaryPrimitiveToZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ZSBoundaryPrimitiveToZSGrid,
           {
               {"ZSParticles", "ZSPartition", "ZSGrid"},
               {"ZSGrid"},
               {},
               {"MPM"},
           });

} // namespace zeno