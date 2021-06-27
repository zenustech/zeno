#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/math/matrix/MatrixUtils.h"
#include "zensim/physics/ConstitutiveModel.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename GridBlocksT, typename TableT,
            typename ParticlesT>
  struct G2PTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename GridBlocksT,
            typename TableT, typename ParticlesT>
  G2PTransfer(wrapv<space>, wrapv<scheme>, float, Model, GridBlocksT, TableT, ParticlesT)
      -> G2PTransfer<scheme, Model, GridBlocksProxy<space, GridBlocksT>,
                     HashTableProxy<space, TableT>, ParticlesProxy<space, ParticlesT>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename GridBlocksT,
            typename TableT, typename ParticlesT>
  struct G2PTransfer<scheme, ModelT, GridBlocksProxy<space, GridBlocksT>,
                     HashTableProxy<space, TableT>, ParticlesProxy<space, ParticlesT>> {
    using model_t = ModelT;  ///< constitutive model
    using gridblocks_t = GridBlocksProxy<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;
    using partition_t = HashTableProxy<space, TableT>;
    using particles_t = ParticlesProxy<space, ParticlesT>;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == gridblocks_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit G2PTransfer(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model,
                         GridBlocksT& gridblocks, TableT& table, ParticlesT& particles)
        : model{model},
          gridblocks{proxy<space>(gridblocks)},
          partition{proxy<space>(table)},
          particles{proxy<space>(particles)},
          dt{dt} {}

    constexpr float dxinv() const {
      return static_cast<decltype(gridblocks._dx.asFloat())>(1.0) / gridblocks._dx.asFloat();
    }

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      float const dx = gridblocks._dx.asFloat();
      float const dx_inv = dxinv();
      if constexpr (particles_t::dim == 3) {
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, particles_t::dim>;
        using vec3 = vec<float, particles_t::dim>;
        using vec9 = vec<float, particles_t::dim * particles_t::dim>;
        using vec3x3 = vec<float, particles_t::dim, particles_t::dim>;
        vec3 pos{particles.pos(parid)};
        vec3 vel{vec3::zeros()};

        vec9 C{vec9::zeros()};
        auto arena = make_local_arena(dx, pos);
        for (auto loc : arena.range()) {
          auto [grid_block, local_index] = unpack_coord_in_grid(
              arena.coord(loc), gridblock_t::side_length(), partition, gridblocks);
          auto xixp = arena.diff(loc);
          float W = arena.weight(loc);

          vec3 vi{grid_block(1, local_index).asFloat(), grid_block(2, local_index).asFloat(),
                  grid_block(3, local_index).asFloat()};
          vel += vi * W;
          for (int d = 0; d < 9; ++d) C[d] += W * vi(d % 3) * xixp(d / 3);
        }
        pos += vel * dt;

        if constexpr (is_same_v<model_t, EquationOfStateConfig>) {
          float J = particles.J(parid);
          J = (1 + (C[0] + C[4] + C[8]) * dt * D_inv) * J;
          // if (J < 0.1) J = 0.1;
          particles.J(parid) = J;
        } else {
          vec9 oldF{particles.F(parid)}, tmp{}, F{};
          for (int d = 0; d < 9; ++d) tmp(d) = C[d] * dt * D_inv + ((d & 0x3) ? 0.f : 1.f);
          matrixMatrixMultiplication3d(tmp.data(), oldF.data(), F.data());
          particles.F(parid) = F;
        }
        particles.pos(parid) = pos;
        particles.vel(parid) = vel;
        particles.C(parid) = C;
      }
    }

    model_t model;
    gridblocks_t gridblocks;
    partition_t partition;
    particles_t particles;
    float dt;
  };

}  // namespace zs