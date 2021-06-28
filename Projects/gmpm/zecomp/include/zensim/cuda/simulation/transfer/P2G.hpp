#pragma once
#include "zensim/cuda/DeviceUtils.cuh"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/physics/ConstitutiveModel.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/simulation/transfer/P2G.hpp"

namespace zs {

  template <transfer_scheme_e scheme, typename ModelT, typename ParticlesT, typename TableT,
            typename GridBlocksT>
  struct P2GTransfer<scheme, ModelT, ParticlesProxy<execspace_e::cuda, ParticlesT>,
      HashTableProxy<execspace_e::cuda, TableT>, GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using model_t = ModelT;  ///< constitutive model
    using particles_t = ParticlesProxy<execspace_e::cuda, ParticlesT>;
    using partition_t = HashTableProxy<execspace_e::cuda, TableT>;
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == gridblocks_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit P2GTransfer(wrapv<execspace_e::cuda>, wrapv<scheme>, float dt, const ModelT& model,
                         ParticlesT& particles, TableT& table, GridBlocksT& gridblocks)
        : model{model},
          particles{proxy<execspace_e::cuda>(particles)},
          partition{proxy<execspace_e::cuda>(table)},
          gridblocks{proxy<execspace_e::cuda>(gridblocks)},
          dt{dt} {}

    constexpr float dxinv() const {
      return static_cast<decltype(gridblocks._dx.asFloat())>(1.0) / gridblocks._dx.asFloat();
    }

    __forceinline__ __device__ void operator()(typename particles_t::size_type parid) noexcept {
      float const dx = gridblocks._dx.asFloat();
      float const dx_inv = dxinv();
      if constexpr (particles_t::dim == 3) {
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, particles_t::dim>;
        using vec3 = vec<float, particles_t::dim>;
        using vec9 = vec<float, particles_t::dim * particles_t::dim>;
        using vec3x3 = vec<float, particles_t::dim, particles_t::dim>;

        vec3 local_pos{particles.pos(parid)};
        vec3 vel{particles.vel(parid)};
        float mass = particles.mass(parid);
        vec9 contrib{}, C{particles.C(parid)};

        if constexpr (is_same_v<model_t, EquationOfStateConfig>) {
          float J = particles.J(parid);
          float vol = model.volume * J;
          float pressure = model.bulk;
          {
            float J2 = J * J;
            float J4 = J2 * J2;
            // pressure = pressure * (powf(J, -model.gamma) - 1.f);
            pressure = pressure * (1 / (J * J2 * J4) - 1);  // from Bow
          }
          contrib[0] = ((C[0] + C[0]) * model.viscosity - pressure) * vol;
          contrib[1] = (C[1] + C[3]) * model.viscosity * vol;
          contrib[2] = (C[2] + C[6]) * model.viscosity * vol;

          contrib[3] = (C[3] + C[1]) * model.viscosity * vol;
          contrib[4] = ((C[4] + C[4]) * model.viscosity - pressure) * vol;
          contrib[5] = (C[5] + C[7]) * model.viscosity * vol;

          contrib[6] = (C[6] + C[2]) * model.viscosity * vol;
          contrib[7] = (C[7] + C[5]) * model.viscosity * vol;
          contrib[8] = ((C[8] + C[8]) * model.viscosity - pressure) * vol;

        } else {
          const auto [mu, lambda] = lame_parameters(model.E, model.nu);
          vec9 F{particles.F(parid)};
          if constexpr (is_same_v<model_t, FixedCorotatedConfig>) {
            compute_stress_fixedcorotated(model.volume, mu, lambda, F, contrib);
          } else if constexpr (is_same_v<model_t, VonMisesFixedCorotatedConfig>) {
            compute_stress_vonmisesfixedcorotated(model.volume, mu, lambda, model.yieldStress, F,
                                                  contrib);
          } else {
            /// with plasticity additionally
            float logJp = particles.logJp(parid);
            if constexpr (is_same_v<model_t, DruckerPragerConfig>) {
              compute_stress_sand(model.volume, mu, lambda, model.cohesion, model.beta,
                                  model.yieldSurface, model.volumeCorrection, logJp, F, contrib);
            } else if constexpr (is_same_v<model_t, NACCConfig>) {
              compute_stress_nacc(model.volume, mu, lambda, model.bulk(), model.xi, model.beta,
                                  model.Msqr(), model.hardeningOn, logJp, F, contrib);
            }
            particles.logJp(parid) = logJp;
          }
        }

        contrib = C * mass * D_inv - contrib * dt * D_inv;

        using VT
            = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
        auto arena = make_local_arena((VT)dx, local_pos);
        for (auto loc : arena.range()) {
          auto [grid_block, local_index] = unpack_coord_in_grid(
              arena.coord(loc), gridblock_t::side_length(), partition, gridblocks);
          auto xixp = arena.diff(loc);
          VT W = arena.weight(loc);
          VT wm = mass * W;
          atomicAdd(&grid_block(0, local_index).asFloat(), wm);
          for (int d = 0; d < particles_t::dim; ++d)
            atomicAdd(
                &grid_block(d + 1, local_index).asFloat(),
                (VT)(wm * vel[d]
                     + (contrib[d] * xixp[0] + contrib[3 + d] * xixp[1] + contrib[6 + d] * xixp[2])
                           * W));
        }
      }
    }

    model_t model;
    particles_t particles;
    partition_t partition;
    gridblocks_t gridblocks;
    float dt;
  };

}  // namespace zs