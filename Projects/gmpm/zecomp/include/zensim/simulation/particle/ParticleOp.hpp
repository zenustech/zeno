#pragma once
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  template <typename ParticlesT> struct SetParticleAttribute;

  template <execspace_e space, typename ParticlesT> SetParticleAttribute(wrapv<space>, ParticlesT)
      -> SetParticleAttribute<ParticlesProxy<space, ParticlesT>>;

  template <execspace_e space, typename ParticlesT>
  struct SetParticleAttribute<ParticlesProxy<space, ParticlesT>> {
    using particles_t = ParticlesProxy<space, ParticlesT>;

    explicit SetParticleAttribute(wrapv<space>, ParticlesT& particles)
        : particles{proxy<space>(particles)} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      for (const auto& [i, j] : ndrange<2>(particles_t::dim))
        if (particles.C(parid)[i * particles_t::dim + j] != 0)
          printf("parid %d, C(%d, %d): %e\n", (int)parid, i, j,
                 particles.C(parid)[i * particles_t::dim + j]);
    }

    particles_t particles;
  };

  template <typename ParticlesT> struct AppendParticles;

  template <execspace_e space, typename ParticlesT, typename... Args>
  AppendParticles(wrapv<space>, ParticlesT, ParticlesT, Args... args)
      -> AppendParticles<ParticlesProxy<space, ParticlesT>>;
  template <execspace_e space, typename ParticlesT>
  struct AppendParticles<ParticlesProxy<space, ParticlesT>> {
    using particles_t = ParticlesProxy<space, ParticlesT>;

    template <
        typename... Args,
        enable_if_t<((
            is_same_v<
                Args,
                ParticleAttributeFlagBit> || is_same_v<Args, ParticleAttributeFlagBits>)&&...)> = 0>
    explicit AppendParticles(wrapv<space>, ParticlesT& dst, ParticlesT& incoming, Args... args)
        : dst{proxy<space>(dst)}, src{proxy<space>(incoming)}, options{(args | ...)} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      constexpr auto dim = particles_t::dim;
      /// M
      dst.mass(dst.size() - src.size() + parid) = src.mass(parid);
      /// X, V
      for (int d = 0; d < dim; ++d) {
        dst.pos(dst.size() - src.size() + parid)[d] = src.pos(parid)[d];
        dst.vel(dst.size() - src.size() + parid)[d] = src.vel(parid)[d];
      }
      /// J
      if (options & ParticleAttributeFlagBit::Particle_J)
        dst.J(dst.size() - src.size() + parid) = src.J(parid);
      /// F
      if (options & ParticleAttributeFlagBit::Particle_F)
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j)
            dst.F(dst.size() - src.size() + parid)[i][j] = src.F(parid)[i][j];
      /// C
      if (options & ParticleAttributeFlagBit::Particle_C)
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j)
            dst.C(dst.size() - src.size() + parid)[i][j] = src.C(parid)[i][j];
    }

    particles_t dst, src;
    ParticleAttributeFlagBits options{0};
  };

}  // namespace zs