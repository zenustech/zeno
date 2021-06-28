#pragma once
#include <tuple>

#include "zensim/math/Vec.h"
#include "zensim/tpls/fmt/core.h"
#include "zensim/tpls/gcem/gcem.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum struct constitutive_model_e : char {
    EquationOfState = 0,
    NeoHookean,
    FixedCorotated,
    VonMisesFixedCorotated,
    DruckerPrager,
    NACC,
    NumConstitutiveModels
  };

  template <typename T> constexpr std::tuple<T, T> lame_parameters(T E, T nu) {
    T mu = 0.5 * E / (1 + nu);
    T lam = E * nu / ((1 + nu) * (1 - 2 * nu));
    return std::make_tuple(mu, lam);
  }

  struct MaterialConfig {
    float rho{1e3};
    float volume{1};
    int dim{3};
  };
  struct EquationOfStateConfig : MaterialConfig {
    float bulk{4e4f};
    float gamma{7.15f};  ///< set to 7 by force
    float viscosity{0.f};
  };
  struct NeoHookeanConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
  };
  struct FixedCorotatedConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
  };
  struct VonMisesFixedCorotatedConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float yieldStress{240e6};
  };
  struct DruckerPragerConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float logJp0{0.f};
    float fa{30.f};  ///< friction angle
    float cohesion{0.f};
    float beta{1.f};
    bool volumeCorrection{true};
    float yieldSurface{0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f)};
  };
  struct NACCConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float logJp0{-0.01f};  ///< alpha
    float fa{45.f};
    float xi{0.8f};  ///< hardening factor
    float beta{0.5f};
    bool hardeningOn{true};
    constexpr float bulk() const noexcept {
      return 2.f / 3.f * (E / (2 * (1 + nu))) + (E * nu / ((1 + nu) * (1 - 2 * nu)));
    }
    constexpr float mohrColumbFriction() const noexcept {
      // 0.503599787772409
      float sin_phi = gcem::sin(fa);
      return gcem::sqrt(2.f / 3.f) * 2.f * sin_phi / (3.f - sin_phi);
    }
    constexpr float M() const noexcept {
      // 1.850343771924453
      return mohrColumbFriction() * dim / gcem::sqrt(2.f / (6.f - dim));
    }
    constexpr float Msqr() const noexcept {
      // 3.423772074299613
      auto ret = M();
      return ret * ret;
    }
  };

  using ConstitutiveModelConfig
      = variant<EquationOfStateConfig, NeoHookeanConfig, FixedCorotatedConfig,
                VonMisesFixedCorotatedConfig, DruckerPragerConfig, NACCConfig>;

  constexpr bool particleHasF(const ConstitutiveModelConfig &model) noexcept {
    return model.index() != 0;
  }
  constexpr bool particleHasJ(const ConstitutiveModelConfig &model) noexcept {
    return model.index() == 0;
  }

  inline void displayConfig(ConstitutiveModelConfig &config) {
    match(
        [](EquationOfStateConfig &config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("bulk {}, gamma {}, viscosity{}\n", config.bulk, config.gamma,
                     config.viscosity);
        },
        [](NeoHookeanConfig &config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}\n", config.E, config.nu);
        },
        [](FixedCorotatedConfig &config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}\n", config.E, config.nu);
        },
        [](VonMisesFixedCorotatedConfig &config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}, yieldStress {}\n", config.E, config.nu, config.yieldStress);
        },
        [](DruckerPragerConfig &config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}, logJp0 {}, fric_angle {}, cohesion {}, beta, yieldSurface {}\n",
                     config.E, config.nu, config.logJp0, config.fa, config.cohesion, config.beta,
                     config.yieldSurface);
        },
        [](NACCConfig &config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}, logJp0 {}, fric_angle {}, xi {}, beta {}, mohrColumbFric {}\n",
                     config.E, config.nu, config.logJp0, config.fa, config.xi, config.beta,
                     config.mohrColumbFriction());
        })(config);
  }

  /// temporary
  template <typename T> constexpr vec<T, 3> bspline_weight(T p, T const dx_inv) noexcept {
    vec<T, 3> dw{};
    T d = p * dx_inv - (lower_trunc(p * dx_inv + 0.5) - 1);
    dw[0] = 0.5f * (1.5 - d) * (1.5 - d);
    d -= 1.0f;
    dw[1] = 0.75 - d * d;
    d = 0.5f + d;
    dw[2] = 0.5 * d * d;
    return dw;
  }
  template <typename T, auto dim>
  constexpr vec<T, dim, 3> bspline_weight(const vec<T, dim> &p, T const dx_inv) noexcept {
    vec<T, dim, 3> dw{};
    for (int i = 0; i < dim; ++i) {
      T d = p(i) * dx_inv - (lower_trunc(p(i) * dx_inv + 0.5) - 1);
      dw(i, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dw(i, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dw(i, 2) = 0.5 * d * d;
    }
    return dw;
  }

}  // namespace zs