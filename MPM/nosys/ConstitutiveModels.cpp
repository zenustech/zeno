#include "../ZensimModel.h"

#include <zen/zen.h>

namespace zenbase {

struct ModelEquationOfState : zen::INode {
  void apply() override {
    auto model = zen::IObject::make<ZenoConstitutiveModel>();

    zs::EquationOfStateConfig res{};
    res.rho = std::get<float>(get_param("density"));
    res.volume = std::get<float>(get_param("volume"));

    res.bulk = std::get<float>(get_param("bulk_modulus"));
    res.gamma = std::get<float>(get_param("gamma"));
    res.viscosity = std::get<float>(get_param("viscosity"));

    model->get() = res;
    set_output("Model", model);
  }
};

static int defEquationOfStateConfig = zen::defNodeClass<ModelEquationOfState>(
    "EquationOfStateConfig", {/* inputs: */ {},
                              /* outputs: */
                              {{"Model"}},
                              /* params: */
                              {{"float", "density", "1e3"},
                               {"float", "volume", "1"},
                               {"float", "bulk_modulus", "1e4"},
                               {"float", "gamma", "7"},
                               {"float", "viscosity", "0.01"}},
                              /* category: */
                              {"constitutive_model"}});

/// fixed corotated
struct ModelFixedCorotated : zen::INode {
  void apply() override {
    auto model = zen::IObject::make<ZenoConstitutiveModel>();

    zs::FixedCorotatedConfig res{};
    res.rho = std::get<float>(get_param("density"));
    res.volume = std::get<float>(get_param("volume"));

    res.E = std::get<float>(get_param("youngs_modulus"));
    res.nu = std::get<float>(get_param("poisson_ratio"));

    model->get() = res;
    set_output("Model", model);
  }
};

static int defFixedCorotatedConfig = zen::defNodeClass<ModelFixedCorotated>(
    "FixedCorotatedConfig", {/* inputs: */ {},
                             /* outputs: */
                             {{"Model"}},
                             /* params: */
                             {{"float", "density", "1e3"},
                              {"float", "volume", "1"},
                              {"float", "youngs_modulus", "1e4"},
                              {"float", "poisson_ratio", "0.4"}},
                             /* category: */
                             {"constitutive_model"}});

} // namespace zenbase