#pragma once


#include <zen/zen.h>
#include <glm/vec3.hpp>
#include <vector>
#include <string>
#include <map>
#include <variant>

//this is a design of a unified particle object, the goal is such that we can provide
//unified way for the particle data transfering between simulation and rendering
namespace zenbase {

struct UnifiedParticleObject : zen::IObject {
    using ParticleQuantityArray = std::variant<std::vector<glm::vec3>, std::vector<float>>;

    std::map<std::string, ParticleQuantityArray> Particles;
    size_t size(std::string &name) const {
    return ParticleQuantities[name].size();
  }
};

}
