#pragma once
#include <array>
#include <string>
#include <vector>

#include "zensim/math/Vec.h"
#include "zensim/tpls/partio/Partio.h"

namespace zs {

  template <typename T, std::size_t dim>
  void write_partio(std::string filename, const std::vector<std::array<T, dim>> &data,
                    std::string tag = std::string{"position"}) {
    Partio::ParticlesDataMutable *parts = Partio::create();

    Partio::ParticleAttribute attrib = parts->addAttribute(tag.c_str(), Partio::VECTOR, dim);

    parts->addParticles(data.size());
    for (int idx = 0; idx < (int)data.size(); ++idx) {
      float *val = parts->dataWrite<float>(attrib, idx);
      for (int k = 0; k < dim; k++) val[k] = data[idx][k];
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
  }

  template <typename T, std::size_t dim>
  void write_partio_with_stress(std::string filename, const std::vector<std::array<T, dim>> &data,
                                const std::vector<T> &stressData) {
    Partio::ParticlesDataMutable *parts = Partio::create();

    Partio::ParticleAttribute pattrib = parts->addAttribute("position", Partio::VECTOR, dim);
    Partio::ParticleAttribute sattrib = parts->addAttribute("stress", Partio::FLOAT, 1);

    parts->addParticles(data.size());
    for (int idx = 0; idx < (int)data.size(); ++idx) {
      float *val = parts->dataWrite<float>(pattrib, idx);
      float *stress = parts->dataWrite<float>(sattrib, idx);
      for (int k = 0; k < dim; k++) {
        val[k] = data[idx][k];
      }
      stress[0] = stressData[idx];
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
  }

  template <typename T, std::size_t dim>
  void write_partio_with_grid(std::string filename, const std::vector<std::array<T, dim>> &pos,
                              const std::vector<std::array<T, dim>> &force) {
    Partio::ParticlesDataMutable *parts = Partio::create();

    Partio::ParticleAttribute pattrib = parts->addAttribute("position", Partio::VECTOR, dim);
    Partio::ParticleAttribute fattrib = parts->addAttribute("force", Partio::VECTOR, dim);

    parts->addParticles(pos.size());
    for (int idx = 0; idx < (int)pos.size(); ++idx) {
      float *p = parts->dataWrite<float>(pattrib, idx);
      float *f = parts->dataWrite<float>(fattrib, idx);
      for (int k = 0; k < dim; ++k) {
        p[k] = pos[idx][k];
        f[k] = force[idx][k];
      }
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
  }

}  // namespace zs
