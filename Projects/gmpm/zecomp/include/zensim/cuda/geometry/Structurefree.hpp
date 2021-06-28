#pragma once
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <typename T, int d> bool convertParticles(Particles<T, d> &particles) {
    if (particles.space() == memsrc_e::device || particles.space() == memsrc_e::um
        || particles.space() == memsrc_e::device_const) {
      constexpr auto dim = d;
      std::vector<PropertyTag> properties{{"mass", 1}, {"pos", dim}, {"vel", dim}};
      if (particles.hasC()) properties.push_back({"C", dim * dim});
      if (particles.hasF()) properties.push_back({"F", dim * dim});
      if (particles.hasJ()) properties.push_back({"J", 1});
      if (particles.haslogJp()) properties.push_back({"logjp", 1});

      particles.particleBins
          = TileVector<f32, 32>{properties, particles.size(), particles.space(), particles.devid()};
      std::vector<SmallString> attribNames(properties.size());
      for (auto &&[dst, src] : zip(attribNames, properties)) dst = src.template get<0>();
      auto cuPol = cuda_exec().device(particles.devid());
      cuPol(
          {particles.size()},
          [dim, parray = proxy<execspace_e::cuda>(particles),
           ptiles = proxy<execspace_e::cuda>(
               attribNames,
               particles.particleBins)] __device__(typename Particles<T, d>::size_type i) mutable {
#if 1
            if (i == 0) {
              printf("num total channels %d\n", (int)ptiles.numChannels());
              printf("mass channel %d offset: %d (%d)\n", (int)ptiles.propIndex("mass"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("mass")], ptiles.hasProp("mass"));
              printf("pos channel %d offset: %d (%d)\n", (int)ptiles.propIndex("pos"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("pos")], ptiles.hasProp("pos"));
              printf("vel channel %d offset: %d (%d)\n", (int)ptiles.propIndex("vel"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("vel")], ptiles.hasProp("vel"));
              printf("F channel %d offset: %d (%d)\n", (int)ptiles.propIndex("F"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("F")], ptiles.hasProp("F"));
              printf("C channel %d offset: %d (%d)\n", (int)ptiles.propIndex("C"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("C")], ptiles.hasProp("C"));
              printf("J channel %d offset: %d (%d)\n", (int)ptiles.propIndex("J"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("J")], ptiles.hasProp("J"));
              printf("logJp channel %d offset: %d (%d)\n", (int)ptiles.propIndex("logjp"),
                     (int)ptiles._tagOffsets[ptiles.propIndex("logjp")], ptiles.hasProp("logjp"));
            }
#endif
            ptiles.template tuple<dim>("pos",
                                       i);  // = vec<float, 3>{0.f, 1.f, 2.f};  // parray.pos(i);
            ptiles.val("mass", i) = parray.mass(i);
            ptiles.template tuple<dim>("vel", i) = parray.vel(i);
            if (ptiles.hasProp("C")) ptiles.template tuple<dim * dim>("C", i) = parray.C(i);
            if (ptiles.hasProp("F")) ptiles.template tuple<dim * dim>("F", i) = parray.F(i);
            if (ptiles.hasProp("J")) ptiles.val("J", i) = parray.J(i);
            if (ptiles.hasProp("logjp")) ptiles.val("logjp", i) = parray.logJp(i);
          });
      return true;
    }
    return false;
  }

}  // namespace zs