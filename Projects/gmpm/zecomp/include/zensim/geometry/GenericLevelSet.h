
#pragma once
#include "AnalyticLevelSet.h"
#include "LevelSet.h"
#include "LevelSetInterface.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <typename T, int dim> using GenericLevelSet
      = variant<AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>,
                AnalyticLevelSet<analytic_geometry_e::Sphere, T, dim>, LevelSet<T, dim>>;

}
