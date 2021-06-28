#pragma once
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/LevelSet.h"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/math/Vec.h"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/tpls/magic_enum/magic_enum.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

#if 0
  struct BoundaryBuilder;
  /// scene setup
  struct Boundary {
    enum struct boundary_e : char { None = 0, HeightField, LevelSet, LevelSetSequence };
    using BoundaryHandle = std::pair<boundary_e, int>;
    std::vector<GeneralParticles> heightFields;
    std::vector<GeneralMesh> levelsets;
    std::vector<GeneralNodes> levelsetSequences;
    static BoundaryBuilder create();
  };

  struct BuilderForBoundaryHeightField;
  struct BuilderForBoundaryLevelset;
  struct BuilderForBoundaryLevelsetSequence;
  struct BuilderForBoundary : BuilderFor<Boundary> {
    explicit BuilderForBoundary(Boundary &boundary) : BuilderFor<Boundary>{boundary} {}
    BuilderForBoundaryHeightField heightfield();
    BuilderForBoundaryLevelset levelset();
    BuilderForBoundaryLevelsetSequence sequence();

    collider_e _boundaryType{collider_e::Sticky};
  };

  struct BoundaryBuilder : BuilderForBoundary {
    BoundaryBuilder() : BuilderForBoundary{_boundary} {}

  protected:
    Boundary _boundary;
  };

  struct BuilderForBoundaryHeightField : BuilderForBoundary {
    explicit BuilderForBoundaryHeightField(Boundary &boundary) : BuilderForBoundary{boundary} {}

    /// push to scene
    BuilderForBoundaryHeightField &commit(MemoryHandle dst);
    /// check build status
    BuilderForBoundaryHeightField &output(std::string fn);

  protected:
    // std::vector<> heightFields;
  };
  struct BuilderForBoundaryLevelset : BuilderForBoundary {
    explicit BuilderForBoundaryLevelset(Boundary &boundary) : BuilderForBoundary{boundary} {}

    /// push to scene
    BuilderForBoundaryLevelset &commit(MemoryHandle dst);
    /// check build status
    BuilderForBoundaryLevelset &output(std::string fn);

  protected:
    // std::vector<> levelsets;
  };
  struct BuilderForBoundaryLevelsetSequence : BuilderForBoundary {
    explicit BuilderForBoundaryLevelsetSequence(Boundary &boundary)
        : BuilderForBoundary{boundary} {}

    /// push to scene
    BuilderForBoundaryLevelsetSequence &commit(MemoryHandle dst);
    /// check build status
    BuilderForBoundaryLevelsetSequence &output(std::string fn);

  protected:
    // std::vector<> sequences;
  };
#endif

}  // namespace zs