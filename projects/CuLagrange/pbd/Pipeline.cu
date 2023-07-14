#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct AddGenericConstraints : zeno::INode {
    using namespace zs;
    virtual void apply() override {
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;

        constexpr auto space = execspace_e::cuda;

        auto zsparticles = RETRIEVE_OBJECT_PTRS(ZenoParticles, "zsparticles");
        auto targets = get_input<ZenoParticles>("target");
        auto constraints = RETRIEVE_OBJECT_PTRS(ZenoParticles, "constraints");

        set_output("zsparticles",zsparticles);
        set_output("constraints",constraints);
    };
};

ZENDEFNODE(AddGenericConstraints, {{{"zsparticles"},{"constraints"},{"target"}},
							{{"zsparticles"},{"constraints"}},
							{},
							{"PBD"}});

struct UpdateConstraintTarget : zeno::INode {
    using namespace zs;
    virtual void apply() override {
        auto constraints = get_input<ZenoParticles>("constraints");
        auto targets = get_input<ZenoParticles>("target");

        set_output("constraints",constraints);
    };
};

ZENDEFNODE(UpdateConstraintTarget, {{{"constraints"},{"target"}},
							{{"constraints"}},
							{},
							{"PBD"}});

struct ZSMerge : zeno::INode {
    using namespace zs;
    virtual void apply() override {
        constexpr auto space = execspace_e::cuda;
        auto zsparticles0 = RETRIEVE_OBJECT_PTRS(ZenoParticles, "zsparticles0");
        auto zsparticles1 = RETRIEVE_OBJECT_PTRS(ZenoParticles, "zsparticles1");

        set_output("zsparticles",zsparticles0);
    };
};

ZENDEFNODE(ZSMerge, {{{"zsparticles0"},{"zsparticles1"}},
							{{"zsparticles"}},
							{},
							{"PBD"}});


struct XPBDSolve : zeno::INode {
    using namespace zs;
    virtual void apply() override {
        constexpr auto space = execspace_e::cuda;
        auto zsparticles = RETRIEVE_OBJECT_PTRS(ZenoParticles, "zsparticles");
        auto constraints = RETRIEVE_OBJECT_PTRS(ZenoParticles, "constraints");        

        set_output("constraints",zsconstraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},{"constraints"}},
							{{"zsparticles"},{"constraints"}},
							{},
							{"PBD"}});

};
