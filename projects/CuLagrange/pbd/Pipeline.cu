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

// we only need to record the topo here
struct MakeGenericConstraintTopology : zeno::INode {
    using namespace zs;
    virtual void apply() override {
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;

        constexpr auto space = execspace_e::cuda;

        auto source = get_input<ZenoParticles>("source");
        auto constraint = std::make_shared<ZenoParticles>();

        auto type = get_param<std::string>("type");
        auto groupID = get_param<std::string>("groupID");

        // set_output("target",target);
        set_output("source",source);
        set_output("constraint",constraint);
    };
};

ZENDEFNODE(MakeGenericConstraintTopology, {{{"source"}},
							{{"constraints"}},
							{
                                {"enum points edges tris tets segment_angle dihedral_angle","topo_type","points"},
                                {"string","groupID",""},
                            },
							{"PBD"}});

struct XPBDSolve : zeno::INode {
    using namespace zs;
    virtual void apply() override {
        constexpr auto space = execspace_e::cuda;
        auto zsparticles = get_input("zsparticles");
        auto constraints = get_input("constraints");        

        set_output("constraints",constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},{"constraints"}},
							{{"zsparticles"},{"constraints"}},
							{},
							{"PBD"}});

};
