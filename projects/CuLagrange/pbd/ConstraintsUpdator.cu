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
#include "../../Utils.hpp"

#include "constraint_function_kernel/constraint.cuh"
#include "../geometry/kernel/tiled_vector_ops.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/geo_math.hpp"
#include "../geometry/kernel/bary_centric_weights.hpp"
#include "constraint_function_kernel/constraint_types.hpp"
#include "../fem/collision_energy/evaluate_collision.hpp"


namespace zeno {

struct UpdateConstraintTarget : INode {

virtual void apply() override {
    using namespace zs;
    using namespace PBD_CONSTRAINT;

    using vec2 = zs::vec<float,2>;
    using vec3 = zs::vec<float,3>;
    using vec4 = zs::vec<float,4>;
    using vec9 = zs::vec<float,9>;
    using mat3 = zs::vec<float,3,3>;
    using vec2i = zs::vec<int,2>;
    using vec3i = zs::vec<int,3>;
    using vec4i = zs::vec<int,4>;
    using mat4 = zs::vec<int,4,4>;
    

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = zs::cuda_exec();

    auto source = get_input<ZenoParticles>("source");
    auto constraint = get_input<ZenoParticles>("constraint");

    auto update_weight = get_input2<bool>("update_weight");
    auto new_uniform_weight = get_input2<bool>("new_uniform_weight");

    if(update_weight) {
        auto& cquads = constraint->getQuadraturePoints();
        TILEVEC_OPS::fill(cudaPol,cquads,"w",new_uniform_weight);
    }

    // auto target = get_input<ZenoParticles>("target");
    
    auto type = constraint->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});
    // auto do_frame_interpolation = get_input2<bool>("do_frame_interpolation");

    // if(type == category_c::vertex_pin_to_cell_constraint || type == category_c) {
            std::cout << "update constraint " << type << std::endl;
            auto target = get_input<ZenoParticles>("target");
    // switch(type) {
    //     // case category_c::follow_animation_constraint : break;
    //     // case category_c::dcd_collision_constraint : break;
    //     case category_c::vertex_pin_to_cell_constraint || category_c::volume_pin_constraint : 
            auto ctarget = constraint->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
            if(target->getParticles().size() != ctarget->getParticles().size()) {
                std::cout << "the input update target and contraint target has different number of particles" << std::endl;
                throw std::runtime_error("the input update target and constraint target has different number of particles");
            }
            if(target->getQuadraturePoints().size() != ctarget->getQuadraturePoints().size()) {
                std::cout << "the input update target and constraint target has different number of quadratures" << std::endl;
                throw std::runtime_error("the input update target and constraint target has different number of quadratures");
            }

            const auto& kverts = target->getParticles();
            auto& ckverts = ctarget->getParticles();
            if(!ckverts.hasProperty("px")) {
                ckverts.append_channels(cudaPol,{{"px",3}});
            }
            TILEVEC_OPS::copy(cudaPol,ckverts,"x",ckverts,"px");
            TILEVEC_OPS::copy(cudaPol,kverts,"x",ckverts,"x");
    //         break;
    // }

    // }
    set_output("constraint",constraint);
    set_output("source",source);
}

};

ZENDEFNODE(UpdateConstraintTarget, {{
    {"source"},
    {"target"},
    {"constraint"},
    {"bool","update_weight","0"},
    {"float","new_uniform_weight","1.0"}
},
{{"source"},{"constraint"}},
{ 
    // {"string","groupID",""},
},
{"PBD"}});

};