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

#include "../geometry/kernel/tiled_vector_ops.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/geo_math.hpp"

namespace zeno {


struct SDFColliderProject : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("zsparticles");

        auto radius = get_input2<float>("radius");
        auto center = get_input2<zeno::vec3f>("center");
        auto cv = get_input2<zeno::vec3f>("center_velocity");
        auto w = get_input2<zeno::vec3f>("angular_velocity");

        // prex
        auto xtag = get_input2<std::string>("xtag");
        // x
        auto ptag = get_input2<std::string>("ptag");
        auto friction = get_input2<T>("friction");

        // auto do_stablize = get_input2<bool>("do_stablize");

        auto& verts = zsparticles->getParticles();

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            ptag = zs::SmallString(ptag),
            friction,
            radius,
            center,
            // do_stablize,
            cv,w] ZS_LAMBDA(int vi) mutable {
                if(verts("minv",vi) < (T)1e-6)
                    return;

                auto pred = verts.pack(dim_c<3>,ptag,vi);
                auto pos = verts.pack(dim_c<3>,xtag,vi);

                auto center_vel = vec3::from_array(cv);
                auto center_pos = vec3::from_array(center);
                auto angular_velocity = vec3::from_array(w);

                auto disp = pred - center_pos;
                auto dist = radius - disp.norm() + verts("pscale",vi);

                if(dist < 0)
                    return;

                auto nrm = disp.normalized();

                auto dp = dist * nrm;
                if(dp.norm() < (T)1e-6)
                    return;

                pred += dp;

                // if(do_stablize) {
                //     pos += dp;
                //     verts.tuple(dim_c<3>,xtag,vi) = pos; 
                // }

                auto collider_velocity_at_p = center_vel + angular_velocity.cross(pred - center_pos);
                auto rel_vel = pred - pos - center_vel;

                auto tan_vel = rel_vel - nrm * rel_vel.dot(nrm);
                auto tan_len = tan_vel.norm();
                auto max_tan_len = dp.norm() * friction;

                if(tan_len > (T)1e-6) {
                    auto alpha = (T)max_tan_len / (T)tan_len;
                    dp = -tan_vel * zs::min(alpha,(T)1.0);
                    pred += dp;
                }

                // dp = dp * verts("m",vi) * verts("minv",vi);

                verts.tuple(dim_c<3>,ptag,vi) = pred;    
        });
        set_output("zsparticles",zsparticles);
    }

};

ZENDEFNODE(SDFColliderProject, {{{"zsparticles"},
                                {"float","radius","1"},
                                {"center"},
                                {"center_velocity"},
                                {"angular_velocity"},
                                {"string","xtag","x"},
                                {"string","ptag","x"},
                                {"float","friction","0"}
                                // {"bool","do_stablize","0"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});

};