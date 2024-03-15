#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/math/Rotation.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include <iostream>


namespace zeno {

struct ZSParticlesTransform : zeno::INode {
    typedef float T;
    typedef zs::vec<T,3> vec3;
    typedef zs::vec<T,4> vec4;

    virtual void apply() override {
        using namespace zs;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();


        auto translate = vec3::from_array(get_input2<zeno::vec3f>("translation"));
        auto quat = vec4::from_array(get_input2<zeno::vec4f>("quatRotation"));
        auto scale = vec3::from_array(get_input2<zeno::vec3f>("scaling"));

        auto zsparticles = get_input2<ZenoParticles>("zsparticles");
        auto srcTag = get_input2<std::string>("srcTag");
        auto dstTag = get_input2<std::string>("dstTag");

        auto& verts = zsparticles->getParticles();

        if(!verts.hasProperty(dstTag))
            verts.append_channels(cudaPol,{{dstTag,3}});
            

        cudaPol(zs::range(verts.size()),[
            verts = proxy<cuda_space>({},verts),
            srcOffset = verts.getPropertyOffset(srcTag),
            dstOffset = verts.getPropertyOffset(dstTag),
            translate = translate,
            quat = quat,
            scale = scale] ZS_LAMBDA(int vi) mutable {
                auto pos = verts.pack(dim_c<3>,srcOffset,vi);
                auto rot = Rotation<T,3>::quaternion2matrix(quat);
                pos = rot * pos;
                for(int i = 0;i != 3;++i)
                    pos[i] = pos[i] * scale[i];

                pos = pos + translate;

                verts.tuple(dim_c<3>,dstOffset,vi) = pos;
        });

        set_output("zsparticles",zsparticles);
    }
};


ZENDEFNODE(ZSParticlesTransform, {{{"zsparticles"},
                                {"vec3f","translation","0,0,0"},
                                {"vec4f","quatRotation","0,0,0,1"},
                                {"vec3f","scaling","1,1,1"},
                                {"string","srcTag","x"},
                                {"string","dstTag","x"}
                            },
                            {{"zsparticles"}},
                            {
                            },
                            {"ZSGeometry"}});


struct SlerpQuaternion : zeno::INode {
    typedef float T;
    typedef zs::vec<T,4> vec4;

    virtual void apply() override {
        auto quatA = vec4::from_array(get_input2<zeno::vec4f>("quatA"));
        auto quatB = vec4::from_array(get_input2<zeno::vec4f>("quatB"));
        auto t = 1 - get_input2<float>("wA");

        auto cosa = quatA.dot(quatB);
        if(cosa < 0) {
            quatB = -quatB;
            cosa = -cosa;
        }

        T k0,k1;
        if(cosa > 0.9995f) {
            k0 = 1.0 - t;
            k1 = t;
        } else {
            auto sina = zs::sqrt(1.0 - cosa * cosa);
            auto a = std::atan2(sina,cosa);
            k0 = sin((1-t) * a) / sina;
            k1 = sin(t * a) / sina;
        }

        auto res = quatA * k0 + quatB * k1;
        res = res / res.norm();


        auto out = std::make_shared<zeno::NumericObject>();
        out->set(res.to_array());
        set_output("res",std::move(out));
    }
};

ZENDEFNODE(SlerpQuaternion, {{{"vec4f","quatA","0,0,0,1"},
                                    {"vec4f","quatB","0,0,0,1"},
                                    {"float","wA","1"}},
                                    {"res"},
                                    {
                                    },
                                    {"ZSGeometry"}});


};