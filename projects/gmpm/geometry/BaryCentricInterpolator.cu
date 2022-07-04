#include "kernel/bary_centric_weights.hpp"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno{

using T = float;
using vec3 = zs::vec<T,3>;
using vec4 = zs::vec<T,4>;
using mat3 = zs::vec<T,3,3>;
using mat4 = zs::vec<T,4,4>;

struct ZSComputeBaryCentricWeights : INode {
    void apply() override {
        using namespace zs;
        auto zsvolume = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");
        // the bvh of zstets
        // auto lbvh = get_input<zeno::LBvh>("lbvh");
        auto thickness = get_param<float>("bvh_thickness");
        auto fitting_in = get_param<int>("fitting_in");

        auto bvh_channel = get_param<std::string>("bvh_channel");
        auto tag = get_param<std::string>("tag");

        const auto& verts = zsvolume->getParticles();
        const auto& eles = zsvolume->getQuadraturePoints();

        const auto& everts = zssurf->getParticles();
        const auto& e_eles = zssurf->getQuadraturePoints();

        auto &bcw = (*zsvolume)[tag];
        bcw = typename ZenoParticles::particles_t({{"inds",1},{"w",4},{"cnorm",1}},everts.size(),zs::memsrc_e::device,0);


        auto cudaExec = zs::cuda_exec();
        const auto numFEMVerts = verts.size();
        const auto numFEMEles = eles.size();
        const auto numEmbedVerts = bcw.size();
        const auto numEmbedEles = e_eles.size();

        constexpr auto space = zs::execspace_e::cuda;

        compute_barycentric_weights<T>(cudaExec,verts,eles,everts,"x",bcw,"inds","w",fitting_in);

        auto e_dim = e_eles.getChannelSize("inds");

        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw)] ZS_LAMBDA (int vi) mutable {
                using T = typename RM_CVREF_T(bcw)::value_type;
                bcw("cnorm",vi) = (T)0.;
        });

        zs::Vector<T> nmEmbedVerts(eles.get_allocator(),eles.size());
        cudaExec(zs::range(eles.size()),[nmEmbedVerts = proxy<space>(nmEmbedVerts)]
            ZS_LAMBDA(int ei) mutable{
                using T = typename RM_CVREF_T(bcw)::value_type;
                nmEmbedVerts[ei] = (T)0.;
        });

        if(e_dim !=3 && e_dim !=4) {
            throw std::runtime_error("INVALID EMBEDDED PRIM TOPO");
        }

        cudaExec(zs::range(bcw.size()),
            [everts = proxy<space>({},everts),bcw = proxy<space>({},bcw),execTag = wrapv<space>{},nmEmbedVerts = proxy<space>(nmEmbedVerts)]
                ZS_LAMBDA (int vi) mutable {
                    using T = typename RM_CVREF_T(bcw)::value_type;
                    auto ei = reinterpret_bits<int>(bcw("inds",vi));
                    if(ei < 0)
                        return;
                    atomic_add(execTag,&nmEmbedVerts[ei],(T)1.0);                  
        });

        cudaExec(zs::range(bcw.size()),
            [bcw = proxy<space>({},bcw),nmEmbedVerts = proxy<space>(nmEmbedVerts)] 
                ZS_LAMBDA(int vi) mutable{
                    auto ei = reinterpret_bits<int>(bcw("inds",vi));
                    if(ei < 0)
                        bcw("cnorm",vi) = (T)0.0;
                    else
                        bcw("cnorm",vi) = (T)1.0/(T)nmEmbedVerts[ei];
        });


        set_output("zsvolume", zsvolume);
    }
};

ZENDEFNODE(ZSComputeBaryCentricWeights, {{{"interpolator","zsvolume"}, {"embed surf", "zssurf"}},
                            {{"interpolator on gpu", "zsvolume"}},
                            {{"float","bvh_thickness","0"},{"int","fitting_in","1"},{"string","tag","skin_bw"},{"string","bvh_channel","x"}},
                            {"ZSGeometry"}});

struct ZSSampleEmbedVectorField : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("volume");
        auto sampler = get_input<ZenoParticles>("vec_field");

        auto tag = get_param<std::string>("bcw_channel");
        auto sample_attr = get_param<std::string>("sampleAttr");
        auto out_attr = get_param<std::string>("outAttr");
        auto tag_type = get_param<std::string>("type");

        auto cudaExec = zs::cuda_exec();

        auto& verts = zstets->getParticles();
        if(!verts.hasProperty(out_attr))
            verts.append_channels(cudaExec,{{out_attr,3}});

        const auto& sample_verts = sampler->getParticles();
        const auto& sample_eles = sampler->getQuadraturePoints();
  
        if(!sampler->hasAuxData(tag)){
            fmt::print("no specified bcw channel detected, create a new one...\n");
            auto& sample_bcw = (*sampler)[tag];
            sample_bcw = typename ZenoParticles::particles_t({{"inds",1},{"w",4}},verts.size(),zs::memsrc_e::device,0);
        }
        const auto& sample_bcw = (*sampler)[tag];

        if(sample_bcw.size() != verts.size())
            throw std::runtime_error("SMAPLE_BCW SIZE UNEQUAL VOLUME SIZE");
        
        constexpr auto space = zs::execspace_e::cuda;

        auto default_val = vec3::from_array(get_input<zeno::NumericObject>("default")->get<zeno::vec3f>());
        bool on_elm = tag_type == "element";

        cudaExec(zs::range(sample_bcw.size()),
            [sample_bcw = proxy<space>({},sample_bcw),verts = proxy<space>({},verts),sample_eles = proxy<space>({},sample_eles),sample_verts = proxy<space>({},sample_verts),
                sample_attr = zs::SmallString(sample_attr),out_attr = zs::SmallString(out_attr),default_val,on_elm] ZS_LAMBDA(int vi) mutable {
                    auto ei = reinterpret_bits<int>(sample_bcw("inds",vi));
                    if(ei < 0){
                        verts.tuple<3>(out_attr,vi) = default_val;
                        return;
                    }
                    if(on_elm){
                        verts.tuple<3>(out_attr,vi) = sample_eles.pack<3>(sample_attr,ei);
                        return;
                    }

                    const auto& w = sample_bcw.pack<4>("w",vi);
                    verts.tuple<3>(out_attr,vi) = vec3::zeros();
                    for(int i = 0;i < 4;++i){
                        auto idx = sample_eles.pack<4>("inds",ei).reinterpret_bits<int>()[i];
                        verts.tuple<3>(out_attr,vi) = verts.pack<3>(out_attr,vi) + w[i] * sample_verts.pack<3>(sample_attr,idx);
                    }
        });


        set_output("volume",zstets);
    }

};

ZENDEFNODE(ZSSampleEmbedVectorField, {{{"volume"}, {"embed vec field", "vec_field"},{"default value","default"}},
                            {{"out volume", "volume"}},
                            {{"string","bcw_channel","bcw"},{"string","sampleAttr","vec_field"},{"string","outAttr"," vec_field"},{"enum element vert","type","element"}},
                            {"ZSGeometry"}});

struct ZSSampleEmbedTagField : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("volume");
        auto sampler = get_input<ZenoParticles>("tag_field");

        auto tag = get_param<std::string>("bcw_channel");
        auto sample_attr = get_param<std::string>("tagAttr");
        auto out_attr = get_param<std::string>("outAttr");
        auto tag_type = get_param<std::string>("type");

        auto default_tag_value = get_param<int>("default");

        auto cudaExec = zs::cuda_exec();

        auto& verts = zstets->getParticles();
        if(!verts.hasProperty(out_attr))
            verts.append_channels(cudaExec,{{out_attr,1}});

        const auto& sample_verts = sampler->getParticles();
        const auto& sample_eles = sampler->getQuadraturePoints();

        if(!sampler->hasAuxData(tag)){
            fmt::print("no specified bcw channel detected, create a new one...\n");
            auto& sample_bcw = (*sampler)[tag];
            sample_bcw = typename ZenoParticles::particles_t({{"inds",1},{"w",4}},verts.size(),zs::memsrc_e::device,0);
        }
        const auto& sample_bcw = (*sampler)[tag];

        if(sample_bcw.size() != verts.size())
            throw std::runtime_error("SMAPLE_BCW SIZE UNEQUAL VOLUME SIZE");
        
        constexpr auto space = zs::execspace_e::cuda;

        bool on_elm = tag_type == "element";

        cudaExec(zs::range(sample_bcw.size()),
            [sample_bcw = proxy<space>({},sample_bcw),verts = proxy<space>({},verts),sample_eles = proxy<space>({},sample_eles),sample_verts = proxy<space>({},sample_verts),
                sample_attr = zs::SmallString(sample_attr),out_attr = zs::SmallString(out_attr),default_tag_value,on_elm] ZS_LAMBDA(int vi) mutable {
                     auto ei = reinterpret_bits<int>(sample_bcw("inds",vi));
                     if(ei < 0){
                         verts(out_attr,vi) = reinterpret_bits<float>(default_tag_value);
                         return;
                     }

                    if(on_elm)
                        verts(out_attr,vi) = sample_eles(sample_attr,ei);
                    else{
                        auto idx = sample_eles.pack<4>("inds",ei).reinterpret_bits<int>()[0];
                        verts(out_attr,vi) = sample_verts(sample_attr,idx);
                    }
        });


        set_output("volume",zstets);
    }

};

ZENDEFNODE(ZSSampleEmbedTagField, {{{"volume"}, {"embed tag field", "tag_field"},{"default value","default"}},
                            {{"out volume", "volume"}},
                            {{"string","interpolate_tag","bws"},{"string","sampleAttr","vec_field"},{"string","outAttr"," vec_field"},{"enum element vert","type","element"}},
                            {"ZSGeometry"}});


struct ZSInterpolateEmbedPrim : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto tag = get_param<std::string>("tag");
        auto outAttr = get_param<std::string>("outAttr");

        auto cudaExec = zs::cuda_exec();

        auto &everts = zssurf->getParticles();
        if(!everts.hasProperty(outAttr))
            everts.append_channels(cudaExec, {{outAttr, 3}});
        
        const auto& verts = zstets->getParticles();
        const auto& eles = zstets->getQuadraturePoints();
        const auto& bcw = (*zstets)[tag];

        const auto nmEmbedVerts = bcw.size();

        if(everts.size() != nmEmbedVerts)
            throw std::runtime_error("INPUT SURF SIZE AND BCWS SIZE DOES NOT MATCH");


        constexpr auto space = zs::execspace_e::cuda;

        cudaExec(zs::range(nmEmbedVerts),
            [outAttr = zs::SmallString{outAttr},verts = proxy<space>({},verts),eles = proxy<space>({},eles),bcw = proxy<space>({},bcw),everts = proxy<space>({},everts)] ZS_LAMBDA (int vi) mutable {
                const auto& ei = bcw.pack<1>("inds",vi).reinterpret_bits<int>()[0];
                if(ei < 0)
                    return;
                const auto& w = bcw.pack<4>("w",vi);

                everts.tuple<3>(outAttr,vi) = vec3::zeros();

                for(size_t i = 0;i < 4;++i){
                    // const auto& idx = eles.pack<4>("inds",ei).reinterpret_bits<int>()[i];
                    const auto idx = reinterpret_bits<int>(eles("inds", i, ei));
                    everts.tuple<3>(outAttr,vi) = everts.pack<3>(outAttr,vi) + w[i] * verts.pack<3>("x", idx);
                }
#if 0
                if(vi == 100){
                    auto vert = everts.pack<3>(outAttr,vi);
                    printf("V<%d>->E<%d>(%f,%f,%f,%f) :\t%f\t%f\t%f\n",vi,ei,w[0],w[1],w[2],w[3],vert[0],vert[1],vert[2]);
                }
#endif
        });
        set_output("zssurf",zssurf);
    }
};

ZENDEFNODE(ZSInterpolateEmbedPrim, {{{"zsvolume"}, {"embed primitive", "zssurf"}},
                            {{"embed primitive", "zssurf"}},
                            {{"string","outAttr","x"},{"string","tag","skin_bw"}},
                            {"ZSGeometry"}});


} // namespace zeno