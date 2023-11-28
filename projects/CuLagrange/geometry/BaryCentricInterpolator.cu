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

#include "zensim/container/Bcht.hpp"
#include "kernel/tiled_vector_ops.hpp"

#include "kernel\global_intersection_analysis.hpp"

#include <iostream>

namespace zeno{

using T = float;
using vec3 = zs::vec<T,3>;
using vec4 = zs::vec<T,4>;
using mat3 = zs::vec<T,3,3>;
using mat4 = zs::vec<T,4,4>;


// 给定一个四面网格与一组点，计算每个点在四面体网格单元中的质心坐标
struct ZSComputeRBFWeights : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        constexpr auto exec_tag = wrapv<space>{};
        auto cudaPol = cuda_exec();

        auto zspars = get_input<ZenoParticles>("zspars");
        auto nei_zspars = get_input<ZenoParticles>("nei_zspars");

        auto uniform_radius = get_input2<float>("uniform_radius");
        auto uniform_nei_radius = get_input2<float>("uniform_neighbor_radius");

        auto radius_attr = get_input2<std::string>("radius_attr");
        auto neighbor_radius_attr = get_input2<std::string>("neighbor_radius_attr");

        auto& verts = zspars->getParticles();
        auto xtag = get_input2<std::string>("xtag");

        const auto& nverts = nei_zspars->getParticles();
        auto nei_xtag = get_input2<std::string>("neighbor_xtag");

        auto max_number_binding_per_vertex = get_input2<size_t>("max_number_binders");
        
        zs::bht<int,2,int> close_proximity{verts.get_allocator(),max_number_binding_per_vertex * nverts.size()};
        close_proximity.reset(cudaPol,true);

        GIA::retrieve_intersected_sphere_pairs(cudaPol,
            verts,xtag,uniform_radius,radius_attr,
            nverts,nei_xtag,uniform_nei_radius,neighbor_radius_attr,
            close_proximity);

        zs::Vector<int> close_proximity_count_buffer{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(close_proximity_count_buffer),[] ZS_LAMBDA(auto& cnt) mutable {cnt = 0;});

        cudaPol(zip(zs::range(close_proximity.size()),close_proximity._activeKeys),[
            close_proximity_count_buffer = proxy<space>(close_proximity_count_buffer),
            exec_tag = exec_tag
        ] ZS_LAMBDA(auto id,const auto& pair) mutable {
            auto vi = pair[0];
            auto nvi = pair[1];
            atomic_add(exec_tag,&close_proximity_count_buffer[vi],1);
        });

        cudaPol(zs::range(verts.size()),[
            close_proximity_count_buffer = proxy<space>(close_proximity_count_buffer)] ZS_LAMBDA(int vi) mutable {
                printf("nm_close_proximity[%d] : %d\n",vi,close_proximity_count_buffer[vi]);
        });

        zs::Vector<int> exclusive_offsets{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(exclusive_offsets),[] ZS_LAMBDA(auto& offset) mutable {offset = 0;});
        exclusive_scan(cudaPol,std::begin(close_proximity_count_buffer),std::end(close_proximity_count_buffer),std::begin(exclusive_offsets));

        cudaPol(zs::range(verts.size()),[
            exclusive_offsets = proxy<space>(exclusive_offsets)] ZS_LAMBDA(int vi) mutable {
                printf("exclusive_offsets[%d] : %d\n",vi,exclusive_offsets[vi]);
        });        

        auto rbf_weights_name = get_input2<std::string>("rbf_weights_name");
        auto& rbf_weights = (*zspars)[rbf_weights_name];

        auto rbf_weights_offset = std::string{rbf_weights_name + "_offset"};
        auto rbf_weights_count = std::string{rbf_weights_name + "_count"};

        verts.append_channels(cudaPol,{{rbf_weights_offset,1},{rbf_weights_count,1}});
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            rbf_weights_offset = zs::SmallString(rbf_weights_offset),
            rbf_weights_count = zs::SmallString(rbf_weights_count),
            close_proximity_count_buffer = proxy<space>(close_proximity_count_buffer),
            exclusive_offsets = proxy<space>(exclusive_offsets)] ZS_LAMBDA(int vi) mutable {
                verts(rbf_weights_count,vi) = zs::reinterpret_bits<float>(close_proximity_count_buffer[vi]);
                verts(rbf_weights_offset,vi) = zs::reinterpret_bits<float>(exclusive_offsets[vi]);
        });



        // verts.append_channels(cudaPol,{{rbf_weights_count,1},{rbf_weights_offset,1}});

        rbf_weights = typename ZenoParticles::particles_t({
            {"inds",1},
            {"w",1}},close_proximity.size(),zs::memsrc_e::device,0);

        auto varience = get_input2<float>("varience");

        cudaPol(zip(zs::range(close_proximity.size()),close_proximity._activeKeys),[
            exec_tag = exec_tag,
            varience = varience,
            xtag = zs::SmallString(xtag),
            nxtag = zs::SmallString(nei_xtag),
            verts = proxy<space>({},verts),
            nverts = proxy<space>({},nverts),
            rbf_weights = proxy<space>({},rbf_weights),
            close_proximity_count_buffer = proxy<space>(close_proximity_count_buffer),
            exclusive_offsets = proxy<space>(exclusive_offsets)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto nvi = pair[1];
                auto offset = exclusive_offsets[vi];
                auto local_idx = atomic_add(exec_tag,&close_proximity_count_buffer[vi],-1) - 1;
                if(local_idx < 0)
                    printf("negative_local_idx detected!!! check algorithm\n");
                
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto np = nverts.pack(dim_c<3>,nxtag,nvi);

                auto dist2 = (p - np).l2NormSqr();
                auto w = zs::exp(-dist2 / varience);

                rbf_weights("inds",offset + local_idx) = zs::reinterpret_bits<float>(nvi);
                rbf_weights("w",offset + local_idx) = w;
        });



        set_output("zspars",get_input("zspars"));
        set_output("nei_zspars",get_input("nei_zspars"));
    }
};

ZENDEFNODE(ZSComputeRBFWeights, {{{"zspars"},
                                    {"nei_zspars"},
                                    {"float","uniform_radius","1"},
                                    {"float","uniform_neighbor_radius","1"},
                                    {"string","radius_attr","r"},
                                    {"string","neighbor_radius_attr","r"},
                                    {"string","xtag","x"},
                                    {"string","neighbor_xtag","x"},
                                    {"int","max_number_binders","10"},
                                    {"string","rbf_weights_name","rbf"},
                                    {"float","varience","10"}
                                },
                            {
                                {"zspars"},
                                {"nei_zspars"},
                            },
                            {},
                            {"ZSGeometry"}});

struct ZSSample : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        constexpr auto exec_tag = wrapv<space>{};
        auto cudaPol = cuda_exec();

        auto dest = get_input<ZenoParticles>("to");
        auto source = get_input<ZenoParticles>("from");
        auto sampler_name = get_input2<std::string>("sampler_name");

        auto& dverts = dest->getParticles();
        auto to_attr = get_input2<std::string>("to_attr");

        const auto& sverts = source->getParticles();
        auto from_attr = get_input2<std::string>("from_attr");

        if(!dest->hasAuxData(sampler_name)) {
            std::cout << "the dest particles has no specified [" << sampler_name << "] sampler" << std::endl;
            throw std::runtime_error("the dest particles has no specified sampler name!");
        }

        auto sampler_count_attr = std::string(sampler_name + "_count");
        auto sampler_offset_attr = std::string(sampler_name + "_offset");

        if(!dverts.hasProperty(sampler_count_attr)) {
            std::cout << "the dest particles has no specified [" << sampler_count_attr << "] channel" << std::endl;
            throw std::runtime_error("the dest particles has no specified sampler_count_attr channel");
        }

        if(!dverts.hasProperty(sampler_offset_attr)) {
            std::cout << "the dest particles has no specified [" << sampler_offset_attr << "] channel" << std::endl;
            throw std::runtime_error("the dest particles has no specified sampler_offset_attr channel");
        }

        if(!dverts.hasProperty(to_attr)) {
            std::cout << "the dest particles has no specified sample attr [" << to_attr << "]" << std::endl;
            throw std::runtime_error("the dest particles has no specified sample_attr");
        }

        if(!sverts.hasProperty(from_attr)) {
            std::cout << "the source particles has no specified sample attr [" << from_attr << "]" << std::endl;
            throw std::runtime_error("the source particles has no specified sample_attr");
        }

        if(dverts.getPropertySize(to_attr) != sverts.getPropertySize(from_attr)) {
            std::cout << "the size of two sample channels does not match"  << std::endl;
            throw std::runtime_error("the size of two sample channels does not match");
        }

        const auto& sampler_buffer = (*dest)[sampler_name];

        TILEVEC_OPS::fill(cudaPol,dverts,to_attr,(T)0);

        cudaPol(zs::range(dverts.size()),[
            attr_dim = dverts.getPropertySize(to_attr),
            sampler_count_attr = zs::SmallString(sampler_count_attr),
            sampler_offset_attr = zs::SmallString(sampler_offset_attr),
            dverts = proxy<space>({},dverts),to_attr_offset = dverts.getPropertyOffset(to_attr),
            sverts = proxy<space>({},sverts),from_attr_offset = sverts.getPropertyOffset(from_attr),
            sampler_buffer = proxy<space>({},sampler_buffer)] ZS_LAMBDA(int dvi) mutable {
                auto nm_samples = zs::reinterpret_bits<int>(dverts(sampler_count_attr,dvi));
                auto sample_weight_offset = zs::reinterpret_bits<int>(dverts(sampler_offset_attr,dvi));
                
                // printf("nm_samples[%d] : %d\n",dvi,nm_samples);

                auto wsum = (T)0.0;

                for(int i = 0;i != nm_samples;++i) {
                    auto idx = sample_weight_offset + i;
                    auto svi = zs::reinterpret_bits<int>(sampler_buffer("inds",idx));
                    auto w  = sampler_buffer("w",idx);
                    wsum += w;
                    for(int d = 0;d != attr_dim;++d)
                        dverts(to_attr_offset + d,dvi) += sverts(from_attr_offset + d,svi) * w;
                }

                for(int d = 0;d != attr_dim;++d)
                    dverts(to_attr_offset + d,dvi) /= (wsum + (T)1e-6);
        });

        set_output("to",get_input("to"));
        set_output("from",get_input("from"));
    }
};

ZENDEFNODE(ZSSample, {{
        {"to"},
        {"from"},
        {"string","sampler_name","sampler_name"},
        {"string","to_attr","to_attr"},
        {"string","from_attr","from_attr"}
    },
    {{"to"},{"from"}},
    {},
    {"ZSGeometry"}});

struct ZSComputeBaryCentricWeights : INode {
    void apply() override {
        using namespace zs;

        // fmt::print("ENTERING NODES\n");
        // std::cout << "ENTERING NODES" << std::endl;

        auto zsvolume = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");
        auto mark_embed_elm = get_input2<int>("mark_elm");
        // the bvh of zstets
        // auto lbvh = get_input<zeno::LBvh>("lbvh");
        auto thickness = get_param<float>("bvh_thickness");
        auto fitting_in = get_param<int>("fitting_in");

        auto bvh_channel = get_param<std::string>("bvh_channel");
        auto tag = get_input2<std::string>("tag");

        auto& verts = zsvolume->getParticles();
        auto& eles = zsvolume->getQuadraturePoints();

        const auto& everts = zssurf->getParticles();
        // const auto& e_eles = zssurf->getQuadraturePoints();

        auto &bcw = (*zsvolume)[tag];

        bcw = typename ZenoParticles::particles_t({
            {"X",3},
            {"inds",1},
            {"w",4},
            {"strength",1},
            {"cnorm",1}},everts.size(),zs::memsrc_e::device,0);
        

        // auto topo_tag = tag + std::string("_topo");
        // auto &bcw_topo = (*zsvolume)[topo_tag];

        // auto e_dim = e_eles.getPropertySize("inds");
        // bcw_topo = typename ZenoParticles::particles_t({{"inds",e_dim}},e_eles.size(),zs::memsrc_e::device,0);


        auto cudaExec = zs::cuda_exec();
        const auto numFEMVerts = verts.size();
        const auto numFEMEles = eles.size();
        const auto numEmbedVerts = bcw.size();
        // const auto numEmbedEles = e_eles.size();
        constexpr auto space = zs::execspace_e::cuda;

        TILEVEC_OPS::copy<3>(cudaExec,everts,"x",bcw,"X");

        compute_barycentric_weights(cudaExec,verts,eles,everts,"x",bcw,"inds","w",thickness,fitting_in);

        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw),fitting_in] ZS_LAMBDA(int vi) mutable {
                auto idx = reinterpret_bits<int>(bcw("inds",vi));
                if(fitting_in && idx < 0)
                    printf("Unbind vert %d under fitting-in mode\n",vi);
            }
        );


        // cudaExec(zs::range(e_eles.size()),[e_dim = e_dim,
        //     e_eles = proxy<space>({},e_eles),bcw_topo = proxy<space>({},bcw_topo)] ZS_LAMBDA(int ei) mutable {
        //         for(int i = 0;i != e_dim;++i)
        //             bcw_topo("inds",i,ei) = e_eles("inds",i,ei);
        // });


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

        // if(e_dim !=3 && e_dim !=4) {
        //     throw std::runtime_error("INVALID EMBEDDED PRIM TOPO");
        // }  

        if(mark_embed_elm && everts.hasProperty("tag")){
            eles.append_channels(cudaExec,{{"nmBones",1},{"bdw",1}});

            cudaExec(zs::range(eles.size()),
                [eles = proxy<space>({},eles)] ZS_LAMBDA(int elm_id) mutable{
                    eles("nmBones",elm_id) = (T)0.0;
                    eles("bdw",elm_id) = (T)1.0;
            });  


            auto nmBones = get_input2<int>("nmCpns");
            using vec2i = zs::vec<int,2>;
            using vec3i = zs::vec<int,3>;
            bcht<vec2i, int, true, universal_hash<vec2i>, 32> ebtab{eles.get_allocator(), eles.size() * nmBones};
            cudaExec(zs::range(bcw.size()),
                [bcw = proxy<space>({},bcw),ebtab = proxy<space>(ebtab),everts = proxy<space>({},everts)] 
                    ZS_LAMBDA(int vi) mutable{
                        auto ei = reinterpret_bits<int>(bcw("inds",vi));
                        if(ei < 0)
                            return;
                        else{
                            int tag = (int)everts("tag",vi);
                            ebtab.insert(vec2i{ei,tag});
                        }
            });

            cudaExec(zs::range(eles.size()),
                [eles = proxy<space>({},eles),ebtab = proxy<space>(ebtab),nmBones] ZS_LAMBDA(int ei) mutable {
                    for(int i = 0;i != nmBones;++i) {
                        auto res = ebtab.query(vec2i{ei,i});
                        if(res < 0)
                            continue;
                        eles("nmBones",ei) += (T)1.0;
                    }
                    // if(eles("nmBones",ei) > 0)
                        // printf("nmEmbedCmps[%d] : [%d]\n",ei,(int)eles("nmBones",ei));
            });
        }else {
            eles.append_channels(cudaExec,{{"nmBones",1},{"bdw",1}});
            cudaExec(zs::range(eles.size()),[
                eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {
                    eles("bdw",ei) = (T)1.0;
                    eles("nmBones",ei) = (T)1.0;
            });
        }

        cudaExec(zs::range(bcw.size()),
            [everts = proxy<space>({},everts),
                    bcw = proxy<space>({},bcw),
                    execTag = wrapv<space>{},
                    nmEmbedVerts = proxy<space>(nmEmbedVerts),
                    eles = proxy<space>({},eles),
                    verts = proxy<space>({},verts)]
                ZS_LAMBDA (int vi) mutable {
                    using T = typename RM_CVREF_T(bcw)::value_type;
                    auto ei = reinterpret_bits<int>(bcw("inds",vi));
                    if(ei < 0)
                        return;
                    auto tet = eles.pack(dim_c<3>,"inds",ei).reinterpret_bits(int_c);
                    atomic_add(execTag,&nmEmbedVerts[ei],(T)1.0);                  
        });

        cudaExec(zs::range(bcw.size()),
            [bcw = proxy<space>({},bcw),nmEmbedVerts = proxy<space>(nmEmbedVerts),eles = proxy<space>({},eles),everts = proxy<space>({},everts)] 
                ZS_LAMBDA(int vi) mutable{
                    auto ei = reinterpret_bits<int>(bcw("inds",vi));
                    if(everts.hasProperty("strength"))
                        bcw("strength",vi) = everts("strength",vi);
                    else
                        bcw("strength",vi) = (T)1.0;
                    if(ei >= 0){
                        auto alpha = (T)1.0/(T)nmEmbedVerts[ei];
                        bcw("cnorm",vi) = (T)alpha;
                        // if(eles("nmBones",ei) > (T)1.5)
                        //     eles("bdw",ei) = (T)0.0;
                    }

                    // if(ei < 0 || eles("nmBones",ei) > (T)1.5){
                    //     // bcw("strength",vi) = (T)0.0;
                    //     bcw("cnorm",vi) = (T)0.0;
                    //     if(ei >= 0)
                    //         eles("bdw",ei) = (T)0.0;
                    // }
                    // else{

                    //     // bcw("cnorm",vi) = (T)1.0;
                    // }
        });

        
        // we might also do some smoothing on cnorm

        set_output("zsvolume", zsvolume);
    }
};

ZENDEFNODE(ZSComputeBaryCentricWeights, {{{"interpolator","zsvolume"}, {"embed surf", "zssurf"},{"int","mark_elm","0"},{"int","nmCpns","1"},{"string","tag","skin"}},
                            {{"interpolator on gpu", "zsvolume"}},
                            {{"float","bvh_thickness","0"},{"int","fitting_in","1"},{"string","bvh_channel","x"}},
                            {"ZSGeometry"}});



struct VisualizeInterpolator : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zsvolume = get_input<ZenoParticles>("zsvolume");
        auto tag = get_input2<std::string>("interpolator_name");
        const auto& bcw = (*zsvolume)[tag].clone({zs::memsrc_e::host});
        auto topo_tag = tag + std::string("_topo");
        const auto &bcw_topo = (*zsvolume)[topo_tag].clone({zs::memsrc_e::host});

        auto bcw_vis = std::make_shared<zeno::PrimitiveObject>();
        bcw_vis->resize(bcw.size());
        auto& bcw_X = bcw_vis->verts;
        auto& bcw_cnorm = bcw_vis->add_attr<float>("cnorm");
        auto& bcw_strength = bcw_vis->add_attr<float>("strength");

        auto ompPol = omp_exec();  
        constexpr auto omp_space = execspace_e::openmp;        
        ompPol(zs::range(bcw.size()),
            [&bcw_X,&bcw_cnorm,&bcw_strength,bcw = proxy<omp_space>({},bcw)] (int vi) mutable {
                bcw_X[vi] = bcw.pack(dim_c<3>,"X",vi).to_array();
                bcw_cnorm[vi] = bcw("cnorm",vi);
                bcw_strength[vi] = bcw("strength",vi);
        });

        set_output("bcw_vis",std::move(bcw_vis));
    }
};

ZENDEFNODE(VisualizeInterpolator, {{{"interpolator","zsvolume"},{"string","interpolator_name","skin"}},
                            {{"visual bcw", "bcw_vis"}},
                            {},
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
                        verts.template tuple<3>(out_attr,vi) = default_val;
                        return;
                    }
                    if(on_elm){
                        verts.template tuple<3>(out_attr,vi) = sample_eles.template pack<3>(sample_attr,ei);
                        return;
                    }

                    const auto& w = sample_bcw.pack<4>("w",vi);
                    verts.template tuple<3>(out_attr,vi) = vec3::zeros();
                    for(int i = 0;i < 4;++i){
                        auto idx = sample_eles.template pack<4>("inds",ei).template reinterpret_bits<int>()[i];
                        verts.template tuple<3>(out_attr,vi) = verts.template pack<3>(out_attr,vi) + w[i] * sample_verts.template pack<3>(sample_attr,idx);
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


struct ZSInterpolateEmbedAttr : zeno::INode {
    template<int DIM,typename SRC_TILEVEC,typename DST_TILEVEC,typename TOPO_TIELVEC,typename BCW_TILEVEC>
    void interpolate_p2p_imp(const std::string& srcAttr,const std::string& dstAttr,
            const SRC_TILEVEC& src_tilevec,DST_TILEVEC& dst_tilevec,const TOPO_TIELVEC& src_topo,const BCW_TILEVEC& bcw) {
        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if(!dst_tilevec.hasProperty(dstAttr))
            dst_tilevec.append_channels(cudaExec, {{dstAttr, DIM}});

        cudaExec(zs::range(dst_tilevec.size()),
            [srcAttr = zs::SmallString{srcAttr},dstAttr = zs::SmallString{dstAttr},
                    src_tilevec = zs::proxy<space>({},src_tilevec), bcw = zs::proxy<space>({},bcw),
                    dst_tilevec = zs::proxy<space>({},dst_tilevec),
                    src_topo = zs::proxy<space>({},src_topo)] ZS_LAMBDA (int vi) mutable {
                using T = typename RM_CVREF_T(dst_tilevec)::value_type;
                const auto& ei = bcw.template pack<1>("inds",vi).template reinterpret_bits<int>()[0];
                if(ei < 0)
                    return;
                const auto& inds = src_topo.template pack<4>("inds",ei).template reinterpret_bits<int>();

                const auto& w = bcw.template pack<4>("w",vi);
                dst_tilevec.template tuple<DIM>(dstAttr,vi) = zs::vec<T,DIM>::zeros();
                for(size_t i = 0;i < 4;++i){
                    auto idx = inds[i];
                    dst_tilevec.template tuple<DIM>(dstAttr,vi) = dst_tilevec.template pack<DIM>(dstAttr,vi) + w[i] * src_tilevec.template pack<DIM>(srcAttr, idx);
                }

        });
    }

    template<int DIM,typename QUAD_TILEVEC,typename POINT_TILEVEC,typename BCW_TILEVEC>
    void interpolate_q2p_imp(const std::string& quadAttr,const std::string& pointAttr,
            const QUAD_TILEVEC& quad_tilevec,POINT_TILEVEC& point_tilevec,const BCW_TILEVEC& bcw) {
        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if(!point_tilevec.hasProperty(pointAttr))
            point_tilevec.append_channels(cudaExec, {{pointAttr, DIM}});   
             
        cudaExec(zs::range(point_tilevec.size()),
            [pointAttr = zs::SmallString{pointAttr},quadAttr = zs::SmallString{quadAttr},
                point_tilevec = zs::proxy<space>({},point_tilevec),bcw = zs::proxy<space>({},bcw),
                quad_tilevec = zs::proxy<space>({},quad_tilevec)] ZS_LAMBDA (int vi) mutable {
            using T = typename RM_CVREF_T(point_tilevec)::value_type;
            const auto& ei = bcw.template pack<1>("inds",vi).template reinterpret_bits<int>()[0];
            if(ei < 0)
                return;
            point_tilevec.template tuple<DIM>(pointAttr,vi) = quad_tilevec.template pack<DIM>(quadAttr,ei);
        });
    }


    void apply() override {
        using namespace zs;
        auto source = get_input<ZenoParticles>("source");
        auto dest = get_input<ZenoParticles>("dest");

        auto srcAttr = get_param<std::string>("srcAttr");
        auto dstAttr = get_param<std::string>("dstAttr");
        auto bcw_tag = get_input2<std::string>("bcw_tag");
        auto strategy = get_param<std::string>("strategy");
        const auto& bcw = (*source)[bcw_tag];
        auto& dest_pars = dest->getParticles();

        if(bcw.size() != dest_pars.size()) {
            fmt::print("the dest and bcw's size not match\n");
            throw std::runtime_error("the dest and bcw's size not match");
        }

        
        if(strategy == "p2p") {
            const auto& source_pars = source->getParticles();
            const auto& topo = source->getQuadraturePoints();
            if(!source_pars.hasProperty(srcAttr)) {
                fmt::print("the source have no {} channel\n",srcAttr);
                throw std::runtime_error("the source have no specified channel");
            }           
            if(topo.getPropertySize("inds") != 4) {
                fmt::print("only support tetrahedra mesh as source\n");
                throw std::runtime_error("only support tetrahedra mesh as source");
            }
            if(dest_pars.hasProperty(dstAttr) && dest_pars.getPropertySize(dstAttr) != source_pars.getPropertySize(srcAttr)){
                fmt::print("the dest attr_{} {} and source attr_{} {} not match in size\n",
                    dstAttr,
                    dest_pars.getPropertySize(dstAttr),
                    srcAttr,
                    source_pars.getPropertySize(srcAttr));
                throw std::runtime_error("the dest attr and source attr not match in size");
            }

            if(source_pars.getPropertySize(srcAttr) == 1)
                interpolate_p2p_imp<1>(srcAttr,dstAttr,source_pars,dest_pars,topo,bcw);
            if(source_pars.getPropertySize(srcAttr) == 2)
                interpolate_p2p_imp<2>(srcAttr,dstAttr,source_pars,dest_pars,topo,bcw);
            if(source_pars.getPropertySize(srcAttr) == 3)
                interpolate_p2p_imp<3>(srcAttr,dstAttr,source_pars,dest_pars,topo,bcw);
        }else if(strategy == "q2p") {
            const auto& source_quads = source->getQuadraturePoints();
            if(!source_quads.hasProperty(srcAttr)) {
                fmt::print("the source have no {} channel\n",srcAttr);
                throw std::runtime_error("the source have no specified channel");
            }    
            if(dest_pars.hasProperty(dstAttr) && dest_pars.getPropertySize(dstAttr) != source_quads.getPropertySize(srcAttr)){
                fmt::print("the dest attr_{} and source attr_{} not match in size\n",dstAttr,srcAttr);
                throw std::runtime_error("the dest attr and source attr not match in size");
            }

            if(source_quads.getPropertySize(srcAttr) == 1)
                interpolate_q2p_imp<1>(srcAttr,dstAttr,source_quads,dest_pars,bcw);
            if(source_quads.getPropertySize(srcAttr) == 2)
                interpolate_q2p_imp<2>(srcAttr,dstAttr,source_quads,dest_pars,bcw);
            if(source_quads.getPropertySize(srcAttr) == 3)
                interpolate_q2p_imp<3>(srcAttr,dstAttr,source_quads,dest_pars,bcw);
        }
        set_output("dest",dest);
    }
};


ZENDEFNODE(ZSInterpolateEmbedAttr, {{{"source"}, {"dest"},{"string","bcw_tag","skin_bw"}},
                            {{"dest"}},
                            {
                                {"string","srcAttr","x"},
                                {"string","dstAttr","x"},
                                {"enum p2p q2p","strategy","p2p"}

                            },
                            {"ZSGeometry"}});

// deprecated
struct ZSInterpolateEmbedPrim : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto tag = get_param<std::string>("tag");
        auto inAttr = get_param<std::string>("inAttr");
        auto outAttr = get_param<std::string>("outAttr");
        // auto refAttr = get_param<std::string>("refAttr");

        // auto useDispMap = get_param<int>("useDispMap");
        // auto refDispMapTag = get_param<std::string>("refDispMapTag");
        // auto outDispMapTag = get_param<std::string>("outDispMapTag");

        // auto use_xform = get_param<int>("use_xform");

        auto &everts = zssurf->getParticles();
    
        const auto& verts = zstets->getParticles();
        const auto& eles = zstets->getQuadraturePoints();
        const auto& bcw = (*zstets)[tag];

        // if(useDispMap && (!everts.hasProperty(refDispMapTag) || !everts.hasProperty(outDispMapTag))) {
        //     fmt::print("the input everts have no {} or {} dispMap when useDispMap is on\n",refDispMapTag,outDispMapTag);
        //     throw std::runtime_error("the input everts have no specified dispMap when useDispMap is on");
        // }


        // if(use_xform && !everts.hasProperty(refAttr)) {
        //     fmt::print("the input everts have no {} channel when use_xform is on\n",refAttr);
        //     throw std::runtime_error("the input everts have no refAttr channel when use_xform is on");
        // }
        // if(use_xform && !verts.hasProperty(refAttr)) {
        //     fmt::print("the input verts have no {} channel when use_xform is on\n",refAttr);
        //     throw std::runtime_error("the input verts have no refAttr channel when use_xform is on");
        // }

        const auto nmEmbedVerts = bcw.size();
        if(everts.size() != nmEmbedVerts)
            throw std::runtime_error("INPUT SURF SIZE AND BCWS SIZE DOES NOT MATCH");


        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        cudaExec(zs::range(nmEmbedVerts),
            [inAttr = zs::SmallString{inAttr},outAttr = zs::SmallString{outAttr},
                    verts = proxy<space>({},verts),eles = proxy<space>({},eles),
                    bcw = proxy<space>({},bcw),everts = proxy<space>({},everts)
                    // use_xform,refAttr = zs::SmallString{refAttr},
                    // useDispMap,
                    // refDispMapTag = zs::SmallString{refDispMapTag},
                    // outDispMapTag = zs::SmallString{outDispMapTag}
                    ] ZS_LAMBDA (int vi) mutable {
                using T = typename RM_CVREF_T(verts)::value_type;
                const auto& ei = bcw.pack<1>("inds",vi).reinterpret_bits<int>()[0];
                if(ei < 0)
                    return;
                const auto& inds = eles.template pack<4>("inds",ei).template reinterpret_bits<int>();
                // if(use_xform || useDispMap) {
                //     zs::vec<T,3,3> F{};
                //     zs::vec<T,3> b{};

                //     LSL_GEO::deformation_xform(
                //         verts.template pack<3>(inAttr,inds[0]),
                //         verts.template pack<3>(inAttr,inds[1]),
                //         verts.template pack<3>(inAttr,inds[2]),
                //         verts.template pack<3>(inAttr,inds[3]),
                //         verts.template pack<3>(refAttr,inds[0]),
                //         eles.template pack<3,3>("IB",ei),F,b);
                    
                //     everts.template tuple<3>(outAttr,vi) = F * everts.template pack<3>(refAttr,vi) + b;

                //     // if(vi == 0){
                //     //     printf("F : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
                //     //         (float)F(0,0),(float)F(0,1),(float)F(0,2),
                //     //         (float)F(1,0),(float)F(1,1),(float)F(1,2),
                //     //         (float)F(2,0),(float)F(2,1),(float)F(2,2));
                //     //     printf("b : %f %f %f\n",(float)b[0],(float)b[1],(float)b[2]);
                //     // }

                //     if(useDispMap) {
                //         everts.template tuple<3>(outDispMapTag,vi) = F * everts.template pack<3>(refDispMapTag,vi);
                //     }
                // }else{
                    const auto& w = bcw.pack<4>("w",vi);
                    everts.tuple<3>(outAttr,vi) = vec3::zeros();
                    for(size_t i = 0;i < 4;++i){
                        // const auto& idx = eles.pack<4>("inds",ei).reinterpret_bits<int>()[i];
                        // const auto idx = reinterpret_bits<int>(eles("inds", i, ei));
                        auto idx = inds[i];
                        everts.tuple<3>(outAttr,vi) = everts.pack<3>(outAttr,vi) + w[i] * verts.pack<3>(inAttr, idx);
                    }
// #if 0
//                     if(vi == 100){
//                         auto vert = everts.pack<3>(outAttr,vi);
//                         printf("V<%d>->E<%d>(%f,%f,%f,%f) :\t%f\t%f\t%f\n",vi,ei,w[0],w[1],w[2],w[3],vert[0],vert[1],vert[2]);
//                     }
// #endif

                // }
        });
        set_output("zssurf",zssurf);
    }
};

ZENDEFNODE(ZSInterpolateEmbedPrim, {{{"zsvolume"}, {"embed primitive", "zssurf"}},
                            {{"embed primitive", "zssurf"}},
                            {
                                {"string","inAttr","x"},
                                {"string","outAttr","x"},
                                // {"string","refAttr","X"},
                                {"string","tag","skin_bw"}
                                // {"int","use_xform","0"},
                                // {"int","useDispMap","0"},
                                // {"string","refDispMapTag","dX"},
                                // {"string","outDispMapTag","dx"}
                                },
                            {"ZSGeometry"}});


struct ZSDeformEmbedPrim : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zsvolume = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto tag = get_param<std::string>("tag");
        auto inAttr = get_param<std::string>("inAttr");
        auto outAttr = get_param<std::string>("outAttr");

        auto deformField = get_param<std::string>("deformField");

        auto &everts = zssurf->getParticles();

        auto cudaExec = zs::cuda_exec();

        if(!everts.hasProperty(inAttr)) {
            fmt::print("the embed prim has no {} attribute as input\n",inAttr);
            throw std::runtime_error("the embed prim has no attribute as input");
        }
        if(!everts.hasProperty(outAttr))
            everts.append_channels(cudaExec, {{outAttr, 3}});

        
        const auto& verts = zsvolume->getParticles();
        const auto& eles = zsvolume->getQuadraturePoints();
        const auto& bcw = (*zsvolume)[tag];

        if(!eles.hasProperty(deformField)) {
            fmt::print("the embed prim has no {} deformField\n",deformField);
            throw std::runtime_error("the embed prim has no deformField");
        }

        const auto nmEmbedVerts = bcw.size();

        if(everts.size() != nmEmbedVerts)
            throw std::runtime_error("INPUT SURF SIZE AND BCWS SIZE DOES NOT MATCH");


        constexpr auto space = zs::execspace_e::cuda;

        cudaExec(zs::range(nmEmbedVerts),
            [inAttr = zs::SmallString{inAttr},outAttr = zs::SmallString{outAttr},
                    everts = proxy<space>({},everts),eles = proxy<space>({},eles),
                    bcw = proxy<space>({},bcw),
                    deformField = zs::SmallString{deformField}] ZS_LAMBDA (int vi) mutable {
                using T = typename RM_CVREF_T(verts)::value_type;
                const auto& ei = bcw.pack<1>("inds",vi).reinterpret_bits<int>()[0];
                if(ei < 0)
                    return;
                everts.template tuple<3>(outAttr,vi) = eles.template pack<3,3>(deformField,ei) * everts.template pack<3>(inAttr,vi);
                // if(vi == 114754){
                //     auto dx = everts.template pack<3>(outAttr,vi);
                //     auto dX = everts.template pack<3>(inAttr,vi);
                //     auto F = eles.template pack<3,3>(deformField,ei);
                //     printf("F : %f %f %f\n%f %f %f\n%f %f %f\n",
                //         (float)F(0,0),(float)F(0,1),(float)F(0,2),
                //         (float)F(1,0),(float)F(1,1),(float)F(1,2),
                //         (float)F(2,0),(float)F(2,1),(float)F(2,2)
                //     );
                //     printf("Fdet : %f\n",(float)zs::determinant(F));
                //     printf("dX : %f %f %f with length %f\n",(float)dX[0],(float)dX[1],(float)dX[2],(float)dX.norm());
                //     printf("dx : %f %f %f with length %f\n",(float)dx[0],(float)dx[1],(float)dx[2],(float)dx.norm());
                // }

        });
        set_output("zssurf",zssurf);
        set_output("zsvolume",zsvolume);
    }
};

ZENDEFNODE(ZSDeformEmbedPrim, {{{"zsvolume"}, {"embed primitive", "zssurf"}},
                            {{"embed primitive", "zssurf"},{"zsvolume"}},
                            {
                                {"string","inAttr","V"},
                                {"string","outAttr","v"},
                                {"string","tag","skin_bw"},
                                {"string","deformField","F"}
                                },
                            {"ZSGeometry"}});

} // namespace zeno