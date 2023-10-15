#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>


#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "kernel/calculate_facet_normal.hpp"
#include "kernel/topology.hpp"
#include "kernel/compute_characteristic_length.hpp"
#include "kernel/calculate_bisector_normal.hpp"
#include "kernel/tiled_vector_ops.hpp"
#include "kernel/calculate_edge_normal.hpp"

#include "../fem/collision_energy/evaluate_collision.hpp"

#include "kernel/topology.hpp"
#include "kernel/intersection.hpp"

#include <iostream>


#define COLLISION_VIS_DEBUG

#define MAX_FP_COLLISION_PAIRS 6

namespace zeno {

    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using mat3 = zs::vec<T,3,3>;
    using mat4 = zs::vec<T,4,4>;
    
    template<typename VTILEVEC> 
    constexpr vec3 eval_center(const VTILEVEC& verts,const zs::vec<int,4>& tet) {
        auto res = vec3::zeros();
        for(int i = 0;i != 4;++i)
            res += verts.template pack<3>("x",tet[i]) / (T)4.0;
        return res;
    } 

    template<typename VTILEVEC> 
    constexpr vec3 eval_center(const VTILEVEC& verts,const zs::vec<int,3>& tri) {
        auto res = vec3::zeros();
        for(int i = 0;i < 3;++i)
            res += verts.template pack<3>("x",tri[i]) / (T)3.0;
        return res;
    } 
    template<typename VTILEVEC> 
    constexpr vec3 eval_center(const VTILEVEC& verts,const zs::vec<int,2>& line) {
        auto res = vec3::zeros();
        for(int i = 0;i < 2;++i)
            res += verts.template pack<3>("x",line[i]) / (T)2.0;
        return res;
    } 

    // struct VisualizeTopology : INode {

    //     virtual void apply() override {
    //         using namespace zs;
            
    //         auto zsparticles = get_input<ZenoParticles>("ZSParticles");
    //         if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag)){
    //             throw std::runtime_error("the input zsparticles has no surface tris");
    //         }
    //         if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag)) {
    //             throw std::runtime_error("the input zsparticles has no surface lines");
    //         }
    //         if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) {
    //             throw std::runtime_error("the input zsparticles has no surface points");
    //         }

    //         const auto& tets = zsparticles->getQuadraturePoints();
    //         auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
    //         auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
    //         auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

    //         if(!tris.hasProperty("ff_inds") || tris.getPropertySize("ff_inds") != 3){
    //             throw std::runtime_error("no valid ff_inds detected in tris");
    //         }            

    //         if(!tris.hasProperty("fe_inds") || tris.getPropertySize("fe_inds") != 3) {
    //             throw std::runtime_error("no valid fe_inds detected in tris");
    //         }

    //         if(!lines.hasProperty("fe_inds") || lines.getPropertySize("fe_inds") != 2) {
    //             throw std::runtime_error("no valid fe_inds detected in lines");
    //         }

    //         const auto& verts = zsparticles->getParticles();
    //         std::vector<zs::PropertyTag> tags{{"x",3}};

    //         int nm_tris = tris.size();
    //         int nm_lines = lines.size();

    //         // output ff topo first
    //         auto ff_topo = typename ZenoParticles::particles_t(tags,nm_tris * 4,zs::memsrc_e::device,0);
    //         auto fe_topo = typename ZenoParticles::particles_t(tags,nm_tris * 4,zs::memsrc_e::device,0);
    //         auto fp_topo = typename ZenoParticles::particles_t(tags,nm_tris * 4,zs::memsrc_e::device,0);
    //         // auto ep_topo = typename ZenoParticles::particles_t(tags,nm_lines * 2,zs::memsrc_e::device,0);
    //         auto ft_topo = typename ZenoParticles::particles_t(tags,nm_tris * 2,zs::memsrc_e::device,0);

    //         // transfer the data from gpu to cpu
    //         constexpr auto cuda_space = execspace_e::cuda;
    //         auto cudaPol = cuda_exec();  
    //         cudaPol(zs::range(nm_tris),
    //             [ff_topo = proxy<cuda_space>({},ff_topo),
    //                 fe_topo = proxy<cuda_space>({},fe_topo),
    //                 fp_topo = proxy<cuda_space>({},fp_topo),
    //                 ft_topo = proxy<cuda_space>({},ft_topo),
    //                 tets = proxy<cuda_space>({},tets),
    //                 tris = proxy<cuda_space>({},tris),
    //                 lines = proxy<cuda_space>({},lines),
    //                 points = proxy<cuda_space>({},points),
    //                 verts = proxy<cuda_space>({},verts)] ZS_LAMBDA(int ti) mutable {
    //                     auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
    //                     auto tet_id = reinterpret_bits<int>(tris("ft_inds",ti));
    //                     auto tet = tets.template pack<4>("inds",tet_id).reinterpret_bits(int_c);
    //                     auto ff_inds = tris.template pack<3>("ff_inds",ti).reinterpret_bits(int_c);
    //                     auto fe_inds = tris.template pack<3>("fe_inds",ti).reinterpret_bits(int_c);
    //                     auto fp_inds = tris.template pack<3>("fp_inds",ti).reinterpret_bits(int_c);
                        
    //                     auto center = eval_center(verts,tri);
    //                     ff_topo.template tuple<3>("x",ti * 4 + 0) = center;
    //                     fe_topo.template tuple<3>("x",ti * 4 + 0) = center;
    //                     fp_topo.template tuple<3>("x",ti * 4 + 0) = center;
    //                     auto tcenter = eval_center(verts,tet);

    //                     ft_topo.template tuple<3>("x",ti * 2 + 0) = center;
    //                     ft_topo.template tuple<3>("x",ti * 2 + 1) = tcenter;

    //                     for(int i = 0;i != 3;++i) {
    //                         auto nti = ff_inds[i];
    //                         auto ntri = tris.template pack<3>("inds",nti).reinterpret_bits(int_c);
    //                         auto ncenter = eval_center(verts,ntri);
    //                         ff_topo.template tuple<3>("x",ti * 4 + i + 1) = ncenter;

    //                         auto nei = fe_inds[i];
    //                         auto nedge = lines.template pack<2>("inds",nei).reinterpret_bits(int_c);
    //                         ncenter = eval_center(verts,nedge);
    //                         // printf("edge[%d] : %d %d\n",nei,nedge[0],nedge[1]);
    //                         fe_topo.template tuple<3>("x",ti * 4 + i + 1) = ncenter;

    //                         auto pidx = reinterpret_bits<int>(points("inds",fp_inds[i]));
    //                         fp_topo.template tuple<3>("x",ti * 4 + i + 1) = verts.template pack<3>("x",pidx);
    //                     }

    //         });   

    //         ff_topo = ff_topo.clone({zs::memsrc_e::host});
    //         fe_topo = fe_topo.clone({zs::memsrc_e::host});
    //         fp_topo = fp_topo.clone({zs::memsrc_e::host});
    //         // ep_topo = ep_topo.clone({zs::memsrc_e::host});
    //         ft_topo = ft_topo.clone({zs::memsrc_e::host});

    //         int ff_size = ff_topo.size();
    //         int fe_size = fe_topo.size();
    //         int fp_size = fp_topo.size();
    //         // int ep_size = ep_topo.size();
    //         int ft_size = ft_topo.size();

    //         constexpr auto omp_space = execspace_e::openmp;
    //         auto ompPol = omp_exec();

    //         auto ff_prim = std::make_shared<zeno::PrimitiveObject>();
    //         auto fe_prim = std::make_shared<zeno::PrimitiveObject>();
    //         auto fp_prim = std::make_shared<zeno::PrimitiveObject>();
    //         // auto ep_prim = std::make_shared<zeno::PrimitiveObject>();
    //         auto ft_prim = std::make_shared<zeno::PrimitiveObject>();

    //         auto& ff_verts = ff_prim->verts;
    //         auto& ff_lines = ff_prim->lines;

    //         auto& fe_verts = fe_prim->verts;
    //         auto& fe_lines = fe_prim->lines;

    //         auto& fp_verts = fp_prim->verts;
    //         auto& fp_lines = fp_prim->lines;

    //         // auto& ep_verts = ep_prim->verts;
    //         // auto& ep_lines = ep_prim->lines;

    //         auto& ft_verts = ft_prim->verts;
    //         auto& ft_lines = ft_prim->lines;

    //         int ff_pair_count = nm_tris * 3;
    //         int fe_pair_count = nm_tris * 3;
    //         int fp_pair_count = nm_tris * 3;
    //         // int ep_pair_count = nm_lines * 1;
    //         int ft_pair_count = nm_tris;

    //         ff_verts.resize(ff_size);
    //         ff_lines.resize(ff_pair_count);
    //         fe_verts.resize(fe_size);
    //         fe_lines.resize(fe_pair_count);
    //         fp_verts.resize(fp_size);
    //         fp_lines.resize(fp_pair_count);
    //         // ep_verts.resize(ep_size);
    //         // ep_lines.resize(ep_pair_count);
    //         ft_verts.resize(ft_size);
    //         ft_lines.resize(ft_pair_count);

    //         ompPol(zs::range(nm_tris),
    //             [&ft_verts,&ft_lines,ft_topo = proxy<omp_space>({},ft_topo)] (int fi) mutable {
    //                 ft_verts[fi * 2 + 0] = ft_topo.template pack<3>("x",fi * 2 + 0).to_array();
    //                 ft_verts[fi * 2 + 1] = ft_topo.template pack<3>("x",fi * 2 + 1).to_array();
    //                 // ft_verts[fi * 2 + 1] = zeno::vec3f(0.0,0.0,0.0);
    //                 ft_lines[fi] = zeno::vec2i(fi * 2 + 0,fi * 2 + 1);
    //         });

    //         ompPol(zs::range(nm_tris),
    //             [&ff_verts,&ff_lines,ff_topo = proxy<omp_space>({},ff_topo)] (int fi) mutable {
    //                 auto v = ff_topo.template pack<3>("x",fi * 4 + 0);
    //                 ff_verts[fi * 4 + 0] = zeno::vec3f(v[0],v[1],v[2]);
    //                 for(int i = 0;i != 3;++i){
    //                     auto v = ff_topo.template pack<3>("x",fi * 4 + i + 1);
    //                     ff_verts[fi * 4 + i + 1] = zeno::vec3f(v[0],v[1],v[2]);
    //                     ff_lines[fi * 3 + i] = zeno::vec2i(fi * 4 + 0,fi * 4 + i + 1);
    //                 }
    //         });

    //         ompPol(zs::range(nm_tris),
    //             [&fe_verts,&fe_lines,fe_topo = proxy<omp_space>({},fe_topo)] (int fi) mutable {
    //                 auto v = fe_topo.template pack<3>("x",fi * 4 + 0);
    //                 fe_verts[fi * 4 + 0] = zeno::vec3f(v[0],v[1],v[2]);
    //                 for(int i = 0;i != 3;++i){
    //                     auto v = fe_topo.template pack<3>("x",fi * 4 + i + 1);
    //                     fe_verts[fi * 4 + i + 1] = zeno::vec3f(v[0],v[1],v[2]);
    //                     fe_lines[fi * 3 + i] = zeno::vec2i(fi * 4 + 0,fi * 4 + i + 1);
    //                 }
    //         });

    //         ompPol(zs::range(nm_tris),
    //             [&fp_verts,&fp_lines,fp_topo = proxy<omp_space>({},fp_topo)] (int fi) mutable {
    //                 auto v = fp_topo.template pack<3>("x",fi * 4 + 0);
    //                 fp_verts[fi * 4 + 0] = zeno::vec3f(v[0],v[1],v[2]);
    //                 for(int i = 0;i != 3;++i){
    //                     auto v = fp_topo.template pack<3>("x",fi * 4 + i + 1);
    //                     fp_verts[fi * 4 + i + 1] = zeno::vec3f(v[0],v[1],v[2]);
    //                     fp_lines[fi * 3 + i] = zeno::vec2i(fi * 4 + 0,fi * 4 + i + 1);
    //                 }
    //         });

    //         // ompPol(zs::range(nm_lines),
    //         //     [&ep_verts,&ep_lines,ep_topo = proxy<omp_space>({},ep_topo)] (int li) mutable {
    //         //         for(int i = 0;i != 2;++i)
    //         //             ep_verts[li * 2 + i] = ep_topo.template pack<3>("x",li * 2 + i).to_array();
    //         //         ep_lines[li] = zeno::vec2i(li * 2 + 0,li * 2 + 1);
    //         // });

    //         // for(int i = 0;i < fe_lines.size();++i)
    //         //     std::cout << "fe_line<" << i << "> : \t" << fe_lines[i][0] << "\t" << fe_lines[i][1] << std::endl;
    //         set_output("ft_topo",std::move(ft_prim));
    //         set_output("fp_topo",std::move(fp_prim));
    //         set_output("ff_topo",std::move(ff_prim));
    //         set_output("fe_topo",std::move(fe_prim));
    //         // set_output("ep_topo",std::move(ep_prim));
    //     }
    // };


    // ZENDEFNODE(VisualizeTopology, {{{"ZSParticles"}},
    //                             {{"ft_topo"},{"ff_topo"},{"fe_topo"},{"fp_topo"}/*,{"ep_topo"}*/},
    //                             {},
    //                             {"ZSGeometry"}});


    // struct CopyShape : INode {
    //     virtual void apply() override {
    //         auto prim1 = get_input<zeno::PrimitiveObject>("prim1");
    //         auto prim2 = get_input<zeno::PrimitiveObject>("prim2");
    //         auto& nx = prim1->add_attr<zeno::vec3f>("npos");
    //         for(int i = 0;i != prim1->size();++i)
    //             nx[i] = prim2->verts[i];
    //         set_output("prim1",prim1);
    //     }
    // };
    // ZENDEFNODE(CopyShape, {{{"prim1"},{"prim2"}},
    //                             {{"prim1"}},
    //                             {},
    //                             {"ZSGeometry"}});


    struct VisualizeSurfaceMesh : INode {
        virtual void apply() override {
            using namespace zs;
            auto zsparticles = get_input<ZenoParticles>("ZSParticles");

            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag)){
                throw std::runtime_error("the input zsparticles has no surface tris");
            }
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag)) {
                throw std::runtime_error("the input zsparticles has no surface lines");
            }
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) {
                throw std::runtime_error("the input zsparticles has no surface points");
            }
            const auto &tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();
            const auto& points  = (*zsparticles)[ZenoParticles::s_surfVertTag];
            const auto& verts = zsparticles->getParticles();

            // if(!tris.hasProperty("fp_inds") || tris.getPropertySize("fp_inds") != 3) {
            //     throw std::runtime_error("call ZSInitSurfaceTopology first before VisualizeSurfaceMesh");
            // }

            // transfer the data from gpu to cpu
            constexpr auto cuda_space = execspace_e::cuda;
            auto cudaPol = cuda_exec(); 

            bcht<int,int,true,universal_hash<int>,32> ptab{points.get_allocator(),points.size()};
            Vector<int> spi{points.get_allocator(),points.size()};
            cudaPol(range(points.size()),
            [ptab = proxy<cuda_space>(ptab),points = proxy<cuda_space>({},points),spi = proxy<cuda_space>(spi)] ZS_LAMBDA(int pi) mutable {
                auto pidx = reinterpret_bits<int>(points("inds",pi));
                if(int no = ptab.insert(pidx);no >= 0)
                    spi[no] = pi;
                else{
                    printf("same point [%d] has been inserted twice\n",pidx);
                }
            });

            auto nm_points = points.size();
            auto nm_tris = tris.size();

            auto xtag = get_param<std::string>("xtag");



            // auto surf_verts_buffer = typename ZenoParticles::particles_t({{"x",3}},points.size(),zs::memsrc_e::device,0);
            zs::Vector<zs::vec<float,3>> surf_verts_buffer{points.get_allocator(),points.size()};
            zs::Vector<zs::vec<int,3>> surf_tris_buffer{tris.get_allocator(),tris.size()};
            // auto surf_tris_buffer  = typename ZenoParticles::particles_t({{"inds",3}},tris.size(),zs::memsrc_e::device,0);
            // copy the verts' pos data to buffer
            cudaPol(zs::range(points.size()),
                [verts = proxy<cuda_space>({},verts),xtag = zs::SmallString(xtag),
                        points = proxy<cuda_space>({},points),surf_verts_buffer = proxy<cuda_space>(surf_verts_buffer)] ZS_LAMBDA(int pi) mutable {
                    auto v_idx = reinterpret_bits<int>(points("inds",pi));
                    surf_verts_buffer[pi] = verts.template pack<3>(xtag,v_idx);
            }); 

            // copy the tris topo to buffer
            // TILEVEC_OPS::copy<3>(cudaPol,tris,"fp_inds",surf_tris_buffer,"inds");
            cudaPol(zs::range(tris.size()),[
                surf_tris_buffer = proxy<cuda_space>(surf_tris_buffer),
                tris = proxy<cuda_space>({},tris),spi = proxy<cuda_space>(spi),ptab = proxy<cuda_space>(ptab)] ZS_LAMBDA(int ti) mutable {
                    auto inds = tris.pack(dim_c<3>,"inds",ti,int_c);
                    zs::vec<int,3> tinds{};
                    for(int i = 0;i != 3;++i) {
                        auto no = ptab.query(inds[i]);
                        tinds[i] = spi[no];
                    }
                    surf_tris_buffer[ti] = tinds;
            });

            surf_verts_buffer = surf_verts_buffer.clone({zs::memsrc_e::host});
            surf_tris_buffer = surf_tris_buffer.clone({zs::memsrc_e::host});


            auto sprim = std::make_shared<zeno::PrimitiveObject>();
            auto& sverts = sprim->verts;
            auto& stris = sprim->tris;

            sverts.resize(nm_points);
            stris.resize(nm_tris);

            auto ompPol = omp_exec();
            constexpr auto omp_space = execspace_e::openmp;

            ompPol(zs::range(sverts.size()),
                [&sverts,surf_verts_buffer = proxy<omp_space>(surf_verts_buffer)] (int vi) mutable {
                    // auto v = surf_verts_buffer.template pack<3>("x",vi);
                    sverts[vi] = surf_verts_buffer[vi].to_array();
            });

            ompPol(zs::range(stris.size()),
                [&stris,surf_tris_buffer = proxy<omp_space>(surf_tris_buffer)] (int ti) mutable {
                    // auto t = surf_tris_buffer[ti];
                    stris[ti] = surf_tris_buffer[ti].to_array();
            });

            set_output("prim",std::move(sprim));
        }
    };

    ZENDEFNODE(VisualizeSurfaceMesh, {{{"ZSParticles"}},
                                {{"prim"}},
                                {
                                    {"string","xtag","x"}
                                },
                                {"ZSGeometry"}});


    struct MarkSelfIntersectionRegion : zeno::INode {

        using T = float;
        using Ti = int;
        using dtiles_t = zs::TileVector<T,32>;
        using tiles_t = typename ZenoParticles::particles_t;
        using bvh_t = zs::LBvh<3,int,T>;
        using bv_t = zs::AABBBox<3, T>;
        using vec3 = zs::vec<T, 3>; 

        virtual void apply() override {
            using namespace zs;
            auto zsparticles = get_input<ZenoParticles>("zsparticles");
            auto& verts = zsparticles->getParticles();
            bool is_tet_volume_mesh = zsparticles->category == ZenoParticles::category_e::tet;
            const auto &tris = is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints(); 

            constexpr auto cuda_space = execspace_e::cuda;
            auto cudaPol = cuda_exec();  

            dtiles_t tri_buffer{tris.get_allocator(),{
                {"inds",3},
                {"nrm",3},
                {"he_inds",1}
            },tris.size()};
            dtiles_t verts_buffer{verts.get_allocator(),{
                {"inds",1},
                {"x",3},
                {"he_inds",1}
            },is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfVertTag].size() : verts.size()};

            TILEVEC_OPS::copy(cudaPol,tris,"he_inds",tri_buffer,"he_inds");
            if(is_tet_volume_mesh) {
                const auto &points = (*zsparticles)[ZenoParticles::s_surfVertTag];
                TILEVEC_OPS::copy(cudaPol,points,"inds",verts_buffer,"inds");
                TILEVEC_OPS::copy(cudaPol,points,"he_inds",verts_buffer,"he_inds");
                topological_sample(cudaPol,points,verts,"x",verts_buffer);
                TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
                reorder_topology(cudaPol,points,tri_buffer);

            }else {
                TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
                TILEVEC_OPS::copy(cudaPol,verts,"x",verts_buffer,"x");
                cudaPol(zs::range(verts.size()),[
                    verts = proxy<cuda_space>({},verts),
                    verts_buffer = proxy<cuda_space>({},verts_buffer)] ZS_LAMBDA(int vi) mutable {
                        verts_buffer("inds",vi) = reinterpret_bits<T>(vi);
                });
            }

            if(!calculate_facet_normal(cudaPol,verts_buffer,"x",tri_buffer,tri_buffer,"nrm")){
                throw std::runtime_error("fail updating facet normal");
            }  

            dtiles_t inst_buffer_info{tris.get_allocator(),{
                {"pair",2},
                {"type",1},
                {"its_edge_mark",6},
                {"int_points",6}
            },tris.size() * 2};

            dtiles_t gia_res{verts_buffer.get_allocator(),{
                {"ring_mask",1},
                {"type_mask",1},
                {"color_mask",1},
                {"is_loop_vertex",1}
            },verts_buffer.size()};

            dtiles_t tris_gia_res{tri_buffer.get_allocator(),{
                {"ring_mask",1},
                {"type_mask",1},
                {"color_mask",1},
            },tri_buffer.size()};

            auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
            // auto nm_insts = do_global_self_intersection_analysis_on_surface_mesh_info(
            //     cudaPol,verts_buffer,"x",tri_buffer,halfedges,inst_buffer_info,gia_res,false);  
            // zs::bht<int,2,int> conn_of_first_ring{halfedges.get_allocator(),halfedges.size()};      
            auto ring_mask_width = do_global_self_intersection_analysis(cudaPol,
                verts_buffer,"x",tri_buffer,halfedges,gia_res,tris_gia_res);    
   

            auto markTag = get_input2<std::string>("markTag");

            if(!verts.hasProperty("markTag")) {
                verts.append_channels(cudaPol,{{markTag,1}});
            }
            TILEVEC_OPS::fill(cudaPol,verts,markTag,(T)0.0);
            cudaPol(zs::range(verts_buffer.size()),[
                gia_res = proxy<cuda_space>({},gia_res),
                verts = proxy<cuda_space>({},verts),
                ring_mask_width = ring_mask_width,
                verts_buffer = proxy<cuda_space>({},verts_buffer),
                markTag = zs::SmallString(markTag)
            ] ZS_LAMBDA(int pi) mutable {
                auto vi = zs::reinterpret_bits<int>(verts_buffer("inds",pi));
                int ring_mask = 0;
                for(int i = 0;i != ring_mask_width;++i) {
                    ring_mask |= zs::reinterpret_bits<int>(gia_res("ring_mask",pi * ring_mask_width + i));
                }
                verts(markTag,vi) = ring_mask == 0 ? (T)0.0 : (T)1.0;
            });
            set_output("zsparticles",zsparticles);
        } 

    };

    ZENDEFNODE(MarkSelfIntersectionRegion, {{{"zsparticles"},{"string","markTag","markTag"}},
                                {{"zsparticles"}},
                                {
                                    
                                },
                                {"ZSGeometry"}});



    struct VisualizeSurfaceNormal : INode {
        virtual void apply() override {
            using namespace zs;
            auto cudaExec = cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;

            auto zsparticles = get_input<ZenoParticles>("ZSParticles");
            auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
                (*zsparticles)[ZenoParticles::s_surfTriTag] : 
                zsparticles->getQuadraturePoints();

            const auto& verts = zsparticles->getParticles();
            if(!tris.hasProperty("nrm"))
                tris.append_channels(cudaExec,{{"nrm",3}});

            if(!calculate_facet_normal(cudaExec,verts,"x",tris,tris,"nrm"))
                throw std::runtime_error("ZSCalNormal::calculate_facet_normal fail"); 

            auto buffer = typename ZenoParticles::particles_t({{"dir",3},{"x",3}},tris.size(),zs::memsrc_e::device,0);

            cudaExec(zs::range(tris.size()),
                [tris = proxy<space>({},tris),
                        buffer = proxy<space>({},buffer),
                        verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
                    auto inds = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                    zs::vec<T,3> tp[3];
                    for(int i = 0;i != 3;++i)
                        tp[i] = verts.template pack<3>("x",inds[i]);
                    auto center = (tp[0] + tp[1] + tp[2]) / (T)3.0;

                    buffer.template tuple<3>("dir",ti) = tris.template pack<3>("nrm",ti);
                    buffer.template tuple<3>("x",ti) = center;
            });                        

            buffer = buffer.clone({zs::memsrc_e::host});
            auto prim = std::make_shared<zeno::PrimitiveObject>();
            auto& pverts = prim->verts;
            pverts.resize(buffer.size() * 2);
            auto& lines = prim->lines;
            lines.resize(buffer.size());

            auto ompExec = omp_exec();
            constexpr auto ompSpace = zs::execspace_e::openmp;

            auto extrude_offset = get_param<float>("offset");

            ompExec(zs::range(buffer.size()),
                [buffer = proxy<ompSpace>({},buffer),&pverts,&lines,extrude_offset] (int ti) mutable {
                    auto xs = buffer.template pack<3>("x",ti);
                    auto dir = buffer.template pack<3>("dir",ti);
                    auto xe = xs + extrude_offset * dir;
                    pverts[ti * 2 + 0] = zeno::vec3f(xs[0],xs[1],xs[2]);
                    pverts[ti * 2 + 1] = zeno::vec3f(xe[0],xe[1],xe[2]);

                    lines[ti] = zeno::vec2i(ti * 2 + 0,ti * 2 + 1);
            });

            set_output("prim",std::move(prim));
        }
    };

    ZENDEFNODE(VisualizeSurfaceNormal, {{{"ZSParticles"}},
                                {{"prim"}},
                                {{"float","offset","1"}},
                                {"ZSGeometry"}});


    struct VisualizeSurfaceEdgeNormal : INode {
        virtual void apply() override {
            using namespace zs;

            auto zsparticles = get_input<ZenoParticles>("ZSParticles");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag)){
                throw std::runtime_error("the input zsparticles has no surface tris");
            }
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag)) {
                throw std::runtime_error("the input zsparticles has no surface lines");
            }   

            auto& tris      = (*zsparticles)[ZenoParticles::s_surfTriTag];
            if(!tris.hasProperty("ff_inds") || !tris.hasProperty("fe_inds"))
                throw std::runtime_error("please call ZSInitTopoConnect first before this node");           
            auto& lines     = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            if(!lines.hasProperty("fe_inds"))
                throw std::runtime_error("please call ZSInitTopoConnect first before this node");             

            const auto& verts = zsparticles->getParticles();
            auto cudaExec = cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;

            if(!tris.hasProperty("nrm"))
                tris.append_channels(cudaExec,{{"nrm",3}});

            // std::cout << "CALCULATE SURFACE NORMAL" << std::endl;

            if(!calculate_facet_normal(cudaExec,verts,"x",tris,tris,"nrm"))
                throw std::runtime_error("VisualizeSurfaceEdgeNormal::calculate_facet_normal fail"); 


            auto buffer = typename ZenoParticles::particles_t({{"nrm",3},{"x",3}},lines.size(),zs::memsrc_e::device,0);  
            if(!calculate_edge_normal_from_facet_normal(cudaExec,tris,"nrm",buffer,"nrm",lines))
                throw std::runtime_error("VisualizeSurfaceEdgeNormal::calculate_edge_normal_from_facet_normal fail");


            cudaExec(zs::range(lines.size()),[
                    buffer = proxy<space>({},buffer),
                    lines = proxy<space>({},lines),
                    tris = proxy<space>({},tris),
                    verts = proxy<space>({},verts)] ZS_LAMBDA(int ei) mutable {
                        auto linds = lines.template pack<2>("inds",ei).reinterpret_bits(int_c);
                        auto fe_inds = lines.template pack<2>("fe_inds",ei).reinterpret_bits(int_c);

                        auto n0 = tris.template pack<3>("nrm",fe_inds[0]);
                        auto n1 = tris.template pack<3>("nrm",fe_inds[1]);

                        auto v0 = verts.template pack<3>("x",linds[0]);
                        auto v1 = verts.template pack<3>("x",linds[1]);

                        // buffer.template tuple<3>("nrm",ei) = (n0 + n1).normalized();
                        // buffer.template tuple<3>("nrm",ei) = lines.template pack<3>("nrm",ei);
                        buffer.template tuple<3>("x",ei) = (v0 + v1) / (T)2.0;
            }); 

            buffer = buffer.clone({zs::memsrc_e::host});

            auto prim = std::make_shared<zeno::PrimitiveObject>();
            auto& pverts = prim->verts;
            auto& plines = prim->lines;
            pverts.resize(buffer.size() * 2);
            plines.resize(buffer.size());

            auto ompExec = omp_exec();
            constexpr auto omp_space = execspace_e::openmp;

            auto offset = get_param<float>("offset");

            ompExec(zs::range(buffer.size()),
                [buffer = proxy<omp_space>({},buffer),&pverts,&plines,offset] (int li) mutable {
                    auto ps = buffer.template pack<3>("x",li);
                    auto dp = buffer.template pack<3>("nrm",li);
                    auto pe = ps + dp * offset;
                    pverts[li * 2 + 0] = zeno::vec3f(ps[0],ps[1],ps[2]);
                    pverts[li * 2 + 1] = zeno::vec3f(pe[0],pe[1],pe[2]);

                    plines[li] = zeno::vec2i(li * 2 + 0,li * 2 + 1);
            });

            set_output("prim",std::move(prim));
        }
    };

    ZENDEFNODE(VisualizeSurfaceEdgeNormal, {{{"ZSParticles"}},
                                {{"prim"}},
                                {{"float","offset","1"}},
                                {"ZSGeometry"}});

    struct ZSCalSurfaceCollisionCell : INode {
        virtual void apply() override {
            using namespace zs;

            auto zsparticles = get_input<ZenoParticles>("ZSParticles");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag)){
                throw std::runtime_error("the input zsparticles has no surface tris");
                // auto& tris = (*particles)[ZenoParticles::s_surfTriTag];
                // tris = typename ZenoParticles::particles_t({{"inds",3}});
            }
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag)) {
                throw std::runtime_error("the input zsparticles has no surface lines");
            }

            auto& tris      = (*zsparticles)[ZenoParticles::s_surfTriTag];
            if(!tris.hasProperty("ff_inds") || !tris.hasProperty("fe_inds"))
                throw std::runtime_error("please call ZSInitTopoConnect first before this node");           
            auto& lines     = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            if(!lines.hasProperty("fe_inds"))
                throw std::runtime_error("please call ZSInitTopoConnect first before this node"); 

            const auto& verts = zsparticles->getParticles();
            auto cudaExec = cuda_exec();
            // constexpr auto space = zs::execspace_e::cuda;

            if(!tris.hasProperty("nrm"))
                tris.append_channels(cudaExec,{{"nrm",3}});

            // std::cout << "CALCULATE SURFACE NORMAL" << std::endl;

            if(!calculate_facet_normal(cudaExec,verts,"x",tris,tris,"nrm"))
                throw std::runtime_error("ZSCalNormal::calculate_facet_normal fail"); 
            // std::cout << "FINISH CALCULATE SURFACE NORMAL" << std::endl;

            auto ceNrmTag = get_param<std::string>("ceNrmTag");
            if(!lines.hasProperty(ceNrmTag))
                lines.append_channels(cudaExec,{{ceNrmTag,3}});
            
            // evalute the normal of edge plane
            // cudaExec(range(lines.size()),
            //     [verts = proxy<space>({},verts),
            //         tris = proxy<space>({},tris),
            //         lines = proxy<space>({},lines),
            //         ceNrmTag = zs::SmallString(ceNrmTag)] ZS_LAMBDA(int ei) mutable {
            //             auto e_inds = lines.template pack<2>("inds",ei).template reinterpret_bits<int>();
            //             auto fe_inds = lines.template pack<2>("fe_inds",ei).template reinterpret_bits<int>();
            //             auto n0 = tris.template pack<3>("nrm",fe_inds[0]);
            //             auto n1 = tris.template pack<3>("nrm",fe_inds[1]);

            //             auto ne = (n0 + n1).normalized();
            //             auto e0 = verts.template pack<3>("x",e_inds[0]);
            //             auto e1 = verts.template pack<3>("x",e_inds[1]);
            //             auto e10 = e1 - e0;

            //             lines.template tuple<3>(ceNrmTag,ei) = e10.cross(ne).normalized();
            // });

            COLLISION_UTILS::calculate_cell_bisector_normal(cudaExec,
                verts,"x",
                lines,
                tris,
                tris,"nrm",
                lines,ceNrmTag);


            set_output("ZSParticles",zsparticles);
        }

    };

    ZENDEFNODE(ZSCalSurfaceCollisionCell, {{{"ZSParticles"}},
                                {{"ZSParticles"}},
                                {{"string","ceNrmTag","nrm"}},
                                {"ZSGeometry"}});




    struct VisualizeCollisionCell : INode {
        virtual void apply() override {
            using namespace zs;

            auto zsparticles = get_input<ZenoParticles>("ZSParticles");
            auto ceNrmTag = get_param<std::string>("ceNrmTag");
            // auto out_offset = get_input2<float>("out_offset");
            // auto in_offset = get_input2<float>("in_offset");
            auto collisionEps = get_input2<float>("collisionEps");
            auto nrm_offset = get_input2<float>("nrm_offset");

            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
                throw std::runtime_error("the input zsparticles has no surface tris");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
                throw std::runtime_error("the input zsparticles has no surface lines");

            auto& tris      = (*zsparticles)[ZenoParticles::s_surfTriTag];
            if(!tris.hasProperty("ff_inds") || !tris.hasProperty("fe_inds"))
                throw std::runtime_error("please call ZSCalSurfaceCollisionCell first before this node");           
            auto& lines     = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            if(!lines.hasProperty("fe_inds") || !lines.hasProperty(ceNrmTag))
                throw std::runtime_error("please call ZSCalSurfaceCollisionCell first before this node"); 
            auto& verts = zsparticles->getParticles();
            // cell data per facet
            std::vector<zs::PropertyTag> tags{{"x",9},{"dir",9},{"nrm",9},{"center",3}};
            auto cell_buffer = typename ZenoParticles::particles_t(tags,tris.size(),zs::memsrc_e::device,0);
            // auto cell_buffer = typename ZenoParticles::particles_t(tags,1,zs::memsrc_e::device,0);
            // transfer the data from gpu to cpu
            constexpr auto cuda_space = execspace_e::cuda;
            auto cudaPol = cuda_exec();      

            cudaPol(zs::range(cell_buffer.size()),
                [cell_buffer = proxy<cuda_space>({},cell_buffer),
                    verts = proxy<cuda_space>({},verts),
                    lines = proxy<cuda_space>({},lines),
                    tris = proxy<cuda_space>({},tris),
                    ceNrmTag = zs::SmallString(ceNrmTag)] ZS_LAMBDA(int ci) mutable {
                auto inds       = tris.template pack<3>("inds",ci).template reinterpret_bits<int>();
                auto fe_inds    = tris.template pack<3>("fe_inds",ci).template reinterpret_bits<int>();

                auto nrm = tris.template pack<3>("nrm",ci);

                #ifdef COLLISION_VIS_DEBUG
                
                    zs::vec<T,3> vs[3];
                    for(int i = 0;i != 3;++i)
                        vs[i] = verts.template pack<3>("x",inds[i]);
                    auto vc = (vs[0] + vs[1] + vs[2]) / (T)3.0;

                    zs::vec<T,3> ec[3];
                    for(int i = 0;i != 3;++i)
                        ec[i] = (vs[i] + vs[(i+1)%3])/2.0;

                    // make sure all the bisector facet orient in-ward
                    // for(int i = 0;i != 3;++i){
                    //     auto ec_vc = vc - ec[i];
                    //     auto e1 = fe_inds[i];
                    //     auto n1 = lines.template pack<3>(ceNrmTag,e1);
                    //     if(is_edge_edge_match(lines.template pack<2>("inds",e1).template reinterpret_bits<int>(),zs::vec<int,2>{inds[i],inds[((i + 1) % 3)]}) == 1)
                    //         n1 = (T)-1 * n1;
                    //     auto check_dir = n1.dot(ec_vc);
                    //     if(check_dir < 0) {
                    //         printf("invalid check dir %f %d %d\n",(float)check_dir,ci,i);
                    //     }
                    // }

                    // auto cell_center = vec3::zeros();
                    // cell_center = (vs[0] + vs[1] + vs[2])/(T)3.0;
                    // T check_dist{};
                    // auto check_intersect = COLLISION_UTILS::is_inside_the_cell(verts,"x",
                    //     lines,tris,
                    //     tris,"nrm",
                    //     lines,ceNrmTag,
                    //     ci,cell_center,in_offset,out_offset);
                    // if(check_intersect == 1)
                    //     printf("invalid cell intersection check offset and inset : %d %f %f %f\n",ci,(float)check_dist,(float)out_offset,(float)in_offset);
                    // if(check_intersect == 2)
                    //     printf("invalid cell intersection check bisector : %d\n",ci);


                #endif

                cell_buffer.template tuple<3>("center",ci) = vec3::zeros();
                for(int i = 0;i < 3;++i){
                    auto vert = verts.template pack<3>("x",inds[i]);
                    cell_buffer.template tuple<3>("center",ci) = cell_buffer.template pack<3>("center",ci) + vert/(T)3.0;
                    for(int j = 0;j < 3;++j) {
                        cell_buffer("x",i * 3 + j,ci) = vert[j];
                    }
                    
#if 0
                    auto e0 = fe_inds[(i + 3 -1) % 3];
                    auto e1 = fe_inds[i];

                    auto n0 = lines.template pack<3>(ceNrmTag,e0);
                    auto n1 = lines.template pack<3>(ceNrmTag,e1);

                    for(int j = 0;j != 3;++j)
                        cell_buffer("nrm",i*3 + j,ci) = n1[j];

                    if(is_edge_edge_match(lines.template pack<2>("inds",e0).template reinterpret_bits<int>(),zs::vec<int,2>{inds[((i + 3 - 1) % 3)],inds[i]}) == 1)
                        n0 =  (T)-1 * n0;
                    if(is_edge_edge_match(lines.template pack<2>("inds",e1).template reinterpret_bits<int>(),zs::vec<int,2>{inds[i],inds[((i + 1) % 3)]}) == 1)
                        n1 = (T)-1 * n1;
#else

                    auto n0 = COLLISION_UTILS::get_bisector_orient(lines,tris,
                        lines,ceNrmTag,
                        ci,(i + 3 - 1) % 3);
                    auto n1 = COLLISION_UTILS::get_bisector_orient(lines,tris,
                        lines,ceNrmTag,ci,i);

                    for(int j = 0;j != 3;++j)
                        cell_buffer("nrm",i*3 + j,ci) = n1[j];

#endif
                    auto dir = n1.cross(n0).normalized();

                    // do some checking
                    // #ifdef COLLISION_VIS_DEBUG

                    // #endif


                    // auto orient = dir.dot(nrm);
                    // if(orient > 0) {
                    //     printf("invalid normal dir %f on %d\n",(float)orient,ci);
                    // }
                    // printf("dir = %f %f %f\n",(float)dir[0],(float)dir[1],(float)dir[2]);
                    // printf("n0 = %f %f %f\n",(float)n0[0],(float)n0[1],(float)n0[2]);
                    // printf("n1 = %f %f %f\n",(float)n1[0],(float)n1[1],(float)n1[2]);
                    for(int j = 0;j < 3;++j){
                        cell_buffer("dir",i * 3 + j,ci) = dir[j];
                        // cell_buffer("dir",i * 3 + j,ci) = nrm[j];
                    }
                    
                }
            });  

            cell_buffer = cell_buffer.clone({zs::memsrc_e::host});   
            constexpr auto omp_space = execspace_e::openmp;
            auto ompPol = omp_exec();            

            auto cell = std::make_shared<zeno::PrimitiveObject>();

            auto& cell_verts = cell->verts;
            auto& cell_lines = cell->lines;
            auto& cell_tris = cell->tris;
            cell_verts.resize(cell_buffer.size() * 6);
            cell_lines.resize(cell_buffer.size() * 9);
            cell_tris.resize(cell_buffer.size() * 6);


            auto offset_ratio = get_input2<float>("offset_ratio");

            ompPol(zs::range(cell_buffer.size()),
                [cell_buffer = proxy<omp_space>({},cell_buffer),
                    &cell_verts,&cell_lines,&cell_tris,collisionEps = collisionEps,offset_ratio = offset_ratio] (int ci) mutable {

                auto vs_ = cell_buffer.template pack<9>("x",ci);
                auto ds_ = cell_buffer.template pack<9>("dir",ci);

                auto center = cell_buffer.template pack<3>("center",ci);

                for(int i = 0;i < 3;++i) {
                    auto p = vec3{vs_[i*3 + 0],vs_[i*3 + 1],vs_[i*3 + 2]};
                    auto dp = vec3{ds_[i*3 + 0],ds_[i*3 + 1],ds_[i*3 + 2]};

                    auto p0 = p - dp * collisionEps;
                    auto p1 = p + dp * collisionEps;

                    auto dp0 = p0 - center;
                    auto dp1 = p1 - center;

                    dp0 *= offset_ratio;
                    dp1 *= offset_ratio;

                    p0 = dp0 + center;
                    p1 = dp1 + center;

                    // printf("ci = %d \t dp = %f %f %f\n",ci,(float)dp[0],(float)dp[1],(float)dp[2]);

                    cell_verts[ci * 6 + i * 2 + 0] = zeno::vec3f{p0[0],p0[1],p0[2]};
                    cell_verts[ci * 6 + i * 2 + 1] = zeno::vec3f{p1[0],p1[1],p1[2]};

                    cell_lines[ci * 9 + 0 + i] = zeno::vec2i{ci * 6 + i * 2 + 0,ci * 6 + i * 2 + 1};
                }

                for(int i = 0;i < 3;++i) {
                    cell_lines[ci * 9 + 3 + i] = zeno::vec2i{ci * 6 + i * 2 + 0,ci * 6 + ((i+1)%3) * 2 + 0};
                    cell_lines[ci * 9 + 6 + i] = zeno::vec2i{ci * 6 + i * 2 + 1,ci * 6 + ((i+1)%3) * 2 + 1}; 

                    cell_tris[ci * 6 + i * 2 + 0] = zeno::vec3i{ci * 6 + i * 2 + 0,ci * 6 + i* 2 + 1,ci * 6 + ((i+1)%3) * 2 + 0};
                    cell_tris[ci * 6 + i * 2 + 1] = zeno::vec3i{ci * 6 + i * 2 + 1,ci * 6 + ((i+1)%3) * 2 + 1,ci * 6 + ((i+1)%3) * 2 + 0};
                }

            });
            cell_lines.resize(0);

            auto tcell = std::make_shared<zeno::PrimitiveObject>();
            // tcell->resize(cell_buffer.size() * 6);
            auto& tcell_verts = tcell->verts;
            tcell_verts.resize(cell_buffer.size() * 6);
            auto& tcell_lines = tcell->lines;
            tcell_lines.resize(cell_buffer.size() * 3);
            ompPol(zs::range(cell_buffer.size()),
                [cell_buffer = proxy<omp_space>({},cell_buffer),
                    &tcell_verts,&tcell_lines,&collisionEps,&offset_ratio,&nrm_offset] (int ci) mutable {

                auto vs_ = cell_buffer.template pack<9>("x",ci);
                auto ds_ = cell_buffer.template pack<9>("dir",ci);


                // printf("vs[%d] : %f %f %f %f %f %f %f %f %f\n",ci,
                //     (float)vs_[0],(float)vs_[1],(float)vs_[2],
                //     (float)vs_[3],(float)vs_[4],(float)vs_[5],
                //     (float)vs_[6],(float)vs_[7],(float)vs_[8]
                // );

                // printf("ds[%d] : %f %f %f %f %f %f %f %f %f\n",ci,
                //     (float)ds_[0],(float)ds_[1],(float)ds_[2],
                //     (float)ds_[3],(float)ds_[4],(float)ds_[5],
                //     (float)ds_[6],(float)ds_[7],(float)ds_[8]
                // );

                auto center = cell_buffer.template pack<3>("center",ci);

                for(int i = 0;i < 3;++i) {
                    auto p = vec3{vs_[i*3 + 0],vs_[i*3 + 1],vs_[i*3 + 2]};
                    auto dp = vec3{ds_[i*3 + 0],ds_[i*3 + 1],ds_[i*3 + 2]};

                    auto p0 = p - dp * collisionEps;
                    auto p1 = p + dp * collisionEps;

                    auto dp0 = p0 - center;
                    auto dp1 = p1 - center;

                    dp0 *= offset_ratio;
                    dp1 *= offset_ratio;

                    p0 = dp0 + center;
                    p1 = dp1 + center;

                    tcell_verts[ci * 6 + i * 2 + 0] = zeno::vec3f{p0[0],p0[1],p0[2]};
                    tcell_verts[ci * 6 + i * 2 + 1] = zeno::vec3f{p1[0],p1[1],p1[2]};

                    tcell_lines[ci * 3 + i] = zeno::vec2i{ci * 6 + i * 2 + 0,ci * 6 + i * 2 + 1};
                }
            });


            auto ncell = std::make_shared<zeno::PrimitiveObject>();
            auto& ncell_verts = ncell->verts;
            auto& ncell_lines = ncell->lines;
            ncell_verts.resize(cell_buffer.size() * 6);
            ncell_lines.resize(cell_buffer.size() * 3);
            ompPol(zs::range(cell_buffer.size()),
                [cell_buffer = proxy<omp_space>({},cell_buffer),
                    &ncell_verts,&ncell_lines,&offset_ratio,&nrm_offset] (int ci) mutable {    
                auto vs_ = cell_buffer.template pack<9>("x",ci);
                auto nrm_ = cell_buffer.template pack<9>("nrm",ci);

                auto center = cell_buffer.template pack<3>("center",ci);
                for(int i = 0;i != 3;++i)   {
                    auto edge_center = vec3::zeros();
                    for(int j = 0;j != 3;++j)
                        edge_center[j] = (vs_[i * 3 + j] + vs_[((i + 1) % 3) * 3 + j])/(T)2.0;
                    auto nrm = vec3{nrm_[i*3 + 0],nrm_[i*3 + 1],nrm_[i*3 + 2]};
                    auto dp = edge_center - center;
                    dp *= offset_ratio;
                    edge_center = dp + center;

                    auto p0 = edge_center;
                    auto p1 = edge_center + nrm * nrm_offset;

                    ncell_verts[ci * 6 + i * 2 + 0] = zeno::vec3f{p0[0],p0[1],p0[2]};
                    ncell_verts[ci * 6 + i * 2 + 1] = zeno::vec3f{p1[0],p1[1],p1[2]};

                    ncell_lines[ci * 3 + i] = zeno::vec2i{ci * 6 + i * 2 + 0,ci * 6 + i * 2 + 1};

                }
            });



            set_output("collision_cell",std::move(cell));
            set_output("ccell_tangent",std::move(tcell));
            set_output("ccell_normal",std::move(ncell));

        }
    };

    ZENDEFNODE(VisualizeCollisionCell, {{{"ZSParticles"},{"float","collisionEps","0.01"},{"float","nrm_offset","0.1"},{"float","offset_ratio","0.8"}},
                                {{"collision_cell"},{"ccell_tangent"},{"ccell_normal"}},
                                {{"string","ceNrmTag","nrm"}},
                                {"ZSGeometry"}});



//     struct VisualizeFacetPointIntersection : zeno::INode {
//         using T = float;
//         using Ti = int;
//         using dtiles_t = zs::TileVector<T,32>;
//         using tiles_t = typename ZenoParticles::particles_t;
//         using bvh_t = zs::LBvh<3,int,T>;
//         using bv_t = zs::AABBBox<3, T>;
//         using vec3 = zs::vec<T, 3>;

//         virtual void apply() override {
//             using namespace zs;

//             auto zsparticles = get_input<ZenoParticles>("ZSParticles");

//             if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
//                 throw std::runtime_error("the input zsparticles has no surface tris");
//             if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
//                 throw std::runtime_error("the input zsparticles has no surface lines");
//             if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
//                 throw std::runtime_error("the input zsparticles has no surface points");
//             // if(!zsparticles->hasBvh(ZenoParticles::s_surfTriTag)) {
//             //     throw std::runtime_error("the input zsparticles has no surface tris's spacial structure");
//             // }
//             // if(!zsparticles->hasBvh(ZenoParticles::s_surfEdgeTag)) {
//             //     throw std::runtime_error("the input zsparticles has no surface edge's spacial structure");
//             // }
//             // if(!zsparticles->hasBvh(ZenoParticles::s_surfVertTag))  {
//             //     throw std::runtime_error("the input zsparticles has no surface vert's spacial structure");
//             // }

//             const auto& verts = zsparticles->getParticles();

//             auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
//             auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
//             auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

//             // auto& stBvh = zsparticles->bvh(ZenoParticles::s_surfTriTag);
//             // auto& seBvh = zsparticles->bvh(ZenoParticles::s_surfEdgeTag);

//             auto in_collisionEps = get_input2<float>("in_collisionEps");
//             auto out_collisionEps = get_input2<float>("out_collisionEps");

//             dtiles_t sttemp(tris.get_allocator(),
//                 {
//                     {"nrm",3}
//                 },tris.size()
//             );
//             dtiles_t setemp(lines.get_allocator(),
//                 {
//                     {"nrm",3}
//                 },lines.size()
//             );
            
//             dtiles_t cptemp(points.get_allocator(),
//                 {
//                     {"inds",4},
//                     {"area",1},
//                     {"inverted",1}
//                 },points.size() * MAX_FP_COLLISION_PAIRS);


//             constexpr auto space = execspace_e::cuda;
//             auto cudaPol = cuda_exec();

//             std::vector<zs::PropertyTag> cv_tags{{"xs",3},{"xe",3}};
//             auto cv_buffer = typename ZenoParticles::particles_t(cv_tags,points.size() * MAX_FP_COLLISION_PAIRS,zs::memsrc_e::device,0);
//             std::vector<zs::PropertyTag> cv_pt_tags{{"p",3},{"t0",3},{"t1",3},{"t2",3}};
//             auto cv_pt_buffer = typename ZenoParticles::particles_t(cv_pt_tags,points.size() * MAX_FP_COLLISION_PAIRS,zs::memsrc_e::device,0);

// #if 0

//             if(!calculate_facet_normal(cudaPol,verts,"x",tris,sttemp,"nrm")){
//                     throw std::runtime_error("fail updating facet normal");
//             }


//             // TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",etemp,"inds");



//             if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
//                 verts,"x",
//                 lines,
//                 tris,
//                 sttemp,"nrm",
//                 setemp,"nrm")){
//                     throw std::runtime_error("fail calculate cell bisector normal");
//             } 

//             auto stbvs = retrieve_bounding_volumes(cudaPol,verts,tris,wrapv<3>{},(T)0.0,"x");
//             auto sebvs = retrieve_bounding_volumes(cudaPol,verts,lines,wrapv<2>{},(T)0.0,"x");
//             stBvh.refit(cudaPol,stbvs);
//             seBvh.refit(cudaPol,sebvs);

//             auto avgl = compute_average_edge_length(cudaPol,verts,"x",tris);
//             auto bvh_thickness = 5 * avgl;

//             TILEVEC_OPS::fill<MAX_FP_COLLISION_PAIRS>(cudaPol,sptemp,"fp_collision_pairs",zs::vec<int,MAX_FP_COLLISION_PAIRS>::uniform(-1).template reinterpret_bits<T>());
//             cudaPol(zs::range(points.size()),[collisionEps = collisionEps,
//                             verts = proxy<space>({},verts),
//                             sttemp = proxy<space>({},sttemp),
//                             setemp = proxy<space>({},setemp),
//                             sptemp = proxy<space>({},sptemp),
//                             points = proxy<space>({},points),
//                             lines = proxy<space>({},lines),
//                             tris = proxy<space>({},tris),
//                             stbvh = proxy<space>(stBvh),thickness = bvh_thickness] ZS_LAMBDA(int svi) mutable {


//                 auto vi = reinterpret_bits<int>(points("inds",svi));
//                 // auto is_vertex_inverted = reinterpret_bits<int>(verts("is_inverted",vi));
//                 // if(is_vertex_inverted)
//                 //     return;

//                 auto p = verts.template pack<3>("x",vi);
//                 auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};

//                 int nm_collision_pairs = 0;
//                 auto process_vertex_face_collision_pairs = [&](int stI) {
//                     auto tri = tris.pack(dim_c<3>, "inds",stI).reinterpret_bits(int_c);
//                     if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
//                         return;

//                     zs::vec<T,3> t[3] = {};
//                     t[0] = verts.template pack<3>("x",tri[0]);
//                     t[1] = verts.template pack<3>("x",tri[1]);
//                     t[2] = verts.template pack<3>("x",tri[2]);

//                     bool collide = false;

//                     if(COLLISION_UTILS::is_inside_the_cell(verts,"x",
//                             lines,tris,
//                             sttemp,"nrm",
//                             setemp,"nrm",
//                             stI,p,collisionEps)) {
//                         collide = true;
//                     }


//                     if(!collide)
//                         return;

//                     if(nm_collision_pairs  < MAX_FP_COLLISION_PAIRS) {
//                         sptemp("fp_collision_pairs",nm_collision_pairs++,svi) = reinterpret_bits<T>(stI);
//                     }
//                 };
//                 stbvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
//             });


//            cudaPol(zs::range(points.size()),
//                 [cv_buffer = proxy<space>({},cv_buffer),cv_pt_buffer = proxy<space>({},cv_pt_buffer),
//                         sptemp = proxy<space>({},sptemp),verts = proxy<space>({},verts),points = proxy<space>({},points),tris = proxy<space>({},tris)] ZS_LAMBDA(int pi) mutable {
//                     auto collision_pairs = sptemp.template pack<MAX_FP_COLLISION_PAIRS>("fp_collision_pairs",pi).reinterpret_bits(int_c);
//                     auto vi = reinterpret_bits<int>(points("inds",pi));
//                     auto pvert = verts.template pack<3>("x",vi);

//                     for(int i = 0;i != MAX_FP_COLLISION_PAIRS;++i){
//                         auto sti = collision_pairs[i];
//                         if(sti < 0){
//                             cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            
//                             cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;

//                         }else {
//                             auto tri = tris.template pack<3>("inds",sti).reinterpret_bits(int_c);
//                             auto t0 = verts.template pack<3>("x",tri[0]);
//                             auto t1 = verts.template pack<3>("x",tri[1]);
//                             auto t2 = verts.template pack<3>("x",tri[2]);
//                             auto center = (t0 + t1 + t2) / (T)3.0;

//                             cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = center;

//                             cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = t0;
//                             cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = t1;
//                             cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = t2;

//                         }
//                     }
//             });

// #else
//             // auto stbvs = retrieve_bounding_volumes(cudaPol,verts,tris,wrapv<3>{},(T)0.0,"x");
//             // stBvh.refit(cudaPol,stbvs);

//             COLLISION_UTILS::do_facet_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
//                 verts,"x",
//                 points,
//                 lines,
//                 tris,
//                 sttemp,
//                 setemp,
//                 cptemp,
//                 // stBvh,
//                 in_collisionEps,out_collisionEps);



//             cudaPol(zs::range(points.size()),
//                 [cptemp = proxy<space>({},cptemp),verts = proxy<space>({},verts),
//                     cv_buffer = proxy<space>({},cv_buffer),
//                     cv_pt_buffer = proxy<space>({},cv_pt_buffer),
//                     points = proxy<space>({},points)] ZS_LAMBDA(int pi) mutable {
//                         for(int i = 0;i != MAX_FP_COLLISION_PAIRS;++i) {
//                             auto inds = cptemp.template pack<4>("inds",pi * MAX_FP_COLLISION_PAIRS + i).reinterpret_bits(int_c);
//                             bool contact = true;
//                             auto pvert = zs::vec<T,3>::zeros();
//                             for(int j = 0;j != 4;++j)
//                                 if(inds[j] < 0)
//                                     contact = false;
//                             if(contact) {
//                                 pvert = verts.template pack<3>("x",inds[0]);
//                                 auto t0 = verts.template pack<3>("x",inds[1]);
//                                 auto t1 = verts.template pack<3>("x",inds[2]);
//                                 auto t2 = verts.template pack<3>("x",inds[3]);
//                                 auto center = (t0 + t1 + t2) / (T)3.0;

//                                 cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                                 cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = center;

//                                 cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                                 cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = t0;
//                                 cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = t1;
//                                 cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = t2;                                
//                             }else{
//                                 cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                                 cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                                
//                                 cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                                 cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                                 cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                                 cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;                                
//                             }
//                         }
//             });
            

// #endif
//             // cudaPol.syncCtx();


//             cv_buffer = cv_buffer.clone({zs::memsrc_e::host});
//             auto collisionFacetVis = std::make_shared<zeno::PrimitiveObject>();
//             auto& cv_verts = collisionFacetVis->verts;
//             auto& cv_lines = collisionFacetVis->lines;
//             cv_verts.resize(points.size() * 2 * MAX_FP_COLLISION_PAIRS);
//             cv_lines.resize(points.size() * MAX_FP_COLLISION_PAIRS);

//             auto ompPol = omp_exec();  
//             constexpr auto omp_space = execspace_e::openmp;

//             ompPol(zs::range(cv_buffer.size()),
//                 [cv_buffer = proxy<omp_space>({},cv_buffer),&cv_verts,&cv_lines] (int pi) mutable {
//                     auto xs = cv_buffer.template pack<3>("xs",pi);
//                     auto xe = cv_buffer.template pack<3>("xe",pi);
//                     cv_verts[pi * 2 + 0] = zeno::vec3f(xs[0],xs[1],xs[2]);
//                     cv_verts[pi * 2 + 1] = zeno::vec3f(xe[0],xe[1],xe[2]);
//                     cv_lines[pi] = zeno::vec2i(pi * 2 + 0,pi * 2 + 1);
//             });

//             set_output("collisionFacetVis",std::move(collisionFacetVis));



//             cv_pt_buffer = cv_pt_buffer.clone({zs::memsrc_e::host});
//             auto colPointFacetPairVis = std::make_shared<zeno::PrimitiveObject>();
//             auto& cv_pt_verts = colPointFacetPairVis->verts;
//             auto& cv_pt_tris = colPointFacetPairVis->tris;

//             cv_pt_verts.resize(cv_pt_buffer.size() * 4);
//             cv_pt_tris.resize(cv_pt_buffer.size());

//             ompPol(zs::range(cv_pt_buffer.size()),
//                 [&cv_pt_verts,&cv_pt_tris,cv_pt_buffer = proxy<omp_space>({},cv_pt_buffer)] (int pi) mutable {
//                     cv_pt_verts[pi * 4 + 0] = cv_pt_buffer.template pack<3>("p",pi).to_array();
//                     cv_pt_verts[pi * 4 + 1] = cv_pt_buffer.template pack<3>("t0",pi).to_array();
//                     cv_pt_verts[pi * 4 + 2] = cv_pt_buffer.template pack<3>("t1",pi).to_array();
//                     cv_pt_verts[pi * 4 + 3] = cv_pt_buffer.template pack<3>("t2",pi).to_array();

//                     cv_pt_tris[pi] = zeno::vec3i(pi * 4 + 1,pi * 4 + 2,pi * 4 + 3);
//             });


//             set_output("colPointFacetPairVis",std::move(colPointFacetPairVis));

//         }
//     };


//     ZENDEFNODE(VisualizeFacetPointIntersection, {{"ZSParticles",{"float","in_collisionEps","0.01"},{"float","out_collisionEps","0.01"}},
//                                     {"collisionFacetVis","colPointFacetPairVis"},
//                                     {
//                                     },
//                                     {"ZSGeometry"}});



//     struct VisualizeEdgeEdgeIntersection : zeno::INode {
//         using T = float;
//         using Ti = int;
//         using dtiles_t = zs::TileVector<T,32>;
//         using tiles_t = typename ZenoParticles::particles_t;

//         virtual void apply() override {
//             using namespace zs;
//             auto zsparticles = get_input<ZenoParticles>("ZSParticles");

//             if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
//                 throw std::runtime_error("the input zsparticles has no surface tris");
//             if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
//                 throw std::runtime_error("the input zsparticles has no surface lines");
//             if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
//                 throw std::runtime_error("the input zsparticles has no surface points");

//             const auto& verts = zsparticles->getParticles();
//             auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
//             auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
//             auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];        

//             auto in_collisionEps = get_input2<float>("in_collisionEps");
//             auto out_collisionEps = get_input2<float>("out_collisionEps");  


//             dtiles_t sttemp(tris.get_allocator(),
//                 {
//                     {"nrm",3}
//                 },tris.size()
//             );
//             dtiles_t setemp(lines.get_allocator(),
//                 {
//                     {"nrm",3},
//                     {"inds",4},
//                     {"area",1},
//                     {"inverted",1},
//                     {"abary",2},
//                     {"bbary",2}
//                 },lines.size()
//             );
            
//             constexpr auto space = execspace_e::cuda;
//             auto cudaPol = cuda_exec();

//             std::cout << "before do edge edge collision detection" << std::endl;

//             COLLISION_UTILS::do_edge_edge_collision_detection(cudaPol,
//                 verts,"x",
//                 points,lines,tris,
//                 sttemp,setemp,
//                 setemp,
//                 in_collisionEps,out_collisionEps);
            
//             // std::vector<zs::PropertyTag> cv_tags{{"xs",3},{"xe",3}};
//             // auto cv_buffer = typename ZenoParticles::particles_t(cv_tags,setemp.size(),zs::memsrc_e::device,0);
//             std::vector<zs::PropertyTag> cv_ee_tags{{"a0",3},{"a1",3},{"b0",3},{"b1",3},{"abary",2},{"bbary",2}};
//             auto cv_ee_buffer = typename ZenoParticles::particles_t(cv_ee_tags,setemp.size(),zs::memsrc_e::device,0);

//             cudaPol(zs::range(setemp.size()),
//                 [setemp = proxy<space>({},setemp),verts = proxy<space>({},verts),
//                     cv_ee_buffer = proxy<space>({},cv_ee_buffer)] ZS_LAMBDA(int ei) mutable {
//                         auto inds = setemp.template pack<4>("inds",ei).reinterpret_bits(int_c);
//                         bool collide = true;
//                         if(inds[0] < 0 || inds[1] < 0 || inds[2] < 0 || inds[3] < 0)
//                             collide = false;
//                         if(collide) {
//                             auto abary = setemp.template pack<2>("abary",ei);
//                             auto bbary = setemp.template pack<2>("bbary",ei);
//                             printf("find collision pairs : %d %d %d %d with bary %f %f %f %f\n",inds[0],inds[1],inds[2],inds[3],
//                                 (float)abary[0],(float)abary[1],(float)bbary[0],(float)bbary[1]);
//                             cv_ee_buffer.template tuple<3>("a0",ei) = verts.template pack<3>("x",inds[0]);
//                             cv_ee_buffer.template tuple<3>("a1",ei) = verts.template pack<3>("x",inds[1]);
//                             cv_ee_buffer.template tuple<3>("b0",ei) = verts.template pack<3>("x",inds[2]);
//                             cv_ee_buffer.template tuple<3>("b1",ei) = verts.template pack<3>("x",inds[3]);
//                             cv_ee_buffer.template tuple<2>("abary",ei) = abary;
//                             cv_ee_buffer.template tuple<2>("bbary",ei) = bbary;
//                         }else {
//                             cv_ee_buffer.template tuple<3>("a0",ei) = zs::vec<T,3>::zeros();
//                             cv_ee_buffer.template tuple<3>("a1",ei) = zs::vec<T,3>::zeros();
//                             cv_ee_buffer.template tuple<3>("b0",ei) = zs::vec<T,3>::zeros();
//                             cv_ee_buffer.template tuple<3>("b1",ei) = zs::vec<T,3>::zeros();
//                             cv_ee_buffer.template tuple<2>("abary",ei) = zs::vec<T,2>((T)1.0,0.0);
//                             cv_ee_buffer.template tuple<2>("bbary",ei) = zs::vec<T,2>((T)1.0,0.0);
//                         }
//                 });

//             cv_ee_buffer = cv_ee_buffer.clone({zs::memsrc_e::host});


//             auto ompPol = omp_exec();  
//             constexpr auto omp_space = execspace_e::openmp;

//             auto collisionEdgeVis = std::make_shared<zeno::PrimitiveObject>();
//             auto& ee_verts = collisionEdgeVis->verts;
//             auto& ee_lines = collisionEdgeVis->lines;
//             ee_verts.resize(cv_ee_buffer.size() * 2);
//             ee_lines.resize(cv_ee_buffer.size());


//             ompPol(zs::range(cv_ee_buffer.size()),
//                 [cv_ee_buffer = proxy<omp_space>({},cv_ee_buffer),&ee_verts,&ee_lines] (int eei) mutable {
//                     auto a0 = cv_ee_buffer.template pack<3>("a0",eei);
//                     auto a1 = cv_ee_buffer.template pack<3>("a1",eei);
//                     auto b0 = cv_ee_buffer.template pack<3>("b0",eei);
//                     auto b1 = cv_ee_buffer.template pack<3>("b1",eei);     
                    
//                     auto abary = cv_ee_buffer.template pack<2>("abary",eei);
//                     auto bbary = cv_ee_buffer.template pack<2>("bbary",eei);

//                     // auto ac = (a0 + a1) / (T)2.0;
//                     // auto bc = (b0 + b1) / (T)2.0;

//                     auto ac = abary[0] * a0 + abary[1] * a1;
//                     auto bc = bbary[0] * b0 + bbary[1] * b1;

//                     ee_verts[eei * 2 + 0] = zeno::vec3f(ac[0],ac[1],ac[2]);
//                     ee_verts[eei * 2 + 1] = zeno::vec3f(bc[0],bc[1],bc[2]);
//                     ee_lines[eei] = zeno::vec2i(eei * 2 + 0,eei * 2 + 1);
//             });

//             set_output("collisionEdgeVis",std::move(collisionEdgeVis));

//             auto colEdgetPairVis = std::make_shared<zeno::PrimitiveObject>();
//             auto& cv_ee_verts = colEdgetPairVis->verts;
//             auto& cv_ee_lines = colEdgetPairVis->lines;

//             cv_ee_verts.resize(cv_ee_buffer.size() * 4);
//             cv_ee_lines.resize(cv_ee_buffer.size() * 2);

//             ompPol(zs::range(cv_ee_buffer.size()),
//                 [&cv_ee_verts,&cv_ee_lines,cv_ee_buffer = proxy<omp_space>({},cv_ee_buffer)] (int eei) mutable {
//                     cv_ee_verts[eei * 4 + 0] = cv_ee_buffer.template pack<3>("a0",eei).to_array();
//                     cv_ee_verts[eei * 4 + 1] = cv_ee_buffer.template pack<3>("a1",eei).to_array();
//                     cv_ee_verts[eei * 4 + 2] = cv_ee_buffer.template pack<3>("b0",eei).to_array();
//                     cv_ee_verts[eei * 4 + 3] = cv_ee_buffer.template pack<3>("b1",eei).to_array();

//                     cv_ee_lines[eei * 2 + 0] = zeno::vec2i(eei * 4 + 0,eei * 4 + 1);
//                     cv_ee_lines[eei * 2 + 1] = zeno::vec2i(eei * 4 + 2,eei * 4 + 3);
//             });


//             set_output("colEdgetPairVis",std::move(colEdgetPairVis));            
//         }
//     };

//     ZENDEFNODE(VisualizeEdgeEdgeIntersection, {{"ZSParticles",{"float","in_collisionEps","0.01"},{"float","out_collisionEps","0.01"}},
//                                     {"collisionEdgeVis","colEdgetPairVis"},
//                                     {
//                                     },
//                                     {"ZSGeometry"}});


struct VisualizeKineCollision : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;
    using vec3 = zs::vec<T, 3>;

    virtual void apply() override {
        using namespace zs;
        auto zsparticles = get_input<ZenoParticles>("ZSParticles");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
            throw std::runtime_error("the input zsparticles has no surface tris");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
            throw std::runtime_error("the input zsparticles has no surface lines");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
            throw std::runtime_error("the input zsparticles has no surface points");
        
        const auto& eles = zsparticles->getQuadraturePoints();
        const auto& verts = zsparticles->getParticles();
        auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
        auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

        // ksurf should be a surface tris
        auto ksurf = get_input<ZenoParticles>("KinematicSurf");
        auto kverts = ksurf->getParticles();
        // if(!kverts.hasProperty("nrm")) {
        //     fmt::print(fg(fmt::color::red),"KinematicSurf has no surface normal\n");
        //     throw std::runtime_error("the Kinematic surf has no surface normal");
        // }
        
        dtiles_t sttemp(tris.get_allocator(),
            {
                {"nrm",3}
            },tris.size()
        );
        dtiles_t setemp(lines.get_allocator(),
            {
                // {"inds",4},
                // {"area",1},
                // {"inverted",1},
                // {"abary",2},
                // {"bbary",2},
                {"nrm",3}
                // {"grad",12},
                // {"H",12*12}
            },lines.size()
        );
        dtiles_t sptemp(points.get_allocator(),
            {
                {"nrm",3}
            },points.size()
        );

    
        dtiles_t fp_buffer(kverts.get_allocator(),
            {
                {"inds",2},
                {"area",1},
                {"inverted",1}
            },kverts.size() * MAX_FP_COLLISION_PAIRS);
        
        dtiles_t gh_buffer(points.get_allocator(),
            {
                {"inds",4},
                {"H",12*12},
                {"grad",12}
            },eles.size());


        auto in_collisionEps = get_input2<float>("in_collisionEps");
        auto out_collisionEps = get_input2<float>("out_collisionEps");

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();


        auto kverts_ = typename ZenoParticles::particles_t({
            {"x",3},
            {"area",1}},kverts.size(),zs::memsrc_e::device,0);  
        TILEVEC_OPS::copy<3>(cudaPol,kverts,"x",kverts_,"x");
        TILEVEC_OPS::fill(cudaPol,kverts_,"area",(T)1.0);
        TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",gh_buffer,"inds");              

        COLLISION_UTILS::do_kinematic_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
            verts,"x",
            points,
            lines,
            tris,
            setemp,
            sttemp,
            kverts_,
            fp_buffer,
            in_collisionEps,out_collisionEps);
        
        std::vector<zs::PropertyTag> cv_tags{{"xp",3},{"xt",3},{"t0",3},{"t1",3},{"t2",3}};
        auto cv_buffer = typename ZenoParticles::particles_t(cv_tags,fp_buffer.size(),zs::memsrc_e::device,0);

        cudaPol(zs::range(fp_buffer.size()),
            [fp_buffer = proxy<space>({},fp_buffer),
                verts = proxy<space>({},verts),
                tris = proxy<space>({},tris),
                kverts = proxy<space>({},kverts),
                cv_buffer = proxy<space>({},cv_buffer)] ZS_LAMBDA(int ci) mutable {
                    auto cp = fp_buffer.pack(dim_c<2>,"inds",ci).reinterpret_bits(int_c);

                    auto contact = true;
                    for(int i = 0;i != 2;++i)
                        if(cp[i] < 0){
                            contact = false;
                            break;
                        }
                    auto pvert = zs::vec<T,3>::zeros();
                    if(contact) {
                        // auto pidx = cp[0];
                        auto tri = tris.pack(dim_c<3>,"inds",cp[1]).reinterpret_bits(int_c);
                        pvert = kverts.pack(dim_c<3>,"x",cp[0]);
                        auto t0 = verts.pack(dim_c<3>,"x",tri[0]);
                        auto t1 = verts.pack(dim_c<3>,"x",tri[1]);
                        auto t2 = verts.pack(dim_c<3>,"x",tri[2]);

                        auto tc = (t0 + t1 + t2)/(T)3.0;

                        cv_buffer.template tuple<3>("xp",ci) = pvert;
                        cv_buffer.template tuple<3>("xt",ci) = tc;
                        cv_buffer.template tuple<3>("t0",ci) = t0;
                        cv_buffer.template tuple<3>("t1",ci) = t1;
                        cv_buffer.template tuple<3>("t2",ci) = t2;
                    } else {
                        cv_buffer.template tuple<3>("xp",ci) = pvert;
                        cv_buffer.template tuple<3>("xt",ci) = pvert;
                        cv_buffer.template tuple<3>("t0",ci) = pvert;
                        cv_buffer.template tuple<3>("t1",ci) = pvert;
                        cv_buffer.template tuple<3>("t2",ci) = pvert;
                    }
                    
        });

        auto ompPol = omp_exec();  
        constexpr auto omp_space = execspace_e::openmp;

        cv_buffer = cv_buffer.clone({zs::memsrc_e::host});
        auto colPointTriPairVis = std::make_shared<zeno::PrimitiveObject>();
        auto& cv_pt_verts = colPointTriPairVis->verts;
        auto& cv_pt_tris = colPointTriPairVis->tris;

        cv_pt_verts.resize(cv_buffer.size() * 4);
        cv_pt_tris.resize(cv_buffer.size());

        ompPol(zs::range(cv_buffer.size()),
            [&cv_pt_verts,&cv_pt_tris,cv_buffer = proxy<omp_space>({},cv_buffer)] (int ci) mutable {
                cv_pt_verts[ci * 4 + 0] = cv_buffer.pack(dim_c<3>,"xp",ci).to_array();
                cv_pt_verts[ci * 4 + 1] = cv_buffer.pack(dim_c<3>,"t0",ci).to_array();
                cv_pt_verts[ci * 4 + 2] = cv_buffer.pack(dim_c<3>,"t1",ci).to_array();
                cv_pt_verts[ci * 4 + 3] = cv_buffer.pack(dim_c<3>,"t2",ci).to_array();
                
                cv_pt_tris[ci] = zeno::vec3i(ci * 4 + 1,ci * 4 + 2,ci * 4 + 3);
        });

        set_output("colPointFacePairVis",std::move(colPointTriPairVis));

        auto colCenterLineVis = std::make_shared<zeno::PrimitiveObject>();
        auto& cv_cl_verts = colCenterLineVis->verts;
        auto& cv_cl_lines = colCenterLineVis->lines;
        
        cv_cl_verts.resize(cv_buffer.size() * 2);
        cv_cl_lines.resize(cv_buffer.size());

        ompPol(zs::range(cv_buffer.size()),
            [cv_buffer = proxy<omp_space>({},cv_buffer),&cv_cl_verts,&cv_cl_lines] (int ci) mutable {
                cv_cl_verts[ci * 2 + 0] = cv_buffer.pack(dim_c<3>,"xp",ci).to_array();
                cv_cl_verts[ci * 2 + 1] = cv_buffer.pack(dim_c<3>,"xt",ci).to_array();
                cv_cl_lines[ci] = zeno::vec2i(ci * 2 + 0,ci * 2 + 1);
        });

        set_output("colConnVis",std::move(colCenterLineVis));


        COLLISION_UTILS::evaluate_kinematic_fp_collision_grad_and_hessian(
            cudaPol,
            eles,
            verts,"x","v",(T)1.0,
            tris,
            kverts_,
            fp_buffer,
            gh_buffer,0,
            in_collisionEps,out_collisionEps,
            (T)1.0,
            (T)1.0,(T)1.0,(T)0.01);

        dtiles_t vtemp(verts.get_allocator(),
            {
                {"x",3},
                {"dir",3},
            },verts.size());
        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"x");
        TILEVEC_OPS::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());

        TILEVEC_OPS::assemble_range(cudaPol,gh_buffer,"grad","inds",vtemp,"dir",0,gh_buffer.size());        
        vtemp = vtemp.clone({zs::memsrc_e::host}); 

        auto nodalForceVis = std::make_shared<zeno::PrimitiveObject>();       
        auto& spverts = nodalForceVis->verts;
        spverts.resize(vtemp.size() * 2);
        auto& splines = nodalForceVis->lines;
        splines.resize(vtemp.size());

        auto scale = get_input2<float>("scale");
        ompPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),&spverts,&splines,scale] (int vi) mutable {
                auto xs = vtemp.template pack<3>("x",vi);
                auto dir = vtemp.template pack<3>("dir",vi);

                auto xe = xs + scale * dir;

                spverts[vi * 2 + 0] = xs.to_array();
                spverts[vi * 2 + 1] = xe.to_array();
                splines[vi] = zeno::vec2i(vi * 2 + 0,vi * 2 + 1);               
        });

        set_output("FPNodalForceVis",std::move(nodalForceVis));


    }
};


ZENDEFNODE(VisualizeKineCollision, {{"ZSParticles","KinematicSurf",{"float","in_collisionEps"},{"float","out_collisionEps"},{"float","scale"}},
                                  {
                                        "colPointFacePairVis",
                                        "colConnVis",
                                        "FPNodalForceVis"
                                    },
                                  {
                                  },
                                  {"ZSGeometry"}});


// struct VisualizeCollision : zeno::INode {

//     using T = float;
//     using Ti = int;
//     using dtiles_t = zs::TileVector<T,32>;
//     using tiles_t = typename ZenoParticles::particles_t;
//     using bvh_t = zs::LBvh<3,int,T>;
//     using bv_t = zs::AABBBox<3, T>;
//     using vec3 = zs::vec<T, 3>;


//     virtual void apply() override {
//         using namespace zs;

//         auto zsparticles = get_input<ZenoParticles>("ZSParticles");

//         if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
//             throw std::runtime_error("the input zsparticles has no surface tris");
//         if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
//             throw std::runtime_error("the input zsparticles has no surface lines");
//         if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
//             throw std::runtime_error("the input zsparticles has no surface points");
//         // if(!zsparticles->hasBvh(ZenoParticles::s_surfTriTag)) {
//         //     throw std::runtime_error("the input zsparticles has no surface tris's spacial structure");
//         // }
//         // if(!zsparticles->hasBvh(ZenoParticles::s_surfEdgeTag)) {
//         //     throw std::runtime_error("the input zsparticles has no surface edge's spacial structure");
//         // }
//         // if(!zsparticles->hasBvh(ZenoParticles::s_surfVertTag))  {
//         //     throw std::runtime_error("the input zsparticles has no surface vert's spacial structure");
//         // }

//         const auto& verts = zsparticles->getParticles();

//         auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
//         auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
//         auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

//         // auto& stBvh = zsparticles->bvh(ZenoParticles::s_surfTriTag);
//         // auto& seBvh = zsparticles->bvh(ZenoParticles::s_surfEdgeTag);

//         dtiles_t sttemp(tris.get_allocator(),
//             {
//                 {"nrm",3},
//                 {"x",3}
//             },tris.size()
//         );
//         dtiles_t setemp(lines.get_allocator(),
//             {
//                 // {"inds",4},
//                 // {"area",1},
//                 // {"inverted",1},
//                 // {"abary",2},
//                 // {"bbary",2},
//                 {"nrm",3}
//                 // {"grad",12},
//                 // {"H",12*12}
//             },lines.size()
//         );
//         dtiles_t sptemp(points.get_allocator(),
//             {
//                 {"nrm",3},
//                 {"x",3}
//             },points.size()
//         );


//         dtiles_t ee_buffer(lines.get_allocator(),
//             {
//                 {"inds",4},
//                 {"area",1},
//                 {"inverted",1},
//                 {"abary",2},
//                 {"bbary",2}
//             },lines.size());

//         dtiles_t gh_buffer(points.get_allocator(),
//             {
//                 {"inds",4},
//                 {"H",12*12},
//                 {"grad",12}
//             },points.size() * MAX_FP_COLLISION_PAIRS + lines.size());


//         dtiles_t vtemp(verts.get_allocator(),
//             {
//                 {"xn",3},
//                 {"dir",3},
//                 {"active",1},
//                 // {"gia_tag"}
//             },verts.size());


//         auto in_collisionEps = get_input2<float>("in_collisionEps");
//         auto out_collisionEps = get_input2<float>("out_collisionEps");

//         constexpr auto space = execspace_e::cuda;
//         auto cudaPol = cuda_exec();

//         // calculate facet-point collision pairs and force




//         #if 1
//         dtiles_t fp_buffer(points.get_allocator(),
//             {
//                 {"inds",4},
//                 {"area",1},
//                 {"inverted",1}
//             },points.size() * MAX_FP_COLLISION_PAIRS);

//         dtiles_t surf_tris_buffer{tris.get_allocator(),{
//             {"inds",3},
//             {"nrm",3}
//         },tris.size()};

//         dtiles_t surf_verts_buffer{points.get_allocator(),{
//             {"inds",1},
//             {"xn",3}
//         },points.size()};

//         TILEVEC_OPS::copy(cudaPol,verts,"x",vtemp,"xn");
//         TILEVEC_OPS::copy(cudaPol,verts,"active",vtemp,"active");

//         // TILEVEC_OPS::copy(cudaPol,points,"inds",surf_verts_buffer,"inds");
//         TILEVEC_OPS::copy(cudaPol,tris,"inds",surf_tris_buffer,"inds");
//         // reorder_topology(cudaPol,points,surf_tris_buffer);
//         // zs::Vector<int> nodal_colors{surf_verts_buffer.get_allocator(),surf_verts_buffer.size()};
//         // zs::Vector<zs::vec<int,2>> instBuffer{surf_verts_buffer.get_allocator(),surf_verts_buffer.size() * 8};

//         // topological_sample(cudaPol,points,vtemp,"xn",surf_verts_buffer);
//         // auto nm_insts = do_global_self_intersection_analysis_on_surface_mesh(cudaPol,
//         //     surf_verts_buffer,"xn",surf_tris_buffer,instBuffer,nodal_colors);
//         // TILEVEC_OPS::fill(cudaPol,vtemp,"gia_tag",(T)0.0);
//         // cudaPol(zs::range(nodal_colors.size()),[
//         //     nodal_colors = proxy<space>(nodal_colors),
//         //     vtemp = proxy<space>({},vtemp),
//         //     points = proxy<space>({},points)] ZS_LAMBDA(int pi) mutable {
//         //         auto vi = zs::reinterpret_bits<int>(points("inds",pi));
//         //         if(nodal_colors[pi] == 1)
//         //             vtemp("gia_tag",vi) = (T)1.0;
//         // });

//         COLLISION_UTILS::do_facet_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
//             vtemp,"xn",
//             points,
//             lines,
//             tris,
//             sttemp,
//             setemp,
//             fp_buffer,
//             in_collisionEps,out_collisionEps);


//         std::vector<zs::PropertyTag> cv_tags{{"xs",3},{"xe",3}};
//         auto cv_buffer = typename ZenoParticles::particles_t(cv_tags,points.size() * MAX_FP_COLLISION_PAIRS,zs::memsrc_e::device,0);
//         std::vector<zs::PropertyTag> cv_pt_tags{{"p",3},{"t0",3},{"t1",3},{"t2",3}};
//         auto cv_pt_buffer = typename ZenoParticles::particles_t(cv_pt_tags,points.size() * MAX_FP_COLLISION_PAIRS,zs::memsrc_e::device,0);


//         cudaPol(zs::range(points.size()),
//             [fp_buffer = proxy<space>({},fp_buffer),verts = proxy<space>({},verts),
//                 cv_buffer = proxy<space>({},cv_buffer),
//                 cv_pt_buffer = proxy<space>({},cv_pt_buffer),
//                 points = proxy<space>({},points)] ZS_LAMBDA(int pi) mutable {
//                     for(int i = 0;i != MAX_FP_COLLISION_PAIRS;++i) {
//                         auto inds = fp_buffer.template pack<4>("inds",pi * MAX_FP_COLLISION_PAIRS + i).reinterpret_bits(int_c);
//                         bool contact = true;
//                         auto pvert = zs::vec<T,3>::zeros();
//                         for(int j = 0;j != 4;++j)
//                             if(inds[j] < 0)
//                                 contact = false;
//                         if(contact) {
//                             pvert = verts.template pack<3>("x",inds[0]);
//                             auto t0 = verts.template pack<3>("x",inds[1]);
//                             auto t1 = verts.template pack<3>("x",inds[2]);
//                             auto t2 = verts.template pack<3>("x",inds[3]);
//                             auto center = (t0 + t1 + t2) / (T)3.0;

//                             cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = center;

//                             cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = t0;
//                             cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = t1;
//                             cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = t2;                                
//                         }else{
//                             cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            
//                             cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
//                             cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;                                
//                         }
//                     }
//         });
        
//         cv_buffer = cv_buffer.clone({zs::memsrc_e::host});
//         auto collisionFacetVis = std::make_shared<zeno::PrimitiveObject>();
//         auto& cv_verts = collisionFacetVis->verts;
//         auto& cv_lines = collisionFacetVis->lines;
//         cv_verts.resize(points.size() * 2 * MAX_FP_COLLISION_PAIRS);
//         cv_lines.resize(points.size() * MAX_FP_COLLISION_PAIRS);

//         auto ompPol = omp_exec();  
//         constexpr auto omp_space = execspace_e::openmp;

//         ompPol(zs::range(cv_buffer.size()),
//             [cv_buffer = proxy<omp_space>({},cv_buffer),&cv_verts,&cv_lines] (int pi) mutable {
//                 auto xs = cv_buffer.template pack<3>("xs",pi);
//                 auto xe = cv_buffer.template pack<3>("xe",pi);
//                 cv_verts[pi * 2 + 0] = zeno::vec3f(xs[0],xs[1],xs[2]);
//                 cv_verts[pi * 2 + 1] = zeno::vec3f(xe[0],xe[1],xe[2]);
//                 cv_lines[pi] = zeno::vec2i(pi * 2 + 0,pi * 2 + 1);
//         });

//         set_output("collisionFacetVis",std::move(collisionFacetVis));

//         cv_pt_buffer = cv_pt_buffer.clone({zs::memsrc_e::host});
//         auto colPointFacetPairVis = std::make_shared<zeno::PrimitiveObject>();
//         auto& cv_pt_verts = colPointFacetPairVis->verts;
//         auto& cv_pt_tris = colPointFacetPairVis->tris;

//         cv_pt_verts.resize(cv_pt_buffer.size() * 4);
//         cv_pt_tris.resize(cv_pt_buffer.size());

//         ompPol(zs::range(cv_pt_buffer.size()),
//             [&cv_pt_verts,&cv_pt_tris,cv_pt_buffer = proxy<omp_space>({},cv_pt_buffer)] (int pi) mutable {
//                 cv_pt_verts[pi * 4 + 0] = cv_pt_buffer.template pack<3>("p",pi).to_array();
//                 cv_pt_verts[pi * 4 + 1] = cv_pt_buffer.template pack<3>("t0",pi).to_array();
//                 cv_pt_verts[pi * 4 + 2] = cv_pt_buffer.template pack<3>("t1",pi).to_array();
//                 cv_pt_verts[pi * 4 + 3] = cv_pt_buffer.template pack<3>("t2",pi).to_array();

//                 cv_pt_tris[pi] = zeno::vec3i(pi * 4 + 1,pi * 4 + 2,pi * 4 + 3);
//         });


//         set_output("colPointFacetPairVis",std::move(colPointFacetPairVis));


//         COLLISION_UTILS::evaluate_fp_collision_grad_and_hessian(
//             cudaPol,
//             verts,"x","v",(T)1.0,
//             fp_buffer,
//             gh_buffer,0,
//             in_collisionEps,out_collisionEps,
//             (T)1.0,
//             (T)1.0,(T)1.0,(T)0.0);

//         TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"xn");
//         TILEVEC_OPS::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());

//         TILEVEC_OPS::assemble_range(cudaPol,gh_buffer,"grad","inds",vtemp,"dir",0,fp_buffer.size());

//         auto scale = get_input2<float>("fp_scale");

//         // auto ompPol = omp_exec();  
//         // constexpr auto omp_space = execspace_e::openmp;
        
//         auto nodalForceVis = std::make_shared<zeno::PrimitiveObject>();
//         auto& spverts = nodalForceVis->verts;
//         spverts.resize(vtemp.size() * 2);
//         auto& splines = nodalForceVis->lines;
//         splines.resize(vtemp.size());

//         // auto scale = get_input2<float>("scale");

//         vtemp = vtemp.clone({zs::memsrc_e::host});
//         ompPol(zs::range(vtemp.size()),
//             [vtemp = proxy<space>({},vtemp),&spverts,&splines,scale] (int vi) mutable {
//                 auto xs = vtemp.template pack<3>("xn",vi);
//                 auto dir = vtemp.template pack<3>("dir",vi);

//                 auto xe = xs + scale * dir;

//                 spverts[vi * 2 + 0] = xs.to_array();
//                 spverts[vi * 2 + 1] = xe.to_array();
//                 splines[vi] = zeno::vec2i(vi * 2 + 0,vi * 2 + 1);               
//         });

//         set_output("FPNodalForceVis",std::move(nodalForceVis));
//         #else

//         zs::Vector<zs::vec<int,4>> csPT{points.get_allocator(),points.size()};
//         int nm_csPT = 0;
//         COLLISION_UTILS::do_facet_point_collision_detection(cudaPol,
//             verts,"x",points,tris,csPT,nm_csPT,in_collisionEps,out_collisionEps);

//         std::cout << "nm_csPT : " << nm_csPT << std::endl;

//         std::vector<zs::PropertyTag> cv_tags{{"xs",3},{"xe",3}};
//         auto cv_buffer = typename ZenoParticles::particles_t{cv_tags,nm_csPT,zs::memsrc_e::device,0};
//         std::vector<zs::PropertyTag> cv_pt_tags{{"p",3},{"t0",3},{"t1",3},{"t2",3}};
//         auto cv_pt_buffer = typename ZenoParticles::particles_t(cv_pt_tags,nm_csPT,zs::memsrc_e::device,0);

//         cudaPol(zs::range(nm_csPT),
//             [csPT = proxy<space>(csPT),verts = proxy<space>({},verts), 
//                 cv_buffer = proxy<space>({},cv_buffer),
//                 cv_pt_buffer = proxy<space>({},cv_pt_buffer)] ZS_LAMBDA(int pi) mutable {
//                     auto inds = csPT[pi];
//                     auto pverts = verts.pack(dim_c<3>,"x",inds[0]);
//                     auto t0 = verts.pack(dim_c<3>,"x",inds[1]);
//                     auto t1 = verts.pack(dim_c<3>,"x",inds[2]);
//                     auto t2 = verts.pack(dim_c<3>,"x",inds[3]);
//                     auto center = (t0 + t1 + t2) / (T)3.0;  

//                     cv_buffer.tuple(dim_c<3>,"xs",pi) = pverts;
//                     cv_buffer.tuple(dim_c<3>,"xe",pi) = center;
//                     cv_pt_buffer.tuple(dim_c<3>,"p",pi) = pverts;
//                     cv_pt_buffer.tuple(dim_c<3>,"t0",pi) = t0;
//                     cv_pt_buffer.tuple(dim_c<3>,"t1",pi) = t1;
//                     cv_pt_buffer.tuple(dim_c<3>,"t2",pi) = t2;                  
//         });        
//         cv_buffer = cv_buffer.clone({zs::memsrc_e::host});
//         auto collisionFacetVis = std::make_shared<zeno::PrimitiveObject>();
//         auto& cv_verts = collisionFacetVis->verts;
//         auto& cv_lines = collisionFacetVis->lines;
//         cv_verts.resize(nm_csPT * 2);
//         cv_lines.resize(nm_csPT);

//         auto ompPol = omp_exec();  
//         constexpr auto omp_space = execspace_e::openmp;
//         ompPol(zs::range(cv_buffer.size()),
//             [cv_buffer = proxy<omp_space>({},cv_buffer),&cv_verts,&cv_lines] (int pi) mutable {
//                 cv_verts[pi * 2 + 0] = cv_buffer.pack(dim_c<3>,"xs",pi).to_array();
//                 cv_verts[pi * 2 + 1] = cv_buffer.pack(dim_c<3>,"xe",pi).to_array();
//                 cv_lines[pi] = zeno::vec2i(pi * 2 + 0,pi * 2 + 1);
//         });   
//         set_output("collisionFacetVis",std::move(collisionFacetVis));

//         cv_pt_buffer = cv_pt_buffer.clone({zs::memsrc_e::host});
//         auto colPointFacetPairVis = std::make_shared<zeno::PrimitiveObject>();
//         auto& cv_pt_verts = colPointFacetPairVis->verts;
//         auto& cv_pt_tris = colPointFacetPairVis->tris;

//         cv_pt_verts.resize(nm_csPT * 4);
//         cv_pt_tris.resize(nm_csPT);
//         ompPol(zs::range(cv_pt_buffer.size()),
//             [&cv_pt_verts,&cv_pt_tris,cv_pt_buffer = proxy<omp_space>({},cv_pt_buffer)] (int pi) mutable {
//                 cv_pt_verts[pi * 4 + 0] = cv_pt_buffer.pack(dim_c<3>,"p",pi).to_array();
//                 cv_pt_verts[pi * 4 + 1] = cv_pt_buffer.pack(dim_c<3>,"t0",pi).to_array();
//                 cv_pt_verts[pi * 4 + 2] = cv_pt_buffer.pack(dim_c<3>,"t1",pi).to_array();
//                 cv_pt_verts[pi * 4 + 3] = cv_pt_buffer.pack(dim_c<3>,"t2",pi).to_array();

//                 cv_pt_tris[pi] = zeno::vec3i(pi *4 + 1,pi * 4 + 2,pi * 4 + 3);
//         });
//         set_output("colPointFacetPairVis",std::move(colPointFacetPairVis));

//         // auto nodalForceVis = std::make_shared<zeno::PrimitiveObject>();
//         // set_output("FPNodalForceVis",std::move(nodalForceVis));

//         dtiles_t fp_buffer(points.get_allocator(),{
//             {"inds",4},
//             {"grad",12},
//             {"H",12 * 12},
//         },nm_csPT);
//         COLLISION_UTILS::evaluate_fp_collision_grad_and_hessian(
//             cudaPol,
//             verts,"x",
//             csPT,nm_csPT,
//             fp_buffer,
//             in_collisionEps,out_collisionEps,
//             (T)1.0,
//             (T)1.0,(T)1.0);

//         TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"xn");
//         TILEVEC_OPS::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());

//         TILEVEC_OPS::assemble_range(cudaPol,fp_buffer,"grad","inds",vtemp,"dir",0,fp_buffer.size());

//         auto scale = get_input2<float>("fp_scale");

//         // auto ompPol = omp_exec();  
//         // constexpr auto omp_space = execspace_e::openmp;
        
//         auto nodalForceVis = std::make_shared<zeno::PrimitiveObject>();
//         auto& spverts = nodalForceVis->verts;
//         spverts.resize(vtemp.size() * 2);
//         auto& splines = nodalForceVis->lines;
//         splines.resize(vtemp.size());

//         // auto scale = get_input2<float>("scale");

//         vtemp = vtemp.clone({zs::memsrc_e::host});
//         ompPol(zs::range(vtemp.size()),
//             [vtemp = proxy<space>({},vtemp),&spverts,&splines,scale] (int vi) mutable {
//                 auto xs = vtemp.template pack<3>("xn",vi);
//                 auto dir = vtemp.template pack<3>("dir",vi);

//                 auto xe = xs + scale * dir;

//                 spverts[vi * 2 + 0] = xs.to_array();
//                 spverts[vi * 2 + 1] = xe.to_array();
//                 splines[vi] = zeno::vec2i(vi * 2 + 0,vi * 2 + 1);               
//         });

//         set_output("FPNodalForceVis",std::move(nodalForceVis));
//         #endif
//     }

// };

// ZENDEFNODE(VisualizeCollision, {{"ZSParticles",{"float","fp_scale","1.0"},{"float","ee_scale","1.0"},{"float","in_collisionEps"},{"float","out_collisionEps"}},
//                                   {
//                                         "collisionFacetVis",
//                                         "colPointFacetPairVis",
//                                         "FPNodalForceVis",
//                                         // "collisionEdgeVis",
//                                         // "colEdgePairVis",
//                                         // "EENodalForceVis",
//                                     },
//                                   {
//                                   },
//                                   {"ZSGeometry"}});


struct VisualizeSelfIntersections : zeno::INode {

    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;
    using vec3 = zs::vec<T, 3>;

    virtual void apply() override {
        using namespace zs;
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        bool is_tet_volume_mesh = zsparticles->category == ZenoParticles::category_e::tet;
        const auto &tris = is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints(); 
        // const auto& points = (*zsparticles)[ZenoParticles::s_surfPointTag];
        const auto& verts = zsparticles->getParticles();

        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();  
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();  
                                                                                                                                                                                                              
        dtiles_t tri_buffer{tris.get_allocator(),{
            {"inds",3},
            {"nrm",3},
            {"he_inds",1}
        },tris.size()};
        dtiles_t verts_buffer{verts.get_allocator(),{
            {"inds",1},
            {"x",3},
            {"X",3},
            {"he_inds",1},
            {"check_pos",3}
        },is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfVertTag].size() : verts.size()};

        TILEVEC_OPS::copy(cudaPol,tris,"he_inds",tri_buffer,"he_inds");
        if(is_tet_volume_mesh) {
            const auto &points = (*zsparticles)[ZenoParticles::s_surfVertTag];
            TILEVEC_OPS::copy(cudaPol,points,"inds",verts_buffer,"inds");
            TILEVEC_OPS::copy(cudaPol,points,"he_inds",verts_buffer,"he_inds");
            topological_sample(cudaPol,points,verts,"x",verts_buffer);
            topological_sample(cudaPol,points,verts,"X",verts_buffer);
            if(verts.hasProperty("check_pos"))
                topological_sample(cudaPol,points,verts,"check_pos",verts_buffer);
            else
                TILEVEC_OPS::copy(cudaPol,verts_buffer,"x",verts_buffer,"check_pos");
            TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
            reorder_topology(cudaPol,points,tri_buffer);

        }else {
            TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
            TILEVEC_OPS::copy(cudaPol,verts,"x",verts_buffer,"x");
            TILEVEC_OPS::copy(cudaPol,verts,"he_inds",verts_buffer,"he_inds");
            cudaPol(zs::range(verts.size()),[
                verts = proxy<cuda_space>({},verts),
                verts_buffer = proxy<cuda_space>({},verts_buffer)] ZS_LAMBDA(int vi) mutable {
                    verts_buffer("inds",vi) = reinterpret_bits<T>(vi);
            });
        }

        // if(!calculate_facet_normal(cudaPol,verts_buffer,"x",tri_buffer,tri_buffer,"nrm")){
        //     throw std::runtime_error("fail updating facet normal");
        // }  

        // zs::Vector<int> nodal_colors{verts_buffer.get_allocator(),verts_buffer.size()};
        // zs::Vector<zs::vec<int,2>> instBuffer{tri_buffer.get_allocator(),tri_buffer.size() * 2};
        // dtiles_t inst_buffer_info{tris.get_allocator(),{
        //     {"pair",2},
        //     {"type",1},
        //     {"its_edge_mark",6},
        //     {"int_points",6}
        // },tris.size() * 2};

        dtiles_t gia_res{verts_buffer.get_allocator(),{
            {"ring_mask",1},
            {"type_mask",1},
            {"color_mask",1},
            {"is_loop_vertex",1}
        },verts_buffer.size()};
        dtiles_t tris_gia_res{tri_buffer.get_allocator(),{
            {"ring_mask",1},
            {"type_mask",1},
            {"color_mask",1},
        },tri_buffer.size()};

        auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        // auto nm_insts = do_global_self_intersection_analysis_on_surface_mesh_info(
        //     cudaPol,verts_buffer,"x",tri_buffer,halfedges,inst_buffer_info,gia_res,false);
        // zs::bht<int,2,int> conn_of_first_ring{halfedges.get_allocator(),halfedges.size()}; 
        auto ring_mask_width = do_global_self_intersection_analysis(cudaPol,
            verts_buffer,"x",tri_buffer,halfedges,gia_res,tris_gia_res);   

        

        // std::cout << "inst_buffer_info : " << std::endl;
        // cudaPol(zs::range(nm_insts),[
        //     inst_buffer_info = proxy<cuda_space>({},inst_buffer_info),
        //     tris = proxy<cuda_space>({},tris),
        //     verts = proxy<cuda_space>({},verts)] ZS_LAMBDA(int isi) mutable {
        //         auto pair = inst_buffer_info.pack(dim_c<2>,"pair",isi,int_c);
        //         printf("pair[%d] : %d %d\n",isi,pair[0],pair[1]);
        // });
        // nm_insts = 0;

        // auto nm_insts = do_global_self_intersection_analysis_on_surface_mesh(cudaPol,
        //     verts_buffer,"x",tri_buffer,instBuffer,nodal_colors);

        // auto nn = conn_of_first_ring.size();
        // if(nm_rings == 1) {

        //     cudaPol(zs::range(halfedges.size()),[
        //         halfedges = proxy<cuda_space>({},halfedges),
        //         tri_buffer = proxy<cuda_space>({},tri_buffer),
        //         verts_buffer = proxy<cuda_space>({},verts_buffer),
        //         conn_of_first_ring = proxy<cuda_space>(conn_of_first_ring)] ZS_LAMBDA(int hi) mutable {
        //             auto broken = halfedges("broken",hi);
        //             if(broken > (T)0.5) {
        //                 auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
        //                 auto tri = tri_buffer.pack(dim_c<3>,"inds",ti,int_c);
        //                 auto local_vid = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
        //                 auto ij = zs::vec<int,2>{tri[local_vid],tri[(local_vid + 1) % 3]};
        //                 auto ji = zs::vec<int,2>{ij[1],ij[0]};
        //                 if(auto no = conn_of_first_ring.query(ij);no >= 0) {
        //                     printf("invalid conn topo detected with ij = [%d %d], and hi = %d\n",ij[0],ij[1],hi);
        //                 }
        //                 if(auto no = conn_of_first_ring.query(ji);no >= 0) {
        //                     printf("invalid conn topo detected with ij = [%d %d], and hi = %d\n",ji[0],ji[1],hi);
        //                 }
        //             }
        //     });
        // }

        // std::cout << "size of conn_of_first_ring : " << nn << std::endl;
        // dtiles_t wire_frame_conn_of_first_ring {verts_buffer.get_allocator(),{
        //     {"x0",3},
        //     {"x1",3}
        // }, nn};
        // cudaPol(zip(range(nn), zs::range(conn_of_first_ring._activeKeys)),[
        //     wire_frame_conn_of_first_ring = proxy<cuda_space>({},wire_frame_conn_of_first_ring),
        //     verts_buffer = proxy<cuda_space>({},verts_buffer)] ZS_LAMBDA(auto i, const auto& ij) mutable {
        //         wire_frame_conn_of_first_ring.tuple(dim_c<3>,"x0",i) = verts_buffer.pack(dim_c<3>,"check_pos",ij[0]);
        //         wire_frame_conn_of_first_ring.tuple(dim_c<3>,"x1",i) = verts_buffer.pack(dim_c<3>,"check_pos",ij[1]);
        // });

        // wire_frame_conn_of_first_ring = wire_frame_conn_of_first_ring.clone({zs::memsrc_e::host});
        // auto wire_frame_conn_fr_vis = std::make_shared<zeno::PrimitiveObject>();
        // auto& wire_verts = wire_frame_conn_fr_vis->verts;
        // auto& wire_lines = wire_frame_conn_fr_vis->lines;
        // wire_verts.resize(wire_frame_conn_of_first_ring.size() * 2);
        // wire_lines.resize(wire_frame_conn_of_first_ring.size());

        // ompPol(zs::range(wire_frame_conn_of_first_ring.size()),[
        //     wire_frame_conn_of_first_ring = proxy<omp_space>({},wire_frame_conn_of_first_ring),
        //     &wire_verts,&wire_lines] (int wi) mutable {
        //         auto x0 = wire_frame_conn_of_first_ring.pack(dim_c<3>,"x0",wi);
        //         auto x1 = wire_frame_conn_of_first_ring.pack(dim_c<3>,"x1",wi);
        //         wire_verts[wi * 2 + 0] = x0.to_array();
        //         wire_verts[wi * 2 + 1] = x1.to_array();
        //         wire_lines[wi] = zeno::vec2i{wi * 2 + 0,wi * 2 + 1};
        // });
        // set_output("wire_fr_vis",std::move(wire_frame_conn_fr_vis));


        // dtiles_t broken_edges_buffer{halfedges.get_allocator(),{
        //     {"x0",3},
        //     {"x1",3},
        //     {"clr",3},
        // },halfedges.size()};
        // cudaPol(zs::range(halfedges.size()),[
        //     halfedges = proxy<cuda_space>({},halfedges),
        //     broken_edges_buffer = proxy<cuda_space>({},broken_edges_buffer),
        //     tri_buffer = proxy<cuda_space>({},tri_buffer),
        //     verts_buffer = proxy<cuda_space>({},verts_buffer)] ZS_LAMBDA(int hi) mutable {
        //         auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
        //         auto local_vid = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
        //         auto tri = tri_buffer.pack(dim_c<3>,"inds",ti,int_c);
        //         broken_edges_buffer.tuple(dim_c<3>,"x0",hi) = verts_buffer.pack(dim_c<3>,"x",tri[local_vid]);
        //         broken_edges_buffer.tuple(dim_c<3>,"x1",hi) = verts_buffer.pack(dim_c<3>,"x",tri[(local_vid + 1) % 3]);
        //         auto broken = halfedges("broken",hi);
        //         // if(broken > (T)0.5)
        //         //     printf("find broken edges : %d %f\n",hi,(float)broken);
        //         broken_edges_buffer.tuple(dim_c<3>,"clr",hi) = broken > (T)0.5 ? zs::vec<T,3>{1.0,0.0,0.0} : zs::vec<T,3>{1.0,1.0,1.0};
        //         // if(broken > (T)0.5) {

        //         // }else {
        //         //     broken_edges_buffer.tuple(dim_c<3>,"x0",hi) = zs::vec<T,3>::zeros();
        //         //     broken_edges_buffer.tuple(dim_c<3>,"x1",hi) = zs::vec<T,3>::zeros();
        //         // }
        // });           
        // broken_edges_buffer = broken_edges_buffer.clone({zs::memsrc_e::host});
        // auto be_vis = std::make_shared<zeno::PrimitiveObject>();
        // auto& be_vis_verts = be_vis->verts;
        // auto& be_vis_lines = be_vis->lines;
        // be_vis_verts.resize(broken_edges_buffer.size() * 2);
        // be_vis_lines.resize(broken_edges_buffer.size());
        // auto& be_vis_clrs = be_vis->add_attr<zeno::vec3f>("clr");
        // ompPol(zs::range(broken_edges_buffer.size()),[
        //     broken_edges_buffer = proxy<omp_space>({},broken_edges_buffer),
        //     &be_vis_verts,&be_vis_lines,&be_vis_clrs] (int hi) mutable {
        //         auto x0 = broken_edges_buffer.pack(dim_c<3>,"x0",hi);
        //         auto x1 = broken_edges_buffer.pack(dim_c<3>,"x1",hi);
        //         auto clr = broken_edges_buffer.pack(dim_c<3>,"clr",hi);
        //         be_vis_verts[hi * 2 + 0] = x0.to_array();
        //         be_vis_verts[hi * 2 + 1] = x1.to_array();
        //         be_vis_clrs[hi * 2 + 0] = clr.to_array();
        //         be_vis_clrs[hi * 2 + 1] = clr.to_array();
        //         be_vis_lines[hi] = zeno::vec2i{hi * 2 + 0,hi *2 + 1};
        // });

        // set_output("be_vis",std::move(be_vis));


        dtiles_t flood_region{verts_buffer.get_allocator(),{
            {"x",3}
        },(size_t)verts_buffer.size()};
        TILEVEC_OPS::copy(cudaPol,verts_buffer,"x",flood_region,"x");
        // verts_buffer = verts_buffer.clone({zs::memsrc_e::host});
        // tri_buffer = tri_buffer.clone({zs::memsrc_e::host});
        flood_region = flood_region.clone({zs::memsrc_e::host});
        gia_res = gia_res.clone({zs::memsrc_e::host});

        auto flood_region_vis = std::make_shared<zeno::PrimitiveObject>();
        flood_region_vis->resize(verts.size());
        auto& flood_region_verts = flood_region_vis->verts;
        auto& flood_region_mark = flood_region_vis->add_attr<float>("flood");
        auto& is_corner_mark = flood_region_vis->add_attr<float>("is_loop_vertex");
        
        ompPol(zs::range(verts_buffer.size()),[
            &flood_region_verts,
            &flood_region_mark,
            &is_corner_mark,
            ring_mask_width = ring_mask_width,
            flood_region = proxy<omp_space>({},flood_region),
            gia_res = proxy<omp_space>({},gia_res)] (int vi) mutable {
                auto p = flood_region.pack(dim_c<3>,"x",vi);
                flood_region_verts[vi] = p.to_array();
                int ring_mask = 0;
                bool is_corner = false;
                for(int i = 0;i != ring_mask_width;++i) {
                    ring_mask |= zs::reinterpret_bits<int>(gia_res("ring_mask",vi * ring_mask_width + i));
                    if(gia_res("is_loop_vertex",vi) > (T)0.5)
                        is_corner = true;
                }


                flood_region_mark[vi] = ring_mask == 0 ? (float)0.0 : (float)1.0;
                // auto is_corner = gia_res("is_loop_vertex",vi);
                is_corner_mark[vi] = is_corner ? (T)1.0 : (T)0.0;
        });
        set_output("flood_region",std::move(flood_region_vis));


        // dtiles_t self_intersect_buffer{tris.get_allocator(),{
        //     {"a0",3},{"A0",3},
        //     {"a1",3},{"A1",3},
        //     {"a2",3},{"A2",3},
        //     {"b0",3},{"B0",3},
        //     {"b1",3},{"B1",3},
        //     {"b2",3},{"B2",3},
        //     // {"p0",3},{"p1",3}
        // },(size_t)nm_insts};
        // cudaPol(zs::range(nm_insts),[
        //     // instBuffer = proxy<cuda_space>(instBuffer),
        //     inst_buffer_info = proxy<cuda_space>({},inst_buffer_info),
        //     verts_buffer = proxy<cuda_space>({},verts_buffer),
        //     self_intersect_buffer = proxy<cuda_space>({},self_intersect_buffer),
        //     tri_buffer = proxy<cuda_space>({},tri_buffer)] ZS_LAMBDA(int sti) mutable {
        //         auto tpair = inst_buffer_info.pack(dim_c<2>,"pair",sti,int_c);
        //         auto ta = tpair[0];
        //         auto tb = tpair[1];

        //         // auto ints_p = inst_buffer_info.pack(dim_c<6>,"int_points",sti);
        //         // self_intersect_buffer.tuple(dim_c<3>,"p0",sti) = zs::vec<T,3>{ints_p[0],ints_p[1],ints_p[2]};
        //         // self_intersect_buffer.tuple(dim_c<3>,"p1",sti) = zs::vec<T,3>{ints_p[3],ints_p[4],ints_p[5]};
        //         // auto ta = instBuffer[sti][0];
        //         // auto tb = instBuffer[sti][1];

        //         auto triA = tri_buffer.pack(dim_c<3>,"inds",ta,int_c);
        //         auto triB = tri_buffer.pack(dim_c<3>,"inds",tb,int_c);
        //         self_intersect_buffer.tuple(dim_c<3>,"a0",sti) = verts_buffer.pack(dim_c<3>,"x",triA[0]);
        //         self_intersect_buffer.tuple(dim_c<3>,"a1",sti) = verts_buffer.pack(dim_c<3>,"x",triA[1]);
        //         self_intersect_buffer.tuple(dim_c<3>,"a2",sti) = verts_buffer.pack(dim_c<3>,"x",triA[2]);

        //         self_intersect_buffer.tuple(dim_c<3>,"b0",sti) = verts_buffer.pack(dim_c<3>,"x",triB[0]);
        //         self_intersect_buffer.tuple(dim_c<3>,"b1",sti) = verts_buffer.pack(dim_c<3>,"x",triB[1]);
        //         self_intersect_buffer.tuple(dim_c<3>,"b2",sti) = verts_buffer.pack(dim_c<3>,"x",triB[2]);

        //         self_intersect_buffer.tuple(dim_c<3>,"A0",sti) = verts_buffer.pack(dim_c<3>,"x",triA[0]);
        //         self_intersect_buffer.tuple(dim_c<3>,"A1",sti) = verts_buffer.pack(dim_c<3>,"x",triA[1]);
        //         self_intersect_buffer.tuple(dim_c<3>,"A2",sti) = verts_buffer.pack(dim_c<3>,"x",triA[2]);

        //         self_intersect_buffer.tuple(dim_c<3>,"B0",sti) = verts_buffer.pack(dim_c<3>,"x",triB[0]);
        //         self_intersect_buffer.tuple(dim_c<3>,"B1",sti) = verts_buffer.pack(dim_c<3>,"x",triB[1]);
        //         self_intersect_buffer.tuple(dim_c<3>,"B2",sti) = verts_buffer.pack(dim_c<3>,"x",triB[2]);
        // });

        // self_intersect_buffer = self_intersect_buffer.clone({zs::memsrc_e::host});

        // auto st_fact_vis = std::make_shared<zeno::PrimitiveObject>();
        // auto& st_verts = st_fact_vis->verts;
        // auto& st_tris = st_fact_vis->tris;
        // st_verts.resize(self_intersect_buffer.size() * 6);
        // st_tris.resize(self_intersect_buffer.size() * 2);

        // ompPol(zs::range(nm_insts),[
        //     &st_verts,&st_tris,self_intersect_buffer = proxy<omp_space>({},self_intersect_buffer)] (int sti) mutable {
        //         st_verts[sti * 6 + 0] = self_intersect_buffer.pack(dim_c<3>,"a0",sti).to_array();
        //         st_verts[sti * 6 + 1] = self_intersect_buffer.pack(dim_c<3>,"a1",sti).to_array();
        //         st_verts[sti * 6 + 2] = self_intersect_buffer.pack(dim_c<3>,"a2",sti).to_array();
        //         st_verts[sti * 6 + 3] = self_intersect_buffer.pack(dim_c<3>,"b0",sti).to_array();
        //         st_verts[sti * 6 + 4] = self_intersect_buffer.pack(dim_c<3>,"b1",sti).to_array();
        //         st_verts[sti * 6 + 5] = self_intersect_buffer.pack(dim_c<3>,"b2",sti).to_array();

        //         st_tris[sti * 2 + 0] = zeno::vec3i(sti * 6 + 0,sti * 6 + 1,sti * 6 + 2);
        //         st_tris[sti * 2 + 1] = zeno::vec3i(sti * 6 + 3,sti * 6 + 4,sti * 6 + 5);
        // });   

        // // std::cout << "nm_insts : " << nm_insts << std::endl;
        // set_output("st_facet_vis",std::move(st_fact_vis));

        // auto st_ring_vis = std::make_shared<zeno::PrimitiveObject>();
        // auto& its_ring_verts = st_ring_vis->verts;
        // auto& its_ring_lines = st_ring_vis->lines;
        // its_ring_verts.resize(nm_insts * 2);
        // its_ring_lines.resize(nm_insts);
        // ompPol(zs::range(nm_insts),[
        //     &its_ring_verts,&its_ring_lines,self_intersect_buffer = proxy<omp_space>({},self_intersect_buffer)] (int sti) mutable {
        //     auto p0 = self_intersect_buffer.pack(dim_c<3>,"p0",sti);
        //     auto p1 = self_intersect_buffer.pack(dim_c<3>,"p1",sti);
        //     its_ring_verts[sti * 2 + 0] = p0.to_array();
        //     its_ring_verts[sti * 2 + 1] = p1.to_array();
        //     its_ring_lines[sti] = zeno::vec2i{sti * 2 + 0,sti * 2 + 1};
        // });

        // set_output("st_ring_vis",std::move(st_ring_vis));

        // auto st_facet_rest_vis = std::make_shared<zeno::PrimitiveObject>();
        // auto& st_rest_verts = st_facet_rest_vis->verts;
        // auto& st_rest_tris = st_facet_rest_vis->tris;
        // st_rest_verts.resize(self_intersect_buffer.size() * 6);
        // st_rest_tris.resize(self_intersect_buffer.size() * 2);
        // ompPol(zs::range(nm_insts),[
        //     &st_rest_verts,&st_rest_tris,self_intersect_buffer = proxy<omp_space>({},self_intersect_buffer)] (int sti) mutable {
        //         st_rest_verts[sti * 6 + 0] = self_intersect_buffer.pack(dim_c<3>,"A0",sti).to_array();
        //         st_rest_verts[sti * 6 + 1] = self_intersect_buffer.pack(dim_c<3>,"A1",sti).to_array();
        //         st_rest_verts[sti * 6 + 2] = self_intersect_buffer.pack(dim_c<3>,"A2",sti).to_array();
        //         st_rest_verts[sti * 6 + 3] = self_intersect_buffer.pack(dim_c<3>,"B0",sti).to_array();
        //         st_rest_verts[sti * 6 + 4] = self_intersect_buffer.pack(dim_c<3>,"B1",sti).to_array();
        //         st_rest_verts[sti * 6 + 5] = self_intersect_buffer.pack(dim_c<3>,"B2",sti).to_array();

        //         st_rest_tris[sti * 2 + 0] = zeno::vec3i(sti * 6 + 0,sti * 6 + 1,sti * 6 + 2);
        //         st_rest_tris[sti * 2 + 1] = zeno::vec3i(sti * 6 + 3,sti * 6 + 4,sti * 6 + 5);
        // });  
        // set_output("st_facet_rest_vis",std::move(st_facet_rest_vis));

        // dtiles_t st_pair_buffer{tris.get_allocator(),{
        //     {"x0",3},
        //     {"x1",3}
        // },(std::size_t)nm_insts};    
        // cudaPol(zs::range(nm_insts),[
        //     inst_buffer_info = proxy<cuda_space>({},inst_buffer_info),
        //     // instBuffer = proxy<cuda_space>(instBuffer),
        //     st_pair_buffer = proxy<cuda_space>({},st_pair_buffer),
        //     verts = proxy<cuda_space>({},verts_buffer),
        //     tris = proxy<cuda_space>({},tri_buffer)] ZS_LAMBDA(int sti) mutable {
        //         auto tpair = inst_buffer_info.pack(dim_c<2>,"pair",sti,int_c);
        //         auto ta = tpair[0];
        //         auto tb = tpair[1];
        //         // auto ta = instBuffer[sti][0];
        //         // auto tb = instBuffer[sti][1];


        //         auto triA = tris.pack(dim_c<3>,"inds",ta,int_c);
        //         auto triB = tris.pack(dim_c<3>,"inds",tb,int_c);

        //         auto x0 = vec3::zeros();
        //         auto x1 = vec3::zeros();

        //         for(int i = 0;i != 3;++i) {
        //             x0 += verts.pack(dim_c<3>,"x",triA[i]) / (T)3.0;
        //             x1 += verts.pack(dim_c<3>,"x",triB[i]) / (T)3.0;
        //         }

        //         st_pair_buffer.tuple(dim_c<3>,"x0",sti) = x0.to_array();
        //         st_pair_buffer.tuple(dim_c<3>,"x1",sti) = x1.to_array();
        // });

        // st_pair_buffer = st_pair_buffer.clone({zs::memsrc_e::host});
        // auto st_pair_vis = std::make_shared<zeno::PrimitiveObject>();
        // auto& st_pair_verts = st_pair_vis->verts;
        // auto& st_pair_lines = st_pair_vis->lines;
        // st_pair_verts.resize(st_pair_buffer.size() * 2);
        // st_pair_lines.resize(st_pair_buffer.size());    

        // ompPol(zs::range(st_pair_buffer.size()),[
        //     st_pair_buffer = proxy<omp_space>({},st_pair_buffer),
        //     &st_pair_verts,&st_pair_lines] (int spi) mutable {
        //         auto x0 = st_pair_buffer.pack(dim_c<3>,"x0",spi);
        //         auto x1 = st_pair_buffer.pack(dim_c<3>,"x1",spi);
        //         st_pair_verts[spi * 2 + 0] = x0.to_array();
        //         st_pair_verts[spi * 2 + 1] = x1.to_array();
        //         st_pair_lines[spi] = zeno::vec2i{spi * 2 + 0,spi * 2 + 1};
        // });

        // set_output("st_pair_vis",std::move(st_pair_vis));

        // dtiles_t corner_verts_buffer{gia_res.get_allocator()}
    }
};

ZENDEFNODE(VisualizeSelfIntersections, {{"zsparticles"},
                                  {
                                        // "st_ring_vis",
                                        // "st_facet_rest_vis",
                                        // "st_facet_vis",                                                           
                                        "flood_region",
                                        // "be_vis"
                                        // "wire_fr_vis"
                                    },
                                  {
                                    
                                  },
                                  {"ZSGeometry"}});



struct VisualizeIntersections : zeno::INode {

    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;
    using vec3 = zs::vec<T, 3>;
    using table_vec2i_type = zs::bht<int,2,int>;
    using table_int_type = zs::bht<int,1,int>;

    virtual void apply() override { 
        using namespace zs;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();  
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();  

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        bool is_tet_volume_mesh = zsparticles->category == ZenoParticles::category_e::tet;
        const auto &tris = is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints(); 
        // const auto& points = (*zsparticles)[ZenoParticles::s_surfPointTag];
        const auto& verts = zsparticles->getParticles();
        auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];

        // auto kinematics = get_input<ListObject>("kinematics")->get2<ZenoParticles*>();
        // auto kinematics = RETRIEVE_OBJECT_PTRS(ZenoParticles,"kinematics");
        auto kinematic = get_input<ZenoParticles>("kinematics");
        auto& kverts = kinematic->getParticles();
        const auto& ktris = kinematic->getQuadraturePoints();
        const auto& khalfedges = (*kinematic)[ZenoParticles::s_surfHalfEdgeTag];
        // if(!verts.hasProperty("flood"))
        //     verts.append_channels(cudaPol,{{"flood",1}});
        // TILEVEC_OPS::fill(cudaPol,verts,"flood",(T)0.0);

        dtiles_t tri_buffer{tris.get_allocator(),{
            {"inds",3},
            {"he_inds",1}
        },tris.size()};
        dtiles_t verts_buffer{verts.get_allocator(),{
            {"inds",1},
            {"x",3},
            {"flood",1},
            // {"he_inds",1},
        },is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfVertTag].size() : verts.size()};
        TILEVEC_OPS::fill(cudaPol,verts_buffer,"flood",(T)0.0);
        TILEVEC_OPS::copy(cudaPol,tris,"he_inds",tri_buffer,"he_inds");
        if(is_tet_volume_mesh) {
            const auto &points = (*zsparticles)[ZenoParticles::s_surfVertTag];
            TILEVEC_OPS::copy(cudaPol,points,"inds",verts_buffer,"inds");
            // TILEVEC_OPS::copy(cudaPol,points,"he_inds",verts_buffer,"he_inds");
            // TILEVEC_OPS::copy(cudaPol,points,"inds",verts_buffer,"inds");
            std::cout << "do_topological_sample" << std::endl;
            topological_sample(cudaPol,points,verts,"x",verts_buffer);
            // topological_sample(cudaPol,points,verts,"X",verts_buffer);
            TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
            std::cout << "do_reorder_topology" << std::endl;
            reorder_topology(cudaPol,points,tri_buffer);
        }else {
            TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
            TILEVEC_OPS::copy(cudaPol,verts,"x",verts_buffer,"x");
            // TILEVEC_OPS::copy(cudaPol,verts,"he_inds",verts_buffer,"he_inds");
            cudaPol(zs::range(verts.size()),[
                verts = proxy<cuda_space>({},verts),
                verts_buffer = proxy<cuda_space>({},verts_buffer)] ZS_LAMBDA(int vi) mutable {
                    verts_buffer("inds",vi) = reinterpret_bits<T>(vi);
            });
        }

        dtiles_t kverts_buffer{verts.get_allocator(),{
            {"flood",1},
            {"x",3}
        },kverts.size()};
        TILEVEC_OPS::fill(cudaPol,kverts_buffer,"flood",(T)0.0);
        TILEVEC_OPS::copy(cudaPol,kverts,"x",kverts_buffer,"x");

        zs::Vector<int> gia_res{verts_buffer.get_allocator(),0};

        zs::Vector<int> tris_gia_res{tri_buffer.get_allocator(),0};


        bool use_zs_interior = get_input2<bool>("use_zsparticles_interior");
        bool use_kin_interior = get_input2<bool>("use_kinematic_interior");

        // TODO: The must ExcludeMark is not include
        {
            std::cout << "do_global_intersection_analysis_with_connected_manifolds" << std::endl;

            auto ring_mask_width = do_global_intersection_analysis_with_connected_manifolds(cudaPol,
                verts_buffer,"x",tri_buffer,halfedges,use_zs_interior,
                kverts_buffer,"x",ktris,khalfedges,use_kin_interior,
                gia_res,tris_gia_res);

            std::cout << "finish do_global_intersection_analysis_with_connected_manifolds" << std::endl;

            zs::Vector<int> nmFloodVerts{verts_buffer.get_allocator(),1};
            nmFloodVerts.setVal(0);

            cudaPol(zs::range(verts_buffer.size()),[
                ring_mask_width = ring_mask_width,
                nmFloodVerts = proxy<cuda_space>(nmFloodVerts),
                verts_buffer = proxy<cuda_space>({},verts_buffer),
                gia_res = proxy<cuda_space>(gia_res)] ZS_LAMBDA(int vi) mutable {
                    for(int i = 0;i != ring_mask_width;++i) {
                        auto ring_mask = gia_res[vi * ring_mask_width + i];
                        if(ring_mask > 0) {
                            verts_buffer("flood",vi) = (T)1.0;
                            atomic_add(zs::exec_cuda,&nmFloodVerts[0],(int)1);
                            return;
                        }
                    }
            });

            std::cout << "nm_flood_vertices : " << nmFloodVerts.getVal(0) << std::endl;

            auto k_offset = verts_buffer.size();
            cudaPol(zs::range(kverts.size()),[
                ring_mask_width = ring_mask_width,
                kverts_buffer = proxy<cuda_space>({},kverts_buffer),
                gia_res = proxy<cuda_space>(gia_res),
                k_offset = k_offset] ZS_LAMBDA(int kvi) mutable {
                    for(int i = 0;i != ring_mask_width;++i) {
                        auto ring_mask = gia_res[(kvi + k_offset) * ring_mask_width + i];
                        if(ring_mask > 0){
                            kverts_buffer("flood",kvi) = (T)1.0;
                            return;
                        }
                    }
            });
        }
        
        std::cout << "finish marking" << std::endl;
        verts_buffer = verts_buffer.clone({zs::memsrc_e::host});
        kverts_buffer = kverts_buffer.clone({zs::memsrc_e::host});

        std::cout << "flood_dynamic" << std::endl;

        auto flood_dynamic = std::make_shared<zeno::PrimitiveObject>();
        auto& dyn_verts = flood_dynamic->verts;
        dyn_verts.resize(verts_buffer.size());
        auto& dfloods = flood_dynamic->add_attr<T>("flood");

        ompPol(zs::range(verts_buffer.size()),[
            verts_buffer = proxy<omp_space>({},verts_buffer),
            &dyn_verts,&dfloods] (int vi) mutable {
                auto p = verts_buffer.pack(dim_c<3>,"x",vi);
                dyn_verts[vi] = p.to_array();
                auto flood = verts_buffer("flood",vi);
                dfloods[vi] = flood > (T)0.5 ? (T)1.0 : (T)0.0;
        });
        
        std::cout << "flood_kinematic" << std::endl;

        auto flood_kinematic = std::make_shared<zeno::PrimitiveObject>();
        auto& kin_verts = flood_kinematic->verts;
        kin_verts.resize(kverts_buffer.size());
        auto& kfloods = flood_kinematic->add_attr<T>("flood");
        ompPol(zs::range(kverts_buffer.size()),[
            kverts_buffer = proxy<omp_space>({},kverts_buffer),
            &kin_verts,&kfloods] (int kvi) mutable {
                auto p = kverts_buffer.pack(dim_c<3>,"x",kvi);
                kin_verts[kvi] = p.to_array();
                auto flood = kverts_buffer("flood",kvi);
                kfloods[kvi] = flood > (T)0.5 ? (T)1.0 : (T)0.0;
        });

        std::cout << "output" << std::endl;

        set_output("flood_dynamic",std::move(flood_dynamic));
        set_output("flood_kinematic",std::move(flood_kinematic));
    }
};


ZENDEFNODE(VisualizeIntersections, {{"zsparticles",
                                        "kinematics",
                                        {"bool","use_zsparticles_interior","1"},    
                                        {"bool","use_kinematic_interior","1"}
                                    },
                                  {
                                        "flood_dynamic",
                                        "flood_kinematic",
                                        // "dyn_edges_vis",
                                        // "kin_edges_vis"
                                    },
                                  {
                                    
                                  },
                                  {"ZSGeometry"}});

struct VisualizeIntersections2 : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;
    using vec3 = zs::vec<T, 3>;
    using table_vec2i_type = zs::bht<int,2,int>;
    using table_int_type = zs::bht<int,1,int>;

    virtual void apply() override { 
        using namespace zs;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();  
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();  

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        bool is_tet_volume_mesh = zsparticles->category == ZenoParticles::category_e::tet;
        const auto &tris = (*zsparticles)[ZenoParticles::s_surfTriTag]; 
        const auto& tets = zsparticles->getQuadraturePoints();
        const auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
        auto& verts = zsparticles->getParticles();
        auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        // const auto& points = 

        auto in_collisionEps = get_input2<T>("in_collisionEps");
        auto out_collisionEps = get_input2<T>("out_collisionEps");

        // auto kinematics = get_input<ListObject>("kinematics")->get2<ZenoParticles*>();
        auto kinematics = RETRIEVE_OBJECT_PTRS(ZenoParticles,"kinematics");
        zs::bht<int,2,int> csPT{verts.get_allocator(),100000};
        csPT.reset(cudaPol,true);
        zs::Vector<int> csPTOffsets{verts.get_allocator(),kinematics.size()};
        auto nm_csPT = COLLISION_UTILS::do_tetrahedra_surface_mesh_and_kinematic_boundary_collision_detection(cudaPol,
            kinematics,
            verts,"x",
            tets,
            points,tris,
            halfedges,
            out_collisionEps,
            in_collisionEps,
            csPT,
            csPTOffsets,
            true);

        int nm_kverts = 0;
        for(auto kinematic : kinematics) {
            nm_kverts += kinematic->getParticles().size();
        }

        dtiles_t flood_dyn{verts.get_allocator(),{
            {"x",3},
            {"flood",1}
        },verts.size()};
        dtiles_t all_kverts_buffer{verts.get_allocator(),{
            {"x",3},
            {"flood",1}
        },(size_t)nm_kverts};

        TILEVEC_OPS::copy(cudaPol,verts,"x",flood_dyn,"x");
        TILEVEC_OPS::copy(cudaPol,verts,"flood",flood_dyn,"flood");

        int voffset = 0;
        for(auto kinematic : kinematics) {
            const auto& kverts = kinematic->getParticles();
            TILEVEC_OPS::copy(cudaPol,kverts,"x",all_kverts_buffer,"x",voffset);
            TILEVEC_OPS::copy(cudaPol,kverts,"flood",all_kverts_buffer,"flood",voffset);
            voffset += kverts.size();
        }

        dtiles_t csPTBuffer{verts.get_allocator(),{
            {"x0",3},
            {"x1",3}
        },(size_t)csPT.size()};
        cudaPol(zip(zs::range(csPT.size()),zs::range(csPT._activeKeys)),[
            tris = proxy<cuda_space>({},tris),
            verts = proxy<cuda_space>({},verts),
            csPTBuffer = proxy<cuda_space>({},csPTBuffer),
            all_kverts_buffer = proxy<cuda_space>({},all_kverts_buffer)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto kvi = pair[0];
                auto ti = pair[1];
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                zs::vec<T,3> tV[3] = {};
                for(int i = 0;i != 3;++i)
                    tV[i] = verts.pack(dim_c<3>,"x",tri[i]);
                
                auto tC = zs::vec<T,3>::zeros();
                for(int i = 0;i != 3;++i)   
                    tC += tV[i] / (T)3.0;
                
                auto kv = all_kverts_buffer.pack(dim_c<3>,"x",kvi);

                csPTBuffer.tuple(dim_c<3>,"x0",ci) = kv;
                csPTBuffer.tuple(dim_c<3>,"x1",ci) = tC;
        });

        flood_dyn = flood_dyn.clone({memsrc_e::host});
        all_kverts_buffer = all_kverts_buffer.clone({memsrc_e::host});
        csPTBuffer = csPTBuffer.clone({memsrc_e::host});

        auto flood_dyn_vis = std::make_shared<zeno::PrimitiveObject>();
        flood_dyn_vis->resize(flood_dyn.size());
        auto& flood_dyn_verts = flood_dyn_vis->verts;
        auto& flood_dyn_tags = flood_dyn_verts.add_attr<T>("flood");
        ompPol(zs::range(flood_dyn.size()),[
            flood_dyn = proxy<omp_space>({},flood_dyn),
            &flood_dyn_verts,&flood_dyn_tags] (int vi) mutable {
                auto pv = flood_dyn.pack(dim_c<3>,"x",vi);
                auto flood = flood_dyn("flood",vi);
                flood_dyn_verts[vi] = pv.to_array();
                flood_dyn_tags[vi] = flood;
        });
        set_output("flood_dynamic",std::move(flood_dyn_vis));

        auto flood_kin_vis = std::make_shared<zeno::PrimitiveObject>();
        flood_kin_vis->resize(all_kverts_buffer.size());
        auto& flood_kin_verts = flood_kin_vis->verts;
        auto& flood_kin_tags = flood_kin_verts.add_attr<T>("flood");
        ompPol(zs::range(all_kverts_buffer.size()),[
            all_kverts_buffer = proxy<omp_space>({},all_kverts_buffer),
            &flood_kin_verts,&flood_kin_tags] (int kvi) mutable {
                auto kv = all_kverts_buffer.pack(dim_c<3>,"x",kvi);
                auto flood =all_kverts_buffer("flood",kvi);
                flood_kin_verts[kvi] = kv.to_array();
                flood_kin_tags[kvi] = flood;
        });
        set_output("flood_kinematic",std::move(flood_kin_vis));

        auto csPTVis = std::make_shared<zeno::PrimitiveObject>();
        auto& csPT_verts = csPTVis->verts;
        csPT_verts.resize(csPTBuffer.size() * 2);
        auto& csPT_lines = csPTVis->lines;
        csPT_lines.resize(csPTBuffer.size());

        ompPol(zs::range(csPTBuffer.size()),[
            csPTBuffer = proxy<omp_space>({},csPTBuffer),
            &csPT_verts,&csPT_lines] (int ci) mutable {
                auto x0 = csPTBuffer.pack(dim_c<3>,"x0",ci);
                auto x1 = csPTBuffer.pack(dim_c<3>,"x1",ci);
                csPT_verts[ci * 2 + 0] = x0.to_array();
                csPT_verts[ci * 2 + 1] = x1.to_array();
                csPT_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });
        set_output("cspt_vis",std::move(csPTVis));
    }
};

ZENDEFNODE(VisualizeIntersections2, {{
                                        "zsparticles",
                                        "kinematics",
                                        {"float","out_collisionEps","0.1"},
                                        {"float","in_collisionEps","0.1"},    
                                    },
                                  {
                                        "flood_dynamic",
                                        "flood_kinematic",
                                        "cspt_vis",
                                        // "kin_edges_vis"
                                    },
                                  {
                                    
                                  },
                                  {"ZSGeometry"}});


struct VisualizeCollision2 : zeno::INode {
    using T = float;

    virtual void apply() override {
        using namespace zs;
        using dtiles_t = typename ZenoParticles::particles_t;

        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();  
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();  

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto& verts = zsparticles->getParticles();
        const auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
        const auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
        const auto& tets = zsparticles->getQuadraturePoints();
        auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        const auto& halffacets = (*zsparticles)[ZenoParticles::s_tetHalfFacetTag];

        zs::bht<int,2,int> csPT{verts.get_allocator(),10000};
        auto out_collisionEps = get_input2<float>("out_collisionEps");
        auto in_collisionEps = get_input2<float>("in_collisionEps");


        dtiles_t verts_buffer{verts.get_allocator(),{
            {"x",3},
            {"flood",1},
            {"ring_mask",1}
        },verts.size()};
        TILEVEC_OPS::copy(cudaPol,verts,"x",verts_buffer,"x");

        COLLISION_UTILS::do_tetrahedra_surface_tris_and_points_self_collision_detection(
            cudaPol,verts_buffer,"x",
            tets,
            points,tris,
            halfedges,halffacets,
            out_collisionEps,
            in_collisionEps,
            csPT,true);

        auto nm_ints = csPT.size();
        std::cout << "nm_ints : " << nm_ints << std::endl;

        dtiles_t tris_buffer{tris.get_allocator(),{
            {"x0",3},
            {"x1",3},
            {"x2",3}
        },nm_ints};

        dtiles_t points_buffer{points.get_allocator(),{
            {"x0",3}
        },nm_ints};

        dtiles_t lines_buffer{points.get_allocator(),{
            {"x0",3},
            {"x1",3}
        },nm_ints};


        cudaPol(zip(zs::range(csPT.size()),zs::range(csPT._activeKeys)),[
            tris_buffer = proxy<cuda_space>({},tris_buffer),
            points_buffer = proxy<cuda_space>({},points_buffer),
            lines_buffer = proxy<cuda_space>({},lines_buffer),
            verts = proxy<cuda_space>({},verts),
            points = proxy<cuda_space>({},points),
            tris = proxy<cuda_space>({},tris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto pi = pair[0];
                auto ti = pair[1];
                auto vi = zs::reinterpret_bits<int>(points("inds",pi));
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                auto p = verts.pack(dim_c<3>,"x",vi);
                zs::vec<T,3> tV[3] = {};
                for(int i = 0;i != 3;++i)
                    tV[i] = verts.pack(dim_c<3>,"x",tri[i]);

                auto tC = zs::vec<T,3>::zeros();
                for(int i = 0;i != 3;++i)
                    tC += tV[i] / (T)3.0;

                tris_buffer.tuple(dim_c<3>,"x0",ci) = tV[0];
                tris_buffer.tuple(dim_c<3>,"x1",ci) = tV[1];
                tris_buffer.tuple(dim_c<3>,"x2",ci) = tV[2];

                points_buffer.tuple(dim_c<3>,"x0",ci) = p;

                lines_buffer.tuple(dim_c<3>,"x0",ci) = p;
                lines_buffer.tuple(dim_c<3>,"x1",ci) = tC;
        });

        tris_buffer = tris_buffer.clone({memsrc_e::host});
        points_buffer = points_buffer.clone({memsrc_e::host});
        lines_buffer = lines_buffer.clone({memsrc_e::host});
        verts_buffer = verts_buffer.clone({memsrc_e::host});

        auto flood_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& flood_verts = flood_vis->verts;
        flood_verts.resize(verts_buffer.size());
        auto& flood_tag = flood_vis->add_attr<float>("flood");
        ompPol(zs::range(verts_buffer.size()),[
            verts_buffer = proxy<omp_space>({},verts_buffer),
            &flood_verts,&flood_tag] (int vi) mutable {
                auto v = verts_buffer.pack(dim_c<3>,"x",vi);
                flood_verts[vi] = v.to_array();
                flood_tag[vi] = verts_buffer("flood",vi);
        });
        set_output("flood_vis",std::move(flood_vis));

        auto tris_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& tris_vis_verts = tris_vis->verts;
        auto& tris_vis_tris = tris_vis->tris;
        tris_vis_verts.resize(nm_ints * 3);
        tris_vis_tris.resize(nm_ints);
        ompPol(zs::range(nm_ints),[
            tris_buffer = proxy<omp_space>({},tris_buffer),
            &tris_vis_verts,&tris_vis_tris] (int ci) mutable {
                auto x0 = tris_buffer.pack(dim_c<3>,"x0",ci);
                auto x1 = tris_buffer.pack(dim_c<3>,"x1",ci);
                auto x2 = tris_buffer.pack(dim_c<3>,"x2",ci);

                tris_vis_verts[ci * 3 + 0] = x0.to_array();
                tris_vis_verts[ci * 3 + 1] = x1.to_array();
                tris_vis_verts[ci * 3 + 2] = x2.to_array();

                tris_vis_tris[ci] = zeno::vec3i{ci * 3 + 0,ci * 3 + 1,ci * 3 + 2};
        });
        set_output("tris_vis",std::move(tris_vis));

        auto points_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& points_vis_verts = points_vis->verts;
        points_vis_verts.resize(nm_ints);
        ompPol(zs::range(nm_ints),[
            points_buffer = proxy<omp_space>({},points_buffer),
            &points_vis_verts] (int ci) mutable {
                auto x0 = points_buffer.pack(dim_c<3>,"x0",ci);
                points_vis_verts[ci] = x0.to_array();
        });
        set_output("points_vis",std::move(points_vis));

        auto lines_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& lines_vis_verts = lines_vis->verts;
        auto& lines_vis_lines = lines_vis->lines;
        lines_vis_verts.resize(nm_ints * 2);
        lines_vis_lines.resize(nm_ints);
        ompPol(zs::range(nm_ints),[
            lines_buffer = proxy<omp_space>({},lines_buffer),
            &lines_vis_verts,&lines_vis_lines] (int ci) mutable {
                auto x0 = lines_buffer.pack(dim_c<3>,"x0",ci);
                auto x1 = lines_buffer.pack(dim_c<3>,"x1",ci);

                lines_vis_verts[ci * 2 + 0] = x0.to_array();
                lines_vis_verts[ci * 2 + 1] = x1.to_array();
                lines_vis_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });
        set_output("lines_vis",std::move(lines_vis));
    }
};

ZENDEFNODE(VisualizeCollision2, {{
                                    "zsparticles",
                                    {"float","out_collisionEps","0.1"},
                                    {"float","in_collisionEps","0.1"},    
                                },
                                  {
                                        "tris_vis",
                                        "points_vis",
                                        "lines_vis",
                                        "flood_vis"
                                    },
                                  {
                                    
                                  },
                                  {"ZSGeometry"}});


struct VisualizeIntersections3 : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;
    using vec3 = zs::vec<T, 3>;
    using table_vec2i_type = zs::bht<int,2,int>;
    using table_int_type = zs::bht<int,1,int>;

    virtual void apply() override { 
        using namespace zs;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();  
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();  

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        const auto &tris = (*zsparticles)[ZenoParticles::s_surfTriTag]; 
        const auto& tets = zsparticles->getQuadraturePoints();
        const auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
        auto& verts = zsparticles->getParticles();
        auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        // const auto& points = 

        auto in_collisionEps = get_input2<T>("in_collisionEps");
        auto out_collisionEps = get_input2<T>("out_collisionEps");

        // auto kinematics = get_input<ListObject>("kinematics")->get2<ZenoParticles*>();
        // auto kinematics = RETRIEVE_OBJECT_PTRS(ZenoParticles,"kinematics");
        auto kinematic = get_input<ZenoParticles>("kinematic");
    
        zs::bht<int,2,int> csPT{verts.get_allocator(),100000};
        csPT.reset(cudaPol,true);
        // zs::Vector<int> csPTOffsets{verts.get_allocator(),kinematics.size()};
        // std::cout << "number of kinematics : " << kinematics.size() << std::endl;
        auto collide_from_exterior = get_input2<bool>("collide_from_exterior");

        auto nm_csPT = COLLISION_UTILS::do_tetrahedra_surface_points_and_kinematic_boundary_collision_detection(cudaPol,
            kinematic,
            verts,"x",
            tets,
            points,tris,
            halfedges,
            out_collisionEps,
            in_collisionEps,
            csPT,
            collide_from_exterior,
            true);

        std::cout << "do_tetrahedra_surface_points_and_kinematic_boundary_collision_detection with csPT : " << nm_csPT << std::endl;

        int nm_kverts = kinematic->getParticles().size();
        int nm_ktris = kinematic->getQuadraturePoints().size();

        // for(auto kinematic : kinematics) {
        //     nm_kverts += kinematic->getParticles().size();
        //     nm_ktris += kinematic->getQuadraturePoints().size();
        // }

        dtiles_t flood_dyn{verts.get_allocator(),{
            {"x",3},
            {"flood",1}
        },verts.size()};
        dtiles_t all_kverts_buffer{verts.get_allocator(),{
            {"x",3},
            {"flood",1}
        },(size_t)nm_kverts};
        dtiles_t all_ktri_verts_buffer{verts.get_allocator(),{
            {"x0",3},
            {"x1",3},
            {"x2",3}
        },(size_t)nm_ktris};

        TILEVEC_OPS::copy(cudaPol,verts,"x",flood_dyn,"x");
        TILEVEC_OPS::copy(cudaPol,verts,"flood",flood_dyn,"flood");

        int toffset = 0;
        int voffset = 0;
        // for(auto kinematic : kinematics) {
        const auto& kverts = kinematic->getParticles();
        const auto& ktris = kinematic->getQuadraturePoints();
        TILEVEC_OPS::copy(cudaPol,kverts,"x",all_kverts_buffer,"x",voffset);
        TILEVEC_OPS::copy(cudaPol,kverts,"flood",all_kverts_buffer,"flood",voffset);
        cudaPol(zs::range(ktris.size()),[
            ktris = proxy<cuda_space>({},ktris),
            kverts = proxy<cuda_space>({},kverts),
            all_ktri_verts_buffer = proxy<cuda_space>({},all_ktri_verts_buffer),
            toffset = toffset] ZS_LAMBDA(int kti) mutable {
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                for(int i = 0;i != 3;++i) {
                    all_ktri_verts_buffer.tuple(dim_c<3>,"x0",toffset + kti) = kverts.pack(dim_c<3>,"x",ktri[0]);
                    all_ktri_verts_buffer.tuple(dim_c<3>,"x1",toffset + kti) = kverts.pack(dim_c<3>,"x",ktri[1]);
                    all_ktri_verts_buffer.tuple(dim_c<3>,"x2",toffset + kti) = kverts.pack(dim_c<3>,"x",ktri[2]);
                }
        });
        //     voffset += kverts.size();
        //     toffset += ktris.size();
        // }

        dtiles_t csPTBuffer{verts.get_allocator(),{
            {"x0",3},
            {"x1",3},
            {"t0",3},
            {"t1",3},
            {"t2",3}
        },(size_t)csPT.size()};
        cudaPol(zip(zs::range(csPT.size()),zs::range(csPT._activeKeys)),[
            tris = proxy<cuda_space>({},tris),
            verts = proxy<cuda_space>({},verts),
            points = proxy<cuda_space>({},points),
            all_ktri_verts_buffer = proxy<cuda_space>({},all_ktri_verts_buffer),
            csPTBuffer = proxy<cuda_space>({},csPTBuffer),
            all_kverts_buffer = proxy<cuda_space>({},all_kverts_buffer)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto pi = pair[0];
                auto kti = pair[1];
                zs::vec<T,3> ktV[3] = {};
                // for(int i = 0;i != 3;++i)
                ktV[0] = all_ktri_verts_buffer.pack(dim_c<3>,"x0",kti);
                ktV[1] = all_ktri_verts_buffer.pack(dim_c<3>,"x1",kti);
                ktV[2] = all_ktri_verts_buffer.pack(dim_c<3>,"x2",kti);
                
                auto ktC = zs::vec<T,3>::zeros();
                for(int i = 0;i != 3;++i)   
                    ktC += ktV[i] / (T)3.0;
                
                auto vi = zs::reinterpret_bits<int>(points("inds",pi));
                auto pv = verts.pack(dim_c<3>,"x",vi);

                csPTBuffer.tuple(dim_c<3>,"x0",ci) = pv;
                csPTBuffer.tuple(dim_c<3>,"x1",ci) = ktC;
                csPTBuffer.tuple(dim_c<3>,"t0",ci) = ktV[0];
                csPTBuffer.tuple(dim_c<3>,"t1",ci) = ktV[1];
                csPTBuffer.tuple(dim_c<3>,"t2",ci) = ktV[2];
        });
        csPTBuffer = csPTBuffer.clone({memsrc_e::host});

        dtiles_t intersectHalfEdges{halfedges.get_allocator(),{
            {"x0",3},
            {"x1",3}
        },(size_t)halfedges.size()};
        cudaPol(zs::range(halfedges.size()),[
            halfedges = proxy<cuda_space>({},halfedges),
            verts = proxy<cuda_space>({},verts),
            tris = proxy<cuda_space>({},tris),
            intersectHalfEdges = proxy<cuda_space>({},intersectHalfEdges)] ZS_LAMBDA(int hi) mutable {
                auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                auto local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));

                auto intersect = halfedges("intersect",hi);
                intersectHalfEdges.tuple(dim_c<3>,"x0",hi) = verts.pack(dim_c<3>,"x",tri[local_vertex_id]);
                intersectHalfEdges.tuple(dim_c<3>,"x1",hi) = verts.pack(dim_c<3>,"x",tri[local_vertex_id]);

                if(intersect > (T)0.5) {
                    intersectHalfEdges.tuple(dim_c<3>,"x1",hi) = verts.pack(dim_c<3>,"x",tri[(local_vertex_id + 1) % 3]);
                }
        });
        intersectHalfEdges = intersectHalfEdges.clone({memsrc_e::host});

        flood_dyn = flood_dyn.clone({memsrc_e::host});
        all_kverts_buffer = all_kverts_buffer.clone({memsrc_e::host});


        auto flood_dyn_vis = std::make_shared<zeno::PrimitiveObject>();
        flood_dyn_vis->resize(flood_dyn.size());
        auto& flood_dyn_verts = flood_dyn_vis->verts;
        auto& flood_dyn_tags = flood_dyn_verts.add_attr<T>("flood");
        ompPol(zs::range(flood_dyn.size()),[
            flood_dyn = proxy<omp_space>({},flood_dyn),
            &flood_dyn_verts,&flood_dyn_tags] (int vi) mutable {
                auto pv = flood_dyn.pack(dim_c<3>,"x",vi);
                auto flood = flood_dyn("flood",vi);
                flood_dyn_verts[vi] = pv.to_array();
                flood_dyn_tags[vi] = flood;
        });
        set_output("flood_dynamic",std::move(flood_dyn_vis));

        auto flood_kin_vis = std::make_shared<zeno::PrimitiveObject>();
        flood_kin_vis->resize(all_kverts_buffer.size());
        auto& flood_kin_verts = flood_kin_vis->verts;
        auto& flood_kin_tags = flood_kin_verts.add_attr<T>("flood");
        ompPol(zs::range(all_kverts_buffer.size()),[
            all_kverts_buffer = proxy<omp_space>({},all_kverts_buffer),
            &flood_kin_verts,&flood_kin_tags] (int kvi) mutable {
                auto kv = all_kverts_buffer.pack(dim_c<3>,"x",kvi);
                auto flood =all_kverts_buffer("flood",kvi);
                flood_kin_verts[kvi] = kv.to_array();
                flood_kin_tags[kvi] = flood;
        });
        set_output("flood_kinematic",std::move(flood_kin_vis));

        auto csPTVis = std::make_shared<zeno::PrimitiveObject>();
        auto& csPT_verts = csPTVis->verts;
        csPT_verts.resize(csPTBuffer.size() * 2);
        auto& csPT_lines = csPTVis->lines;
        csPT_lines.resize(csPTBuffer.size());

        ompPol(zs::range(csPTBuffer.size()),[
            csPTBuffer = proxy<omp_space>({},csPTBuffer),
            &csPT_verts,&csPT_lines] (int ci) mutable {
                auto x0 = csPTBuffer.pack(dim_c<3>,"x0",ci);
                auto x1 = csPTBuffer.pack(dim_c<3>,"x1",ci);
                csPT_verts[ci * 2 + 0] = x0.to_array();
                csPT_verts[ci * 2 + 1] = x1.to_array();
                csPT_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });
        set_output("cspt_vis",std::move(csPTVis));

        auto csPTTri = std::make_shared<zeno::PrimitiveObject>();
        auto& csPT_tri_verts = csPTTri->verts;
        csPT_tri_verts.resize(csPTBuffer.size() * 3);
        auto& csPT_tri_tris = csPTTri->tris;
        csPT_tri_tris.resize(csPTBuffer.size());

        ompPol(zs::range(csPTBuffer.size()),[
            csPTBuffer = proxy<omp_space>({},csPTBuffer),
            &csPT_tri_verts,&csPT_tri_tris] (int ci) mutable {
                auto t0 = csPTBuffer.pack(dim_c<3>,"t0",ci);
                auto t1 = csPTBuffer.pack(dim_c<3>,"t1",ci);
                auto t2 = csPTBuffer.pack(dim_c<3>,"t2",ci);
                csPT_tri_verts[ci * 3 + 0] = t0.to_array();
                csPT_tri_verts[ci * 3 + 1] = t1.to_array();
                csPT_tri_verts[ci * 3 + 2] = t2.to_array();
                csPT_tri_tris[ci] = zeno::vec3i{ci * 3 + 0,ci * 3 + 1,ci * 3 + 2};
        });
        set_output("cspt_tri_vis",std::move(csPTTri));

        // intersectHalfEdges
        auto intersect_edges = std::make_shared<zeno::PrimitiveObject>();
        auto& ih_verts = intersect_edges->verts;
        auto& ih_lines = intersect_edges->lines;
        ih_verts.resize(intersectHalfEdges.size() * 2);
        ih_lines.resize(intersectHalfEdges.size());

        ompPol(zs::range(intersectHalfEdges.size()),[
            intersectHalfEdges = proxy<omp_space>({},intersectHalfEdges),
            &ih_verts,&ih_lines] (int hi) mutable {
                auto x0 = intersectHalfEdges.pack(dim_c<3>,"x0",hi);
                auto x1 = intersectHalfEdges.pack(dim_c<3>,"x1",hi);

                ih_verts[hi * 2 + 0] = x0.to_array();
                ih_verts[hi * 2 + 1] = x1.to_array();
                ih_lines[hi] = zeno::vec2i{hi * 2 + 0,hi * 2 + 1};
        });

        set_output("intersect_edges",std::move(intersect_edges));
    }
};

ZENDEFNODE(VisualizeIntersections3, {{
                                        "zsparticles",
                                        "kinematic",
                                        {"float","out_collisionEps","0.1"},
                                        {"float","in_collisionEps","0.1"},    
                                        {"bool","collide_from_exterior","1"}
                                    },
                                  {
                                        "flood_dynamic",
                                        "flood_kinematic",
                                        "cspt_vis",
                                        "cspt_tri_vis",
                                        "intersect_edges"
                                    },
                                  {
                                    
                                  },
                                  {"ZSGeometry"}});

};