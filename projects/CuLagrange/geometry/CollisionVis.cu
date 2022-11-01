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

#include <iostream>


// #define COLLISION_VIS_DEBUG

namespace zeno {

    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using mat3 = zs::vec<T,3,3>;
    using mat4 = zs::vec<T,4,4>;
    // using vec2i = zs::vec<int,2>;
    // using vec3i = zs::vec<int,3>;
    // using vec4i = zs::vec<int,4>;


    // for each triangle, find the three incident triangles
    // TODO: build a half edge structure
    struct ZSInitSurfaceTopoConnect : INode {

        void apply() override {
            using namespace zs;

            auto surf = get_input<ZenoParticles>("zssurf");
            // auto bvh_thickness = get_input2<float>("bvh_thickness");
            if(!surf->hasAuxData(ZenoParticles::s_surfTriTag))
                throw std::runtime_error("the input zsparticles has no surface tris");
            if(!surf->hasAuxData(ZenoParticles::s_surfEdgeTag))
                throw std::runtime_error("the input zsparticles has no surface lines");
            auto& tris  = (*surf)[ZenoParticles::s_surfTriTag];
            auto& lines = (*surf)[ZenoParticles::s_surfEdgeTag];
            auto& points = (*surf)[ZenoParticles::s_surfVertTag];

            
            if(!lines.hasProperty("inds") || lines.getChannelSize("inds") != 2) {
                throw std::runtime_error("the line has no inds channel");
            }
            if(!points.hasProperty("inds") || points.getChannelSize("inds") != 1) {
                throw std::runtime_error("the point has no inds channel");
            }

            const auto& verts = surf->getParticles();

            auto cudaExec = cuda_exec();
            // constexpr auto space = zs::execspace_e::cuda;

            // cudaExec(range(lines.size()),
            //     [lines = proxy<space>({},lines)] ZS_LAMBDA(int li) mutable {
            //         auto inds = lines.template pack<2>("inds",li).template reinterpret_bits<int>();
            //             printf("line[%d] : %d %d\n",(int)li,(int)inds[0],(int)inds[1]);
            // });

            auto bvh_thickness = (T)2 * compute_average_edge_length(cudaExec,verts,"x",tris);

            // std::cout << "bvh_thickness : " << bvh_thickness << std::endl;

            tris.append_channels(cudaExec,{{"ff_inds",3},{"fe_inds",3},{"fp_inds",3}});
            lines.append_channels(cudaExec,{{"fe_inds",2}});
            if(!compute_ff_neigh_topo(cudaExec,verts,tris,"ff_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_neigh_topo fail");
            if(!compute_fe_neigh_topo(cudaExec,verts,lines,tris,"fe_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_neigh_topo fail");
            if(!compute_fp_neigh_topo(cudaExec,verts,points,tris,"fp_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_point_neigh_topo fail");

            set_output("zssurf",surf);
        }
    };

    ZENDEFNODE(ZSInitSurfaceTopoConnect, {{{"zssurf"}},
                                {{"zssurf"}},
                                {},
                                {"ZSGeometry"}});


    template<typename VTILEVEC> 
    constexpr vec3 eval_center(const VTILEVEC& verts,const zs::vec<int,4>& tet) {
        auto res = vec3::zeros();
        for(int i = 0;i < 4;++i)
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

    struct VisualizeTopology : INode {

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

            auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
            auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

            if(!tris.hasProperty("ff_inds") || tris.getPropertySize("ff_inds") != 3){
                throw std::runtime_error("no valid ff_inds detected in tris");
            }            

            if(!tris.hasProperty("fe_inds") || tris.getPropertySize("fe_inds") != 3) {
                throw std::runtime_error("no valid fe_inds detected in tris");
            }

            if(!lines.hasProperty("fe_inds") || lines.getPropertySize("fe_inds") != 2) {
                throw std::runtime_error("no valid fe_inds detected in lines");
            }

            const auto& verts = zsparticles->getParticles();
            std::vector<zs::PropertyTag> tags{{"x",3}};

            int nm_tris = tris.size();

            // output ff topo first
            auto ff_topo = typename ZenoParticles::particles_t(tags,nm_tris * 4,zs::memsrc_e::device,0);
            auto fe_topo = typename ZenoParticles::particles_t(tags,nm_tris * 4,zs::memsrc_e::device,0);
            auto fp_topo = typename ZenoParticles::particles_t(tags,nm_tris * 4,zs::memsrc_e::device,0);

            // transfer the data from gpu to cpu
            constexpr auto cuda_space = execspace_e::cuda;
            auto cudaPol = cuda_exec();  
            cudaPol(zs::range(nm_tris),
                [ff_topo = proxy<cuda_space>({},ff_topo),
                    fe_topo = proxy<cuda_space>({},fe_topo),
                    fp_topo = proxy<cuda_space>({},fp_topo),
                    tris = proxy<cuda_space>({},tris),
                    lines = proxy<cuda_space>({},lines),
                    points = proxy<cuda_space>({},points),
                    verts = proxy<cuda_space>({},verts)] ZS_LAMBDA(int ti) mutable {
                        auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                        auto ff_inds = tris.template pack<3>("ff_inds",ti).reinterpret_bits(int_c);
                        auto fe_inds = tris.template pack<3>("fe_inds",ti).reinterpret_bits(int_c);
                        auto fp_inds = tris.template pack<3>("fp_inds",ti).reinterpret_bits(int_c);
                        
                        auto center = eval_center(verts,tri);
                        ff_topo.template tuple<3>("x",ti * 4 + 0) = center;
                        fe_topo.template tuple<3>("x",ti * 4 + 0) = center;
                        fp_topo.template tuple<3>("x",ti * 4 + 0) = center;
                        for(int i = 0;i != 3;++i) {
                            auto nti = ff_inds[i];
                            auto ntri = tris.template pack<3>("inds",nti).reinterpret_bits(int_c);
                            auto ncenter = eval_center(verts,ntri);
                            ff_topo.template tuple<3>("x",ti * 4 + i + 1) = ncenter;

                            auto nei = fe_inds[i];
                            auto nedge = lines.template pack<2>("inds",nei).reinterpret_bits(int_c);
                            ncenter = eval_center(verts,nedge);
                            // printf("edge[%d] : %d %d\n",nei,nedge[0],nedge[1]);
                            fe_topo.template tuple<3>("x",ti * 4 + i + 1) = ncenter;

                            auto pidx = reinterpret_bits<int>(points("inds",fp_inds[i]));
                            fp_topo.template tuple<3>("x",ti * 4 + i + 1) = verts.template pack<3>("x",pidx);
                        }

            });   

            ff_topo = ff_topo.clone({zs::memsrc_e::host});
            fe_topo = fe_topo.clone({zs::memsrc_e::host});
            fp_topo = fp_topo.clone({zs::memsrc_e::host});

            int ff_size = ff_topo.size();
            int fe_size = fe_topo.size();
            int fp_size = fp_topo.size();

            constexpr auto omp_space = execspace_e::openmp;
            auto ompPol = omp_exec();

            auto ff_prim = std::make_shared<zeno::PrimitiveObject>();
            auto fe_prim = std::make_shared<zeno::PrimitiveObject>();
            auto fp_prim = std::make_shared<zeno::PrimitiveObject>();

            auto& ff_verts = ff_prim->verts;
            auto& ff_lines = ff_prim->lines;

            auto& fe_verts = fe_prim->verts;
            auto& fe_lines = fe_prim->lines;

            auto& fp_verts = fp_prim->verts;
            auto& fp_lines = fp_prim->lines;

            int ff_pair_count = nm_tris * 3;
            int fe_pair_count = nm_tris * 3;
            int fp_pair_count = nm_tris * 3;

            ff_verts.resize(ff_size);
            ff_lines.resize(ff_pair_count);
            fe_verts.resize(fe_size);
            fe_lines.resize(fe_pair_count);
            fp_verts.resize(fp_size);
            fp_lines.resize(fp_pair_count);

            ompPol(zs::range(nm_tris),
                [&ff_verts,&ff_lines,ff_topo = proxy<omp_space>({},ff_topo)] (int fi) mutable {
                    auto v = ff_topo.template pack<3>("x",fi * 4 + 0);
                    ff_verts[fi * 4 + 0] = zeno::vec3f(v[0],v[1],v[2]);
                    for(int i = 0;i != 3;++i){
                        auto v = ff_topo.template pack<3>("x",fi * 4 + i + 1);
                        ff_verts[fi * 4 + i + 1] = zeno::vec3f(v[0],v[1],v[2]);
                        ff_lines[fi * 3 + i] = zeno::vec2i(fi * 4 + 0,fi * 4 + i + 1);
                    }
            });

            ompPol(zs::range(nm_tris),
                [&fe_verts,&fe_lines,fe_topo = proxy<omp_space>({},fe_topo)] (int fi) mutable {
                    auto v = fe_topo.template pack<3>("x",fi * 4 + 0);
                    fe_verts[fi * 4 + 0] = zeno::vec3f(v[0],v[1],v[2]);
                    for(int i = 0;i != 3;++i){
                        auto v = fe_topo.template pack<3>("x",fi * 4 + i + 1);
                        fe_verts[fi * 4 + i + 1] = zeno::vec3f(v[0],v[1],v[2]);
                        fe_lines[fi * 3 + i] = zeno::vec2i(fi * 4 + 0,fi * 4 + i + 1);
                    }
            });

            ompPol(zs::range(nm_tris),
                [&fp_verts,&fp_lines,fp_topo = proxy<omp_space>({},fp_topo)] (int fi) mutable {
                    auto v = fp_topo.template pack<3>("x",fi * 4 + 0);
                    fp_verts[fi * 4 + 0] = zeno::vec3f(v[0],v[1],v[2]);
                    for(int i = 0;i != 3;++i){
                        auto v = fp_topo.template pack<3>("x",fi * 4 + i + 1);
                        fp_verts[fi * 4 + i + 1] = zeno::vec3f(v[0],v[1],v[2]);
                        fp_lines[fi * 3 + i] = zeno::vec2i(fi * 4 + 0,fi * 4 + i + 1);
                    }
            });

            // for(int i = 0;i < fe_lines.size();++i)
            //     std::cout << "fe_line<" << i << "> : \t" << fe_lines[i][0] << "\t" << fe_lines[i][1] << std::endl;
            set_output("fp_topo",std::move(fp_prim));
            set_output("ff_topo",std::move(ff_prim));
            set_output("fe_topo",std::move(fe_prim));
        }
    };


    ZENDEFNODE(VisualizeTopology, {{{"ZSParticles"}},
                                {{"ff_topo"},{"fe_topo"},{"fp_topo"}},
                                {},
                                {"ZSGeometry"}});


    // struct VisualizeSurfaceMesh : INode {
    //     virtual void apply() override {
    //         using namespace zs;
    //         auto zsparticles = get_input<ZenoParticles>("ZSParticles");

    //         if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag)){
    //             throw std::runtime_error("the input zsparticles has no surface tris");
    //             // auto& tris = (*particles)[ZenoParticles::s_surfTriTag];
    //             // tris = typename ZenoParticles::particles_t({{"inds",3}});
    //         }
    //         if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag)) {
    //             throw std::runtime_error("the input zsparticles has no surface lines");
    //         }
    //         if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) {
    //             throw std::runtime_error("the input zsparticles has no surface points");
    //         }
    //         const auto& tris    = (*zsparticles)[ZenoParticles::s_surfTriTag];
    //         const auto& points  = (*zsparticles)[ZenoParticles::s_surfVertTag];
    //         const auto& verts = zsparticles->getParticles();

    //         if(!tris.hasProperty("fp_inds") || tris.getChannelSize("fp_inds") != 3) {
    //             throw std::runtime_error("call ZSInitSurfaceTopology first before VisualizeSurfaceMesh");
    //         }

    //         auto nm_points = points.size();
    //         auto nm_tris = tris.size();
    //         // output ff topo first
    //         auto surf_verts = typename ZenoParticles::particles_t({{"x",3}},nm_tris * 4,zs::memsrc_e::device,0);
    //         auto surf_tris = typename ZenoParticles::particles_t({{"inds",3}},nm_tris * 4,zs::memsrc_e::device,0);            

    //         // transfer the data from gpu to cpu
    //         constexpr auto cuda_space = execspace_e::cuda;
    //         auto cudaPol = cuda_exec(); 

    //         // cudaPol(zs::range())     
    //     }
    // };


    struct VisualizeSurfaceNormal : INode {
        virtual void apply() override {
            using namespace zs;
            auto zsparticles = get_input<ZenoParticles>("ZSParticles");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag)){
                throw std::runtime_error("the input zsparticles has no surface tris");
                // auto& tris = (*particles)[ZenoParticles::s_surfTriTag];
                // tris = typename ZenoParticles::particles_t({{"inds",3}});
            }

            auto cudaExec = cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;

            const auto& verts = zsparticles->getParticles();
            auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
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
                throw std::runtime_error("ZSCalNormal::calculate_facet_normal fail"); 


            auto buffer = typename ZenoParticles::particles_t({{"nrm",3},{"x",3}},lines.size(),zs::memsrc_e::device,0);  

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

                        buffer.template tuple<3>("nrm",ei) = (n0 + n1).normalized();
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
            constexpr auto space = zs::execspace_e::cuda;

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
            auto out_offset = get_input2<float>("out_offset");
            auto in_offset = get_input2<float>("in_offset");
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
                    ceNrmTag = zs::SmallString(ceNrmTag),
                    out_offset,in_offset] ZS_LAMBDA(int ci) mutable {
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

                    auto cell_center = vec3::zeros();
                    cell_center = (vs[0] + vs[1] + vs[2])/(T)3.0;
                    T check_dist{};
                    auto check_intersect = is_inside_the_cell(verts,"x",
                        lines,tris,
                        tris,"nrm",
                        lines,ceNrmTag,
                        ci,cell_center,in_offset,out_offset);
                    if(check_intersect == 1)
                        printf("invalid cell intersection check offset and inset : %d %f %f %f\n",ci,(float)check_dist,(float)out_offset,(float)in_offset);
                    if(check_intersect == 2)
                        printf("invalid cell intersection check bisector : %d\n",ci);


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
                    &cell_verts,&cell_lines,&cell_tris,&out_offset,&in_offset,&offset_ratio] (int ci) mutable {

                auto vs_ = cell_buffer.template pack<9>("x",ci);
                auto ds_ = cell_buffer.template pack<9>("dir",ci);

                auto center = cell_buffer.template pack<3>("center",ci);

                for(int i = 0;i < 3;++i) {
                    auto p = vec3{vs_[i*3 + 0],vs_[i*3 + 1],vs_[i*3 + 2]};
                    auto dp = vec3{ds_[i*3 + 0],ds_[i*3 + 1],ds_[i*3 + 2]};

                    auto p0 = p - dp * in_offset;
                    auto p1 = p + dp * out_offset;

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
                    &tcell_verts,&tcell_lines,&out_offset,&in_offset,&offset_ratio] (int ci) mutable {

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

                    auto p0 = p - dp * in_offset;
                    auto p1 = p + dp * out_offset;

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
                    &ncell_verts,&ncell_lines,&nrm_offset,&offset_ratio] (int ci) mutable {    
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

    ZENDEFNODE(VisualizeCollisionCell, {{{"ZSParticles"},{"float","out_offset","0.1"},{"float","in_offset","0.1"},{"float","nrm_offset","0.1"},{"float","offset_ratio","0.8"}},
                                {{"collision_cell"},{"ccell_tangent"},{"ccell_normal"}},
                                {{"string","ceNrmTag","nrm"}},
                                {"ZSGeometry"}});

}