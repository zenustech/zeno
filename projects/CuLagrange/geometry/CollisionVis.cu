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

#include <iostream>

namespace zeno {

    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using mat3 = zs::vec<T,3,3>;
    using mat4 = zs::vec<T,4,4>;


    // for each triangle, find the three incident triangles
    struct ZSInitTopoConnect : INode {

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
            const auto& verts = surf->getParticles();

            auto cudaExec = cuda_exec();
            auto bvh_thickness = (T)5 * compute_average_edge_length(cudaExec,verts,tris);

            tris.append_channels(cudaExec,{{"ff_inds",3},{"fe_inds",3}});
            lines.append_channels(cudaExec,{{"fe_inds",2}});
            if(!compute_ff_neigh_topo(cudaExec,verts,tris,"ff_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_neight_topo fail");
            if(!compute_fe_neigh_topo(cudaExec,verts,lines,tris,"fe_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_neight_topo fail");

            set_output("zssurf",surf);
        }
    };

    ZENDEFNODE(ZSInitTopoConnect, {{{"zssurf"}},
                                {{"zssurf"}},
                                {},
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

            std::cout << "CALCULATE SURFACE NORMAL" << std::endl;

            if(!calculate_facet_normal(cudaExec,verts,tris,"nrm"))
                throw std::runtime_error("ZSCalNormal::calculate_facet_normal fail"); 
            std::cout << "FINISH CALCULATE SURFACE NORMAL" << std::endl;

            auto ceNrmTag = get_param<std::string>("ceNrmTag");
            if(!lines.hasProperty(ceNrmTag))
                lines.append_channels(cudaExec,{{ceNrmTag,9}});
            
            // evalute the normal of edge plane
            cudaExec(range(lines.size()),
                [verts = proxy<space>({},verts),
                    tris = proxy<space>({},tris),
                    lines = proxy<space>({},lines),
                    ceNrmTag = zs::SmallString(ceNrmTag)] ZS_LAMBDA(int ei) mutable {
                        auto e_inds = lines.template pack<2>("inds",ei).template reinterpret_bits<int>();
                        auto fe_inds = lines.template pack<2>("fe_inds",ei).template reinterpret_bits<int>();
                        auto n0 = tris.template pack<3>("nrm",fe_inds[0]);
                        auto n1 = tris.template pack<3>("nrm",fe_inds[1]);

                        auto ne = (n0 + n1).normalized();
                        auto e0 = verts.template pack<3>("x",e_inds[0]);
                        auto e1 = verts.template pack<3>("x",e_inds[1]);
                        auto e10 = e1 - e0;

                        lines.template tuple<3>(ceNrmTag,ei) = e10.cross(ne).normalized();
            });

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
            std::vector<zs::PropertyTag> tags{{"x",9},{"dir",9}};
            auto cell_buffer = typename ZenoParticles::particles_t(tags,tris.size(),zs::memsrc_e::device,0);
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
                for(int i = 0;i < 3;++i){
                    auto vert = verts.template pack<3>("x",inds[i]);
                    for(int j = 0;j < 3;++j)
                        cell_buffer("x",j,i) = vert[j];
                    
                    auto e0 = fe_inds[(i-1) % 3];
                    auto e1 = fe_inds[i];

                    auto n0 = lines.template pack<3>(ceNrmTag,e0);
                    auto n1 = lines.template pack<3>(ceNrmTag,e1);

                    if(is_edge_match(lines.template pack<2>("inds",e0).template reinterpret_bits<int>(),zs::vec<int,2>{inds[((i - 1) % 3)],inds[i]}) == 1)
                        n0 =  (T)-1 * n0;
                    if(is_edge_match(lines.template pack<2>("inds",e1).template reinterpret_bits<int>(),zs::vec<int,2>{inds[i],inds[((i + 1) % 3)]}) == 1)
                        n1 = (T)-1 * n1;

                    auto dir = n1.cross(n0).normalized();
                    for(int j = 0;j < 3;++j)
                        cell_buffer("dir",j,i) = dir[j];
                    
                }
            });  

            cell_buffer = cell_buffer.clone({zs::memsrc_e::host});   
            constexpr auto omp_space = execspace_e::openmp;
            auto ompPol = omp_exec();            

            auto cell = std::make_shared<zeno::PrimitiveObject>();
            cell->resize(cell_buffer.size() * 6);

            auto out_offset = get_param<float>("out_offset");
            auto in_offset = get_param<float>("in_offset");

            auto& cell_verts = cell->attr<zeno::vec3f>("pos");
            auto& cell_lines = cell->lines;
            cell_lines.resize(cell_buffer.size() * 9);

            ompPol(zs::range(cell_buffer.size()),
                [cell_buffer = proxy<omp_space>({},cell_buffer),
                    &cell_verts,&cell_lines,&out_offset,&in_offset] (int ci) mutable {
                auto vs_ = cell_buffer.template pack<9>("x",ci);
                auto ds_ = cell_buffer.template pack<9>("dir",ci);

                for(int i = 0;i < 3;++i) {
                    auto p = vec3{vs_[i*3 + 0],vs_[i*3 + 1],vs_[i*3 + 2]};
                    auto dp = vec3{ds_[i*3 + 0],ds_[i*3 + 1],ds_[i*3 + 2]};

                    auto p0 = p + dp * in_offset;
                    auto p1 = p + dp * out_offset;

                    cell_verts[ci * 6 + i * 2 + 0] = zeno::vec3f{p0[0],p0[1],p0[2]};
                    cell_verts[ci * 6 + i * 2 + 1] = zeno::vec3f{p1[0],p1[1],p1[2]};

                    cell_lines[ci * 9 + 0 + i] = zeno::vec2i{ci * 6 + i * 2 + 0,ci * 6 + i * 2 + 1};
                }

                for(int i = 0;i < 3;++i) {
                    cell_lines[ci * 9 + 3 + i] = zeno::vec2i{ci * 6 + i * 2 + 0,ci * 6 + ((i+1)%3) * 2 + 0};
                    cell_lines[ci * 9 + 6 + i] = zeno::vec2i{ci * 6 + i * 2 + 1,ci * 6 + ((i+1)%3) * 2 + 1}; 
                }
            });

            set_output("collision_cell",std::move(cell));
        }
    };

    ZENDEFNODE(VisualizeCollisionCell, {{{"ZSParticles"}},
                                {{"zssurf"}},
                                {{"float","out_offset","0.1"},{"float","in_offset","0.1"},{"string","ceNrmTag","nrm"}},
                                {"ZSGeometry"}});

}