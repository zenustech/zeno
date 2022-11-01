#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>


#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "kernel/calculate_area.hpp"

#include <iostream>


namespace zeno {
    using T = float;

    struct ZSCalcSurfaceArea : INode {
        void apply() override {
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

            auto& verts = zsparticles->getParticles();

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

            auto type = get_param<std::string>("type");

            // transfer the data from gpu to cpu
            // constexpr auto cuda_space = execspace_e::cuda;
            auto cudaPol = cuda_exec();  

            bool cal_points = false;
            bool cal_lines = false;
            // bool cal_tris = false;

            if(type == "point")
                cal_points = true;
            if(type == "line")
                cal_lines = true;
            if(type == "all") {
                cal_points = true;
                cal_lines = true;
            }

            calculate_tris_area(cudaPol,verts,tris,"x","area");
            if(cal_points)
                calculate_points_one_ring_area(cudaPol,tris,points,"area","area");
            if(cal_lines)
                calculate_lines_area(cudaPol,verts,lines,tris,"area","area");
            

            set_output("ZSParticles",zsparticles);
        }
    };


    ZENDEFNODE(ZSCalcSurfaceArea, {{{"ZSParticles"}},
                                {{"ZSParticles"}},
                                {},
                                {"ZSGeometry"}});

};