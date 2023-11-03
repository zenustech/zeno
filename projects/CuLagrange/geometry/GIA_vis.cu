#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>

#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/global_intersection_analysis.hpp"

namespace zeno {

struct VisualizeIntersectionLoops : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        auto cudaExec = cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        const auto& verts = zsparticles->getParticles();
        const auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        const auto& tris = zsparticles->getQuadraturePoints();

        auto source_tag = get_input2<std::string>("source_tag");
        auto dest_tag = get_input2<std::string>("dest_tag");

        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();
        
        zs::bht<int,2,int> csHT{verts.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};
        csHT.reset(cudaExec,true);

        std::cout << "retrieve_self_intersection_tri_halfedges_pairs" << std::endl;

        GIA::retrieve_self_intersection_tri_halfedge_pairs(cudaExec,verts,source_tag,tris,halfedges,csHT);

        std::cout << "nm_csHT : " << csHT.size() << std::endl;

        bht<int,2,int> boundary_pairs_set{csHT.get_allocator(),GIA::DEFAULT_MAX_GIA_INTERSECTION_PAIR};
        boundary_pairs_set.reset(cudaExec,true);
        bht<int,2,int> turning_pairs_set{csHT.get_allocator(),GIA::DEFAULT_MAX_NM_TURNING_POINTS};
        turning_pairs_set.reset(cudaExec,true);
        
        std::cout << "finding the turning pairs " << std::endl;

        zs:bht<int,2,int> csHT_host{csHT.get_allocator(),csHT.size()};csHT_host.reset(cudaExec,true);
        cudaExec(zip(zs::range(csHT.size()),csHT._activeKeys),[
            csHT_host = proxy<space>(csHT_host),
            halfedges = proxy<space>({},halfedges),
            tris = proxy<space>({},tris),
            boundary_pairs_set = proxy<space>(boundary_pairs_set),
            turning_pairs_set = proxy<space>(turning_pairs_set)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                csHT_host.insert(pair);
                // printf("try_testing pair[%d %d] for turning point\n",pair[0],pair[1]);
                if(GIA::is_boundary_halfedge(halfedges,pair[0]))
                    boundary_pairs_set.insert(pair);
                if(auto tpi = GIA::find_intersection_turning_point(halfedges,tris,pair);tpi >= 0) {
                    // printf("finding turning pairs : %d %d %d\n",pair[0],pair[1],tpi);
                    turning_pairs_set.insert(pair);
                }
        });

        std::cout << "nm_csHT_host : " << csHT_host.size() << std::endl;

        csHT_host = csHT_host.clone({memsrc_e::host});
        auto csHT_host_proxy = proxy<omp_space>(csHT_host);
        auto verts_host = verts.clone({memsrc_e::host});
        auto tris_host = tris.clone({memsrc_e::host});
        auto halfedges_host = halfedges.clone({memsrc_e::host});
        // auto turning_pairs_vec = turning_pairs_set._activeKeys.clone({zs::memsrc_e::host});

        turning_pairs_set = turning_pairs_set.clone({zs::memsrc_e::host});
        std::cout << "nm_turning_pairs_set : " << turning_pairs_set.size() << std::endl;
        std::vector<zs::vec<int,2>> turning_pairs_vec{}; turning_pairs_vec.resize(turning_pairs_set.size());

        ompPol(zip(zs::range(turning_pairs_set.size()),turning_pairs_set._activeKeys),[
            &turning_pairs_vec] (auto pi,const auto& pair) mutable {turning_pairs_vec[pi] = pair;});

        
        std::cout << "nm_turning_pairs_vec : " << turning_pairs_vec.size() << std::endl;
        std::cout << "output all the turning pairs : " << std::endl;
        for(int i = 0;i != turning_pairs_vec.size();++i)
            std::cout << "turning_pairs[" << i << "] : " << turning_pairs_vec[i][0] << "\t" << turning_pairs_vec[i][1] << std::endl;

        boundary_pairs_set = boundary_pairs_set.clone({zs::memsrc_e::host});
        std::vector<zs::vec<int,2>> boundary_pairs_vec{};boundary_pairs_vec.resize(boundary_pairs_set.size());
        ompPol(zip(zs::range(boundary_pairs_set.size()),boundary_pairs_set._activeKeys),[
            &boundary_pairs_vec] (auto pi,const auto& pair) mutable {boundary_pairs_vec[pi] = pair;});


        std::vector<int> intersection_pairs_mask{};
        intersection_pairs_mask.resize(csHT.size(),0);

        auto trace_list = std::make_shared<ListObject>();   

        std::cout << "tracing the turning pairs : " << turning_pairs_vec.size() << std::endl;

        for(const auto& trace_start : turning_pairs_vec) {
            std::cout << "trace turning pair[ " << trace_start[0] << "," << trace_start[1] << "]" << std::endl;

            auto pi = csHT_host_proxy.query(trace_start);
            if(pi < 0)
                std::cout << "negative trace_start query" << std::endl;
            if(intersection_pairs_mask[pi]) {
                std::cout << "turning pair[ " << trace_start[0] << "," << trace_start[1] << "] is masked already, skip it" << std::endl;
                continue;
            }
            intersection_pairs_mask[pi] = 1;
            
            std::vector<zs::vec<int,2>> trace_0{},trace_1{},trace{};
            std::vector<int> sides{};


            std::cout << "trace_intersection_loop" << std::endl;
 
            GIA::trace_intersection_loop(
                proxy<omp_space>({},tris_host),
                proxy<omp_space>({},halfedges_host),
                csHT_host_proxy,
                trace_start,
                sides,
                trace,
                trace_0,
                trace_1);

            if(GIA::find_intersection_turning_point(
                proxy<omp_space>({},halfedges_host),
                proxy<omp_space>({},tris_host),
                trace_0[trace_0.size() - 1])) {
                    std::cout << "is_LL_trace\n" << std::endl;
            } else if(GIA::is_boundary_halfedge(proxy<omp_space>({},halfedges_host),trace_0[trace_0.size() - 1][0])) {
                std::cout << "is_LB_trace\n" << std::endl;
            }

            std::cout << "trace_length : " << trace.size() << "\t" << trace_0.size() << "\t" << trace_1.size() << std::endl;

            std::cout << "compute trace baries " << std::endl;

            std::vector<zs::vec<float,5>> trace_baries{};
            trace_baries.resize(trace.size());

            ompPol(zs::range(trace.size()),[
                &trace,&trace_baries,
                source_tag = zs::SmallString(source_tag),
                tris_proxy = proxy<omp_space>({},tris_host),
                halfedges_proxy = proxy<omp_space>({},halfedges_host),
                verts_proxy = proxy<omp_space>({},verts_host)] (int pi) mutable {
                    auto pair = trace[pi];
                    zs::vec<T,2> edge_bary{};
                    zs::vec<T,3> tri_bary{};
                    GIA::compute_HT_intersection_barycentric(verts_proxy,source_tag,
                        tris_proxy,
                        halfedges_proxy,
                        pair,
                        edge_bary,
                        tri_bary);
                    trace_baries[pi] = zs::vec<float,5>{edge_bary[0],edge_bary[1],tri_bary[0],tri_bary[1],tri_bary[2]};
            });

            auto trace_prim = std::make_shared<PrimitiveObject>();
            auto& trace_verts = trace_prim->verts;
            auto& trace_lines = trace_prim->lines;

            trace_verts.resize(trace.size() * 2);
            trace_lines.resize(trace.size() * 2 - 2);

            std::cout << "visualized the L-L trace : " << trace.size() << std::endl;
            for(int i = 0;i != trace.size();++i)
                std::cout << "T[" << i << "] : " << trace[i][0] << "\t" << trace[i][1] << std::endl;


            ompPol(zs::range(trace.size()),[
                &trace,&trace_baries,&sides,
                &trace_verts,&trace_lines,
                dest_tag = zs::SmallString(dest_tag),
                tris_proxy = proxy<omp_space>({},tris_host),
                halfedges_proxy = proxy<omp_space>({},halfedges_host),
                verts_proxy = proxy<omp_space>({},verts_host)] (int pi) mutable {
                    auto pair = trace[pi];
                    auto hi = pair[0];
                    auto ti = pair[1];
                    auto edge = half_edge_get_edge(hi,halfedges_proxy,tris_proxy);
                    auto tri = tris_proxy.pack(dim_c<3>,"inds",ti,int_c);
                    zs::vec<T,3> eps[2] = {};
                    zs::vec<T,3> tps[3] = {};
                    for(int i = 0;i != 2;++i)
                        eps[i] = verts_proxy.pack(dim_c<3>,dest_tag,edge[i]);
                    for(int i = 0;i != 3;++i)
                        tps[i] = verts_proxy.pack(dim_c<3>,dest_tag,tri[i]);
                    
                    zs::vec<T,2> edge_bary{trace_baries[pi][0],trace_baries[pi][1]};
                    zs::vec<T,3> tri_bary{trace_baries[pi][2],trace_baries[pi][3],trace_baries[pi][4]};

                    auto eip = zs::vec<T,3>::zeros();
                    for(int i = 0;i != 2;++i)
                        eip += eps[i] * edge_bary[i];
                    auto tip = zs::vec<T,3>::zeros();
                    for(int i = 0;i != 3;++i)
                        tip += tps[i] * tri_bary[i];
                    
                    if(sides[pi] == 0) {
                        trace_verts[pi * 2 + 0] = eip.to_array();
                        trace_verts[pi * 2 + 1] = tip.to_array();
                    } else {
                        trace_verts[pi * 2 + 0] = tip.to_array();
                        trace_verts[pi * 2 + 1] = eip.to_array();
                    }

                    if(pi * 2 + 2 < trace_verts.size())
                        trace_lines[pi * 2 + 0] = zeno::vec2i{pi * 2 + 0,pi * 2 + 2};
                    if(pi * 2 + 3 < trace_verts.size())
                        trace_lines[pi * 2 + 1] = zeno::vec2i{pi * 2 + 1,pi * 2 + 3};
            });

            
            std::cout << "finish visualizing the L-L trace " << std::endl;

            trace_list->arr.push_back(std::move(trace_prim));
        }

        std::cout << "output trace" << std::endl;

        set_output("trace_list",std::move(trace_list));
    }


};

ZENDEFNODE(VisualizeIntersectionLoops, {
    {
        {"zsparticles"},
        {"string", "source_tag", "x"},
        {"string", "dest_tag", "x"},
    },
    {
        {"list", "trace_list"},
    },
    {
    },
    {"GIA"},
});

};