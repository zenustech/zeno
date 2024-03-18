#pragma once

#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/zpc_tpls/fmt/format.h"
#include "tiled_vector_ops.hpp"

#include "zensim/math/matrix/SparseMatrix.hpp"

#include "zensim/graph/ConnectedComponents.hpp"

#include "zensim/container/Bht.hpp"
#include "zensim/graph/Coloring.hpp"
#include "compute_characteristic_length.hpp"


namespace zeno {

    template<int SIMPLEX_SIZE>
    constexpr int order_indices(zs::vec<int,SIMPLEX_SIZE>& simplex) {
        // constexpr int len = 3;
        int nm_swap = 0;
        for(int i = 0;i != SIMPLEX_SIZE - 1;++i)
            for(int j = 0;j != SIMPLEX_SIZE - 1 - i;++j)
                if(simplex[j] > simplex[j + 1]) {
                    auto tmp = simplex[j];
                    simplex[j] = simplex[j + 1];
                    simplex[j + 1] = tmp;
                    ++nm_swap;
                }
        return nm_swap;
    }    

    template<typename Pol,typename TriTileVec>
    bool is_manifold_check(Pol& pol,const TriTileVec& tris) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>;

        zs::Vector<int> nm_non_manifold_edges{tris.get_allocator(),1};
        nm_non_manifold_edges.setVal(0);
        zs::bht<int,2,int> tab{tris.get_allocator(),tris.size() * 3};
        tab.reset(pol,true);

        pol(zs::range(tris.size()),[
            exec_tag,
            nm_non_manifold_edges = proxy<space>(nm_non_manifold_edges),
            tris = proxy<space>({},tris),
            tab = proxy<space>(tab)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                for(int i = 0;i != 3;++i) {
                    if(auto no = tab.insert(vec2i{tri[(i + 0) % 3],tri[(i + 1) % 3]});no < 0)
                        atomic_add(exec_tag,&nm_non_manifold_edges[0],(int)1);
                }
        });

        return nm_non_manifold_edges.getVal(0) > 0 ? false : true;
    }

    template<typename VecTi, zs::enable_if_all<VecTi::dim == 1, (VecTi::extent <= 4), (VecTi::extent > 1)> = 0>
    constexpr auto elm_to_edges(const zs::VecInterface<VecTi>& elm) {
        using Ti = typename VecTi::value_type;
        constexpr auto CODIM = VecTi::extent;
        constexpr auto NM_EDGES = (CODIM - 1) * (CODIM) / 2;

        zs::vec<zs::vec<Ti,2>, NM_EDGES> out_edges{};
        int nm_out_edges = 0;
        for(int i = 0;i != CODIM;++i)
            for(int j = i + 1;j != CODIM;++j)
                out_edges[nm_out_edges++] = zs::vec<Ti,2>{elm[i],elm[j]};

        return out_edges;
    }

    template<typename Pol>
    int mark_disconnected_island(Pol& pol,
            const zs::Vector<zs::vec<int,2>>& topo,
            // const zs::Vector<bool>& topo_disable_buffer,
            zs::Vector<int>& fasBuffer) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;

        using IV = zs::vec<int,2>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        // setup the incident matrix
        // auto simplex_size = topo.getPropertySize("inds");
        constexpr int simplex_size = 2;

        zs::bcht<IV,int,true,zs::universal_hash<IV>,16> tab{topo.get_allocator(),topo.size() * simplex_size};
        zs::Vector<int> is{topo.get_allocator(),0},js{topo.get_allocator(),0};

        // std::cout << "initialize incident matrix topo" << std::endl;
        pol(range(topo.size()),[
            topo = proxy<space>(topo),

            tab = proxy<space>(tab)] ZS_LAMBDA(int ei) mutable {
                auto a = topo[ei][0];
                auto b = topo[ei][1];
                if(a > b){
                    auto tmp = a;
                    a = b;
                    b = tmp;
                }
                tab.insert(IV{a,b});                    
        });

        auto nmEntries = tab.size();
        // std::cout << "nmEntries of Topo : " << nmEntries << std::endl;
        is.resize(nmEntries);
        js.resize(nmEntries);

        pol(zip(is,js,range(tab._activeKeys)),[] ZS_LAMBDA(int &i,int &j,const auto& ij){
            i = ij[0];
            j = ij[1];
        });
        {
            int offset = is.size();
            is.resize(offset + fasBuffer.size());
            js.resize(offset + fasBuffer.size());
            pol(range(fasBuffer.size()),[is = proxy<space>(is),js = proxy<space>(js),offset] ZS_LAMBDA(int i) mutable {
                is[offset + i] = i;
                js[offset + i] = i;
            });
        }

        zs::SparseMatrix<int,true> spmat{topo.get_allocator(),(int)fasBuffer.size(),(int)fasBuffer.size()};
        spmat.build(pol,(int)fasBuffer.size(),(int)fasBuffer.size(),range(is),range(js),true_c);

        union_find(pol,spmat,range(fasBuffer));
        zs::bcht<int, int, true, zs::universal_hash<int>, 16> vtab{fasBuffer.get_allocator(),fasBuffer.size()};        
        pol(range(fasBuffer.size()),[
            vtab = proxy<space>(vtab),
            fasBuffer = proxy<space>(fasBuffer)] ZS_LAMBDA(int vi) mutable {
                auto fa = fasBuffer[vi];
                while(fa != fasBuffer[fa])
                    fa = fasBuffer[fa];
                fasBuffer[vi] = fa;
                vtab.insert(fa);
        });

        pol(range(fasBuffer.size()),[
            fasBuffer = proxy<space>(fasBuffer),vtab = proxy<space>(vtab)] ZS_LAMBDA(int vi) mutable {
                auto ancestor = fasBuffer[vi];
                auto setNo = vtab.query(ancestor);
                fasBuffer[vi] = setNo;
        });

        auto nmSets = vtab.size();
        return nmSets;
        // fmt::print("{} disjoint sets in total.\n",nmSets);
    }


    template<typename Pol>
    int mark_disconnected_island(Pol& pol,
            const zs::SparseMatrix<int,true>& spmat,
            zs::Vector<int>& fasBuffer) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;

        using IV = zs::vec<int,2>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        if(spmat.rows() != spmat.cols()){
            throw std::runtime_error("mark_disconnected_island : only square incident matrix is supported");
        }
        if(spmat.rows() != fasBuffer.size() || spmat.cols() != fasBuffer.size()){
            throw std::runtime_error("mark_diconnnected_island : the input fasBuffer size and spmat size not match");
        }

        union_find(pol,spmat,range(fasBuffer));
        zs::bcht<int, int, true, zs::universal_hash<int>, 16> vtab{fasBuffer.get_allocator(),fasBuffer.size()};        
        pol(range(fasBuffer.size()),[
            vtab = proxy<space>(vtab),
            fasBuffer = proxy<space>(fasBuffer)] ZS_LAMBDA(int vi) mutable {
                auto fa = fasBuffer[vi];
                while(fa != fasBuffer[fa])
                    fa = fasBuffer[fa];
                fasBuffer[vi] = fa;
                vtab.insert(fa);
        });

        pol(range(fasBuffer.size()),[
            fasBuffer = proxy<space>(fasBuffer),vtab = proxy<space>(vtab)] ZS_LAMBDA(int vi) mutable {
                auto ancestor = fasBuffer[vi];
                auto setNo = vtab.query(ancestor);
                fasBuffer[vi] = setNo;
        });

        auto nmSets = vtab.size();
        return nmSets;
    }

    template<typename Pol>
    int mark_disconnected_island(Pol& pol,
            const zs::Vector<zs::vec<int,2>>& topo,
            const zs::bht<int,1,int>& disable_points,
            const zs::bht<int,2,int>& disable_lines,
            zs::Vector<int>& fasBuffer
    ) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using table_vec2i_type = zs::bht<int,2,int>;
        // using table_int_type = zs::bcht<int,int,true,zs::universal_hash<int>,16>;
        using table_int_type = zs::bht<int,1,int>;
        using IV = zs::vec<int,2>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;    

        table_vec2i_type tab{topo.get_allocator(),topo.size() * 2};
        tab.reset(pol,true);
        zs::Vector<int> is{topo.get_allocator(),0},js{topo.get_allocator(),0};
        // bool use_disable = topo_disable_buffer.size() == fasBuffer.size();

        pol(range(topo.size()),[
            topo = proxy<space>(topo),
            disable_points = proxy<space>(disable_points),
            disable_lines = proxy<space>(disable_lines),
            tab = proxy<space>(tab)] ZS_LAMBDA(int ei) mutable {
                auto a = topo[ei][0];
                auto b = topo[ei][1];

                auto setNo = disable_points.query(a);
                if(setNo >= 0){
                    // printf("skip line [%d %d] due to corner point[%d]\n",a,b,a);
                    return;
                }
                setNo = disable_points.query(b);
                if(setNo >= 0){
                    // printf("skip line [%d %d] due to corner point[%d]\n",a,b,b);
                    return;
                }

                if(a < 0 || b < 0)
                    return;
                if(a > b){
                    auto tmp = a;
                    a = b;
                    b = tmp;
                }
                setNo = disable_lines.query(vec2i{a,b});
                if(setNo >= 0)
                    return;
                // setNo = disable_lines.query(vec2i{b,a});
                // if(setNo >= 0)
                //     return;

                tab.insert(IV{a,b});                    
        });

        auto nmEntries = tab.size();
        // std::cout << "nmEntries of Topo : " << nmEntries << std::endl;
        is.resize(nmEntries);
        js.resize(nmEntries);

        pol(zip(is,js,range(tab._activeKeys)),[] ZS_LAMBDA(int &i,int &j,const auto& ij){
            i = ij[0];
            j = ij[1];
        });

        zs::SparseMatrix<int,true> spmat{topo.get_allocator(),(int)fasBuffer.size(),(int)fasBuffer.size()};
        spmat.build(pol,(int)fasBuffer.size(),(int)fasBuffer.size(),range(is),range(js),true_c);

        return mark_disconnected_island(pol,spmat,fasBuffer);    
    }


    template<typename Pol>
    int mark_disconnected_island(Pol& pol,
            const zs::Vector<zs::vec<int,2>>& topo,
            const zs::bht<int,1,int>& disable_points,
            const zs::bht<int,2,int>& disable_lines,
            zs::Vector<int>& fasBuffer,
            zs::bht<int,2,int>& tab
            ) {
        using namespace zs;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using table_vec2i_type = zs::bht<int,2,int>;
        // using table_int_type = zs::bcht<int,int,true,zs::universal_hash<int>,16>;
        using table_int_type = zs::bht<int,1,int>;
        using IV = zs::vec<int,2>;

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;    

        // table_vec2i_type tab{topo.get_allocator(),topo.size() * 2};
        tab.reset(pol,true);
        zs::Vector<int> is{topo.get_allocator(),0},js{topo.get_allocator(),0};
        // bool use_disable = topo_disable_buffer.size() == fasBuffer.size();

        pol(range(topo.size()),[
            topo = proxy<space>(topo),
            disable_points = proxy<space>(disable_points),
            disable_lines = proxy<space>(disable_lines),
            tab = proxy<space>(tab)] ZS_LAMBDA(int ei) mutable {
                auto a = topo[ei][0];
                auto b = topo[ei][1];

                auto setNo = disable_points.query(a);
                if(setNo >= 0){
                    // printf("skip line [%d %d] due to corner point[%d]\n",a,b,a);
                    return;
                }
                setNo = disable_points.query(b);
                if(setNo >= 0){
                    // printf("skip line [%d %d] due to corner point[%d]\n",a,b,b);
                    return;
                }

                if(a < 0 || b < 0)
                    return;
                if(a > b){
                    auto tmp = a;
                    a = b;
                    b = tmp;
                }
                setNo = disable_lines.query(vec2i{a,b});
                if(setNo >= 0)
                    return;
                // setNo = disable_lines.query(vec2i{b,a});
                // if(setNo >= 0)
                //     return;

                tab.insert(IV{a,b});                    
        });

        auto nmEntries = tab.size();
        // std::cout << "nmEntries of Topo : " << nmEntries << std::endl;
        is.resize(nmEntries);
        js.resize(nmEntries);

        pol(zip(is,js,range(tab._activeKeys)),[] ZS_LAMBDA(int &i,int &j,const auto& ij){
            i = ij[0];
            j = ij[1];
        });

        zs::SparseMatrix<int,true> spmat{topo.get_allocator(),(int)fasBuffer.size(),(int)fasBuffer.size()};
        spmat.build(pol,(int)fasBuffer.size(),(int)fasBuffer.size(),range(is),range(js),true_c);

        return mark_disconnected_island(pol,spmat,fasBuffer);    
    }


    template<typename Pol,typename TopoRangT/*,zs::enable_if_all<VecTI::dim == 1, (VecTI::extent >= 2), (VecTI::etent <= 4)> = 0*/>
    void topological_incidence_matrix(Pol& pol,
            // size_t nm_points,
            const TopoRangT& topos,
            zs::SparseMatrix<zs::u32,true>& spmat,bool output_debug_inform = false) {
        using namespace zs;
        using ICoord = zs::vec<int, 2>;
        // constexpr auto CDIM = VecTI::extent;
        constexpr auto CDIM = RM_CVREF_T(topos[0])::extent;
        constexpr auto space = Pol::exec_tag::value;
        constexpr auto execTag = wrapv<space>{};

        zs::Vector<int> max_pi_vec{topos.get_allocator(),1};
        max_pi_vec.setVal(0);
        pol(zs::range(topos),[max_pi_vec = proxy<space>(max_pi_vec),execTag,CDIM] ZS_LAMBDA(const auto& topo) mutable {
            for(int i = 0;i != CDIM;++i) 
                if(topo[i] >= 0)
                    atomic_max(execTag,&max_pi_vec[0],(int)topo[i]);
        });
        auto nm_points = max_pi_vec.getVal(0) + 1; 

        zs::Vector<int> exclusive_offsets{topos.get_allocator(),(size_t)(nm_points)};
        zs::Vector<int> p2ts{topos.get_allocator(),0};
        zs::Vector<int> max_tp_incidences{topos.get_allocator(),1};
        zs::Vector<int> cnts{topos.get_allocator(),(size_t)nm_points};

        {    
            zs::Vector<int> tab_buffer{topos.get_allocator(), topos.size() * CDIM};
            bht<int,2,int> tab{topos.get_allocator(), topos.size() * CDIM};
            tab.reset(pol, true);

            // cnts.reset(0);
            pol(zs::range(cnts),[] ZS_LAMBDA(auto& cnt) {cnt = 0;});
            pol(zs::range(topos.size()),[
                topos = proxy<space>(topos),
                tab = proxy<space>(tab),
                tab_buffer = proxy<space>(tab_buffer),
                cnts = proxy<space>(cnts)] ZS_LAMBDA(int ti) mutable {
                    for(int i = 0;i != CDIM;++i) {
                        if(topos[ti][i] < 0)
                            break;
                        else{
                            auto local_offset = atomic_add(execTag,&cnts[topos[ti][i]], (int)1);
                            if(auto id = tab.insert(ICoord{topos[ti][i],(int)local_offset}); id != bht<int,2,int>::sentinel_v){
                                tab_buffer[id] = ti;
                            }
                        }
                    }
            });

            std::cout << "finish computing tab_buffer" << std::endl;
            // pol(zs::range(cnts.size()),[cnts = proxy<space>(cnts)] ZS_LAMBDA(int pi) mutable {printf("cnts[%d] = %d\n",pi,cnts[pi]);});
            pol(zs::range(exclusive_offsets),[] ZS_LAMBDA(auto& eoffset) {eoffset = 0;});

            exclusive_scan(pol,std::begin(cnts),std::end(cnts),std::begin(exclusive_offsets));
            // pol(zs::range(exclusive_offsets.size()),[exclusive_offsets = proxy<space>(exclusive_offsets)] ZS_LAMBDA(int pi) mutable {printf("eooffset[%d] = %d\n",pi,exclusive_offsets[pi]);});
            auto nmPTEntries = exclusive_offsets.getVal(nm_points - 1) + cnts.getVal(nm_points - 1);
            // std::cout << "nmPTEntries " << nmPTEntries << std::endl;
            p2ts.resize(nmPTEntries);


            max_tp_incidences.setVal(0);
            pol(zs::range(nm_points),[
                topos = proxy<space>(topos),
                tab = proxy<space>(tab),
                cnts = proxy<space>(cnts),
                execTag,
                max_tp_incidences = proxy<space>(max_tp_incidences),
                p2ts = proxy<space>(p2ts),
                tab_buffer = proxy<space>(tab_buffer),
                exclusive_offsets = proxy<space>(exclusive_offsets)] ZS_LAMBDA(int pi) mutable {
                    auto pt_count = cnts[pi];
                    atomic_max(execTag,&max_tp_incidences[0],pt_count);
                    auto ex_offset = exclusive_offsets[pi];
                    for(int i = 0;i != pt_count;++i)
                        if(auto id = tab.query(ICoord{pi,i}); id != bht<int,2,int>::sentinel_v) {
                            auto ti = tab_buffer[id];
                            p2ts[ex_offset + i] = ti;
                            // printf("p[%d] -> t[%d]\n",pi,ti);
                        }
            });
        }


        bht<int,2,int> tij_tab{topos.get_allocator(), topos.size() * max_tp_incidences.getVal(0) * CDIM};
        tij_tab.reset(pol,true);

        pol(range(topos.size()),[
            topos = proxy<space>(topos),
            p2ts = proxy<space>(p2ts),
            tij_tab = proxy<space>(tij_tab),
            execTag,
            CDIM,
            cnts = proxy<space>(cnts),
            exclusive_offsets = proxy<space>(exclusive_offsets)] ZS_LAMBDA(int ti) mutable {
                auto topo = topos[ti];
                for(int i = 0;i != CDIM;++i){
                    auto vi = topo[i];
                    if(vi < 0)
                        return;
                    auto ex_offset = exclusive_offsets[vi];
                    auto nm_nts = cnts[vi];
                    for(int j = 0;j != nm_nts;++j) {
                        auto nti = p2ts[ex_offset + j];
                        if(nti < ti)
                            continue;
                        tij_tab.insert(ICoord{ti,nti});
                    }
                }
        });

        std::cout << "finish computing tij_tab" << std::endl;

        zs::Vector<int> is{topos.get_allocator(),tij_tab.size()};
        zs::Vector<int> js{topos.get_allocator(),tij_tab.size()};
        pol(zip(zs::range(tij_tab.size()),zs::range(tij_tab._activeKeys)),[
                is = proxy<space>(is),
                js = proxy<space>(js),
                output_debug_inform = output_debug_inform] ZS_LAMBDA(auto idx,const auto& pair) {
            is[idx] = pair[0];js[idx] = pair[1];
            if(output_debug_inform)
                printf("pair[%d] : %d %d\n",idx,pair[0],pair[1]);
        });

        // pol(zs::range(is.size()),[is = proxy<space>(is),js = proxy<space>(js)] ZS_LAMBDA(int i) mutable {printf("ijs[%d] : %d %d\n",i,is[i],js[i]);});
        // std::cout << "topos.size() = " << topos.size() << std::endl;
        // for(int i = 0;i != topos.size();++i)
        //     std::cout << topos.getVal(i)[0] << "\t" << topos.getVal(i)[1] << std::endl;

        // spmat = zs::SparseMatrix<u32,true>{topos.get_allocator(),(int)topos.size(),(int)topos.size()};
        std::cout << "build sparse matrix" << std::endl;
        spmat.build(pol,(int)topos.size(),(int)topos.size(),zs::range(is),zs::range(js)/*,zs::range(rs)*/,zs::true_c);
        // spmat.localOrdering(pol,zs::false_c);
        spmat._vals.resize(spmat.nnz());
        pol(spmat._vals, []ZS_LAMBDA(u32 &v) { v = 1; });
        // std::cout << "done connectivity graph build" << std::endl;

        // spmat._vals.reset((int)1);
    }

    template<typename Pol,typename TopoRangeT,typename ColorRangeT>
    void topological_coloring(Pol& pol,
            // int nm_points,
            const TopoRangeT& topo,
            ColorRangeT& colors,
            bool output_debug_information = false) {
        using namespace zs;
        constexpr auto space = Pol::exec_tag::value;
        using Ti = RM_CVREF_T(colors[0]);

        if(output_debug_information)
            std::cout << "do coloring with topos : " << topo.size() << std::endl;


        colors.resize(topo.size());
        zs::SparseMatrix<u32,true> topo_incidence_matrix{topo.get_allocator(),(int)topo.size(),(int)topo.size()};
        std::cout << "compute incidence matrix " << std::endl;
        

        topological_incidence_matrix(pol,topo,topo_incidence_matrix,output_debug_information);
        std::cout << "finish compute incidence matrix " << std::endl;

        auto ompPol = omp_exec();
        constexpr auto omp_space = execspace_e::openmp;
        zs::Vector<u32> weights(/*topo.get_allocator(),*/topo.size());
        {
            bht<int, 1, int> tab{weights.get_allocator(),topo.size() * 100};
            tab.reset(ompPol, true);
            ompPol(enumerate(weights), [tab1 = proxy<omp_space>(tab)] (int seed, u32 &w) mutable {
                using tab_t = RM_CVREF_T(tab);
                std::mt19937 rng;
                rng.seed(seed);
                u32 v = rng() % (u32)4294967291u;
                // prevent weight duplications
                while (tab1.insert(v) != tab_t::sentinel_v)
                    v = rng() % (u32)4294967291u;
                w = v;
            });
        }

        // pol(zs::range())
        weights = weights.clone(colors.memoryLocation());
        // for(int i = 0;i != weights.size();++i)
        //     printf("w[%d] : %u\n",i,weights.getVal(i));
        std::cout << "do maximum set " << std::endl;
        auto iterRef = maximum_independent_sets(pol, topo_incidence_matrix, weights, colors);
        std::cout << "nm_colors : " << iterRef << std::endl;
        pol(zs::range(colors),[] ZS_LAMBDA(auto& clr) mutable {clr = clr - (Ti)1;});

    }

    template<typename Pol,typename REORDERED_MAP_RANGE,typename COLOR_RANGE,typename EXCLUSIVE_OFFSET_RANGE>
    void sort_topology_by_coloring_tag(Pol& pol,
            const COLOR_RANGE& colors,
            REORDERED_MAP_RANGE& reordered_map,
            EXCLUSIVE_OFFSET_RANGE& offset_out) {
        using namespace zs;
        constexpr auto space = Pol::exec_tag::value;
        constexpr auto exec_tag = wrapv<space>{};

        // zs::Vector<int> reordered_map{colors.get_allocator(),colors.size()};
        reordered_map.resize(colors.size());
        zs::Vector<int> max_color{colors.get_allocator(),1};
        max_color.setVal(0);

        pol(zs::range(colors.size()),[
            colors = proxy<space>(colors),\
            exec_tag = exec_tag,
            max_color = proxy<space>(max_color)] ZS_LAMBDA(int ci) mutable {
                auto color = (int)colors[ci];
                atomic_max(exec_tag,&max_color[0],color);
        });

        size_t nm_total_colors = max_color.getVal(0) + 1;
        // zs::bht<int,1,int> color_buffer{}
        zs::Vector<int> nm_colors{colors.get_allocator(),nm_total_colors};
        pol(zs::range(nm_colors),[] ZS_LAMBDA(auto& nclr) mutable {nclr = 0;});
        pol(zs::range(colors),[nm_colors = proxy<space>(nm_colors),exec_tag] ZS_LAMBDA(const auto& clrf) mutable {
            auto clr = (int)clrf;
            atomic_add(exec_tag,&nm_colors[clr],1);
        });

        zs::Vector<int> exclusive_offsets{colors.get_allocator(),nm_total_colors};
        pol(zs::range(exclusive_offsets),[] ZS_LAMBDA(auto& eoffset) {eoffset = 0;});
        exclusive_scan(pol,std::begin(nm_colors),std::end(nm_colors),std::begin(exclusive_offsets));
        pol(zs::range(nm_colors),[] ZS_LAMBDA(auto& nclr) {nclr = 0;});

        offset_out.resize(nm_total_colors);
    
        pol(zip(zs::range(exclusive_offsets.size()),exclusive_offsets),[offset_out = proxy<space>(offset_out)] ZS_LAMBDA(auto i,auto offset) mutable {offset_out[i] = offset;});
        pol(zs::range(colors.size()),[
                nm_colors = proxy<space>(nm_colors),
                colors = proxy<space>(colors),
                exec_tag,
                exclusive_offsets = proxy<space>(exclusive_offsets),
                reordered_map = proxy<space>(reordered_map)] ZS_LAMBDA(auto ci) mutable {
            auto clr = (int)colors[ci];
            auto offset = atomic_add(exec_tag,&nm_colors[clr],1);
            auto eoffset = exclusive_offsets[clr];

            reordered_map[eoffset + offset] = ci;
        });        

        // zs::Vector<VecTI> topos_copy{topos.get_allocator(),topos.size()};
        // pol(zip(zs::range(topos.size()),topos),[topos_copy = proxy<space>(topos_copy)] ZS_LAMBDA(auto ti,const auto& topo) mutable {topos_copy[ti] = topo;});

        // pol(zip(zs::range(topos.size()),topos),[
        //     topos_copy = proxy<space>(topos_copy),
        //     reordered_map = proxy<space>(reordered_map)] ZS_LAMBDA(auto ti,auto& topo) mutable {topo = topos_copy[reordered_map[ti]];});
    }

    template<typename Pol,typename TopoTileVec,int codim,typename VecTi = zs::vec<int,codim>>
    zs::Vector<VecTi> tilevec_topo_to_zsvec_topo(Pol& pol,const TopoTileVec& source,zs::wrapv<codim>) {
        zs::Vector<VecTi> out_topo{source.get_allocator(),source.size()};
        auto sr = zs::range(source, "inds", zs::dim_c<codim>, zs::int_c);
        pol(zip(sr, out_topo), []ZS_LAMBDA(auto id, VecTi& dst) mutable {
            if constexpr (std::is_integral_v<RM_CVREF_T(id)>)
                dst[0] = id;
            else
                dst = id;
        });
        return out_topo;
    }

    // the topos:  triA: [idx0,idx2,idx3] triB: [idx1,idx3,idx2]
    template<typename Pol,typename TriTileVec,typename HalfEdgeTileVec>
    void retrieve_tri_bending_topology(Pol& pol,
        const TriTileVec& tris,
        const HalfEdgeTileVec& halfedges,
        zs::Vector<zs::vec<int,4>>& tb_topos) {
            using namespace zs;
            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};

            // zs::Vector<int> nm_interior_edges{halfedges.get_allocator(),1};
            // nm_interior_edges.setVal(0);

            zs::bht<int,1,int> interior_edges{halfedges.get_allocator(),halfedges.size()};
            interior_edges.reset(pol,true);

            pol(zs::range(halfedges.size()),[
                halfedges = proxy<space>({},halfedges),
                exec_tag,
                interior_edges = proxy<space>(interior_edges)] ZS_LAMBDA(int hi) mutable {
                    auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    // the boundary halfedge will return -1 for opposite_he here, so it is automatically neglected
                    if(ohi < hi)
                        return;
                    interior_edges.insert(hi);
            });
        
            tb_topos.resize(interior_edges.size());
            pol(zs::zip(zs::range(interior_edges.size()),interior_edges._activeKeys),[
                tb_topos = proxy<space>(tb_topos),
                halfedges = proxy<space>({},halfedges),
                tris = proxy<space>({},tris)] ZS_LAMBDA(auto id,auto hi_vec) mutable {
                    auto hi = hi_vec[0];
                    auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto vid = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    auto oti = zs::reinterpret_bits<int>(halfedges("to_face",ohi));
                    auto ovid = zs::reinterpret_bits<int>(halfedges("local_vertex_id",ohi));

                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    auto otri = tris.pack(dim_c<3>,"inds",oti,int_c);

                    // tb_topos[id] = zs::vec<int,4>(tri[(vid + 0) % 3],tri[(vid + 1) % 3],tri[(vid + 2) % 3],otri[(ovid + 2) % 3]);
                    tb_topos[id] = zs::vec<int,4>(tri[(vid + 2) % 3],otri[(ovid + 2) % 3],tri[(vid + 0) % 3],tri[(vid + 1) % 3]);
            });
    }

    template<typename Pol,typename TriTileVec,typename HalfEdgeTileVec>
    void retrieve_dihedral_spring_topology(Pol& pol,
        const TriTileVec& tris,
        const HalfEdgeTileVec& halfedges,
        zs::Vector<zs::vec<int,2>>& ds_topos) {
            using namespace zs;
            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};

            // zs::Vector<int> nm_interior_edges{halfedges.get_allocator(),1};
            // nm_interior_edges.setVal(0);

            zs::bht<int,1,int> interior_edges{halfedges.get_allocator(),halfedges.size()};
            interior_edges.reset(pol,true);

            pol(zs::range(halfedges.size()),[
                halfedges = proxy<space>({},halfedges),
                exec_tag,
                interior_edges = proxy<space>(interior_edges)] ZS_LAMBDA(int hi) mutable {
                    auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    // the boundary halfedge will return -1 for opposite_he here, so it is automatically neglected
                    if(ohi < hi)
                        return;
                    interior_edges.insert(hi);
            });
        
            ds_topos.resize(interior_edges.size());
            pol(zs::zip(zs::range(interior_edges.size()),interior_edges._activeKeys),[
                ds_topos = proxy<space>(ds_topos),
                halfedges = proxy<space>({},halfedges),
                tris = proxy<space>({},tris)] ZS_LAMBDA(auto id,auto hi_vec) mutable {
                    auto hi = hi_vec[0];
                    auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto vid = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    auto oti = zs::reinterpret_bits<int>(halfedges("to_face",ohi));
                    auto ovid = zs::reinterpret_bits<int>(halfedges("local_vertex_id",ohi));

                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    auto otri = tris.pack(dim_c<3>,"inds",oti,int_c);

                    // ds_topos[id] = zs::vec<int,4>(tri[(vid + 0) % 3],tri[(vid + 1) % 3],tri[(vid + 2) % 3],otri[(ovid + 2) % 3]);
                    ds_topos[id] = zs::vec<int,2>(tri[(vid + 2) % 3],otri[(ovid + 2) % 3]);
            });
    }

    template<typename Pol,typename VecTi,typename Ti = typename VecTi::value_type,int CDIM = VecTi::extent,int NM_EDGES = CDIM * (CDIM - 1) / 2>
    void retrieve_edges_topology(Pol& pol,
        const zs::Vector<VecTi>& src_topos,
        zs::Vector<zs::vec<Ti,2>>& edges_topos) {
            using namespace zs;
            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            // constexpr auto CDIM = VecTi::extent;
            // constexpr auto NM_EDGES =  CDIM * (CDIM - 1) / 2;

            zs::bht<int,2,int> edges_tab{src_topos.get_allocator(),src_topos.size() * 6};
            edges_tab.reset(pol,true);
            pol(zs::range(src_topos.size()),[
                src_topos = proxy<space>(src_topos),
                edges_tab = proxy<space>(edges_tab)] ZS_LAMBDA(int ei) mutable {
                    auto elm_edges = elm_to_edges(src_topos[ei]);
                    for(int i = 0;i != NM_EDGES;++i) {
                        auto edge = elm_edges[i];
                        if(edge[0] < edge[1])
                            edges_tab.insert(zs::vec<Ti,2>{edge[0],edge[1]});
                        else
                            edges_tab.insert(zs::vec<Ti,2>{edge[1],edge[0]});
                    }
            });

            edges_topos.resize(edges_tab.size());
            pol(zip(zs::range(edges_tab.size()),zs::range(edges_tab._activeKeys)),[
                edges_topos = proxy<space>(edges_topos)] ZS_LAMBDA(auto ei,const auto& edge){edges_topos[ei] = edge;});
    }

    template<typename Pol,typename TopoTileVec>
    void reorder_topology(Pol& pol,
        const TopoTileVec& reorder_map,
        TopoTileVec& dst) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        using T = typename RM_CVREF_T(reorder_map)::value_type;

        zs::bcht<int,int,true,zs::universal_hash<int>,16> v2p_tab{reorder_map.get_allocator(),reorder_map.size()};
        zs::Vector<int> v2p_buffer{reorder_map.get_allocator(),reorder_map.size()};
        pol(zs::range(reorder_map.size()),[
            points = proxy<space>({},reorder_map),
            v2p_buffer = proxy<space>(v2p_buffer),
            v2p_tab = proxy<space>(v2p_tab)] ZS_LAMBDA(int pi) mutable {
                auto vi = zs::reinterpret_bits<int>(points("inds",pi));
                auto vNo = v2p_tab.insert(vi);
                v2p_buffer[vNo] = pi;
        });  

        auto simplex_size = dst.getPropertySize("inds");

        pol(zs::range(dst.size()),[
            dst = proxy<space>({},dst),
            simplex_size,
            reorder_map = proxy<space>({},reorder_map),
            v2p_tab = proxy<space>(v2p_tab),
            v2p_buffer = proxy<space>(v2p_buffer)] ZS_LAMBDA(int ti) mutable {
                for(int i = 0;i != simplex_size;++i) {
                    auto di = zs::reinterpret_bits<int>(dst("inds",i,ti));
                    auto vNo = v2p_tab.query(di);
                    auto pi = v2p_buffer[vNo];
                    dst("inds",i,ti) = zs::reinterpret_bits<T>(pi);
                }
        });      
    }

    template<typename Pol,typename SampleTileVec,typename TopoTileVec>
    void topological_sample(Pol& pol,
        const TopoTileVec& points,
        const SampleTileVec& verts,
        const zs::SmallString& attr_name,
        SampleTileVec& dst
    ) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        using T = typename RM_CVREF_T(verts)::value_type;    

        dst.resize(points.size());
        int attr_dim = verts.getPropertySize(attr_name);

        pol(zs::range(points.size()),[
            dst = proxy<space>({},dst),
            points = proxy<space>({},points),
            verts = proxy<space>({},verts),
            attr_dim,
            attr_name = zs::SmallString(attr_name)] ZS_LAMBDA(int pi) mutable {
                auto vi = reinterpret_bits<int>(points("inds",pi));
                for(int i = 0;i != attr_dim;++i)
                    dst(attr_name,i,pi) = verts(attr_name,i,vi);
        });    
    }

    template<typename Pol,typename SampleTileVec,typename TopoTileVec>
    void topological_sample(Pol& pol,
        const TopoTileVec& points,
        const SampleTileVec& src,
        const std::string& src_attr_name,
        SampleTileVec& dst,
        const std::string& dst_attr_name) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        using T = typename SampleTileVec::value_type;    

        dst.resize(points.size());
        int attr_dim = src.getPropertySize(src_attr_name);

        pol(zs::range(points.size()),[
            dst = proxy<space>({},dst),
            points = proxy<space>({},points),
            src = proxy<space>({},src),
            attr_dim,
            src_attr_name = zs::SmallString(src_attr_name),
            dst_attr_name = zs::SmallString(dst_attr_name)] ZS_LAMBDA(int pi) mutable {
                auto vi = reinterpret_bits<int>(points("inds",pi));
                for(int i = 0;i != attr_dim;++i)
                    dst(dst_attr_name,i,pi) = src(src_attr_name,i,vi);
        });    
    }

};