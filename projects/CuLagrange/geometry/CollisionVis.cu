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


#include <iostream>


#define COLLISION_VIS_DEBUG

#define MAX_FP_COLLISION_PAIRS 6

namespace zeno {

    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using mat3 = zs::vec<T,3,3>;
    using mat4 = zs::vec<T,4,4>;
    // using vec2i = zs::vec<int,2>;
    // using vec2i = zs::vec<int,2>;
    // using vec3i = zs::vec<int,3>;
    // using vec4i = zs::vec<int,4>;


    // for each triangle, find the three incident triangles
    // TODO: build a half edge structure
    struct ZSInitSurfaceTopoConnect : INode {

        void compute_surface_neighbors(zs::CudaExecutionPolicy &pol, typename ZenoParticles::particles_t &sfs,
                                    typename ZenoParticles::particles_t &ses, typename ZenoParticles::particles_t &svs) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            using vec2i = zs::vec<int, 2>;
            using vec3i = zs::vec<int, 3>;
            sfs.append_channels(pol, {{"ff_inds", 3}, {"fe_inds", 3}, {"fp_inds", 3}});
            ses.append_channels(pol, {{"fe_inds", 2},{"ep_inds",2}});

            fmt::print("sfs size: {}, ses size: {}, svs size: {}\n", sfs.size(), ses.size(), svs.size());

            bcht<vec2i, int, true, universal_hash<vec2i>, 32> etab{sfs.get_allocator(), sfs.size() * 3};
            Vector<int> sfi{sfs.get_allocator(), sfs.size() * 3}; // surftri indices corresponding to edges in the table

            bcht<int,int,true, universal_hash<int>,32> ptab(svs.get_allocator(),svs.size());
            Vector<int> spi{svs.get_allocator(),svs.size()};

            /// @brief compute hash table
            {
                // compute directed edge to triangle idx hash table
                pol(range(sfs.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs),
                                        sfi = proxy<space>(sfi)] __device__(int ti) mutable {
                    auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
                    for (int i = 0; i != 3; ++i)
                        if (auto no = etab.insert(vec2i{tri[i], tri[(i + 1) % 3]}); no >= 0) {
                            sfi[no] = ti;
                        } else {
                            auto oti = sfi[etab.query(vec2i{tri[i], tri[(i + 1) % 3]})];
                            auto otri = sfs.pack(dim_c<3>, "inds", oti).reinterpret_bits(int_c);
                            printf("the same directed edge <%d, %d> has been inserted twice! original sfi %d <%d, %d, %d>, cur "
                                "%d <%d, %d, %d>\n",
                                tri[i], tri[(i + 1) % 3], oti, otri[0], otri[1], otri[2], ti, tri[0], tri[1], tri[2]);
                        }
                });
                // // compute surface point to vert hash table
                // pol(range(svs.size()),[ptab = proxy<space>(ptab),svs = proxy<space>({},svs),
                //     spi = proxy<space>(spi)] __device__(int pi) mutable {
                //         auto pidx = reinterpret_bits<int>(svs("inds",pi));
                //         if(auto no = ptab.insert(pidx); no >= 0)
                //             spi[no] = pi;
                //         else {
                //             auto opi = spi[ptab.query(pidx)];
                //             auto opidx = reinterpret_bits<int>(svs("inds",opi));
                //             printf("the same surface point <%d> has been inserted twice! origin svi %d <%d>, cur "
                //                 "%d <%d>\n",
                //                 pidx,opi,opidx,pi,pidx);
                //         }
                // });
            }
            /// @brief compute ep neighbors
            // {
            //     pol(range(ses.size()),[ptab = proxy<space>(ptab),ses = proxy<space>({},ses),
            //         svs = proxy<space>({},svs),spi = proxy<space>(spi)] __device__(int ei) mutable {
            //             auto neighpIds = vec2i::uniform(-1);
            //             auto edge = ses.pack(dim_c<2>,"inds",ei).reinterpret_bits(int_c);
            //             for(int i = 0;i != 2;++i)
            //                 if(auto no = ptab.query(edge[i]);no >= 0) {
            //                     neighpIds[i] = spi[no];
            //                 }
            //             ses.tuple(dim_c<2>,"ep_inds",ei) = neighpIds.reinterpret_bits(float_c);
            //     });
            // } 

            /// @brief compute ff neighbors
            {
                pol(range(sfs.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs),
                                        sfi = proxy<space>(sfi)] __device__(int ti) mutable {
                    auto neighborIds = vec3i::uniform(-1);
                    auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
                    for (int i = 0; i != 3; ++i)
                        if (auto no = etab.query(vec2i{tri[(i + 1) % 3], tri[i]}); no >= 0) {
                            neighborIds[i] = sfi[no];
                        }
                    sfs.tuple(dim_c<3>, "ff_inds", ti) = neighborIds.reinterpret_bits(float_c);
                });
            }
            /// @brief compute fe neighbors
            {
                auto sfindsOffset = sfs.getPropertyOffset("inds");
                auto sfFeIndsOffset = sfs.getPropertyOffset("fe_inds");
                auto seFeIndsOffset = ses.getPropertyOffset("fe_inds");
                pol(range(ses.size()),
                    [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs), ses = proxy<space>({}, ses),
                    sfi = proxy<space>(sfi), sfindsOffset, sfFeIndsOffset, seFeIndsOffset] __device__(int li) mutable {
                        auto findLineIdInTri = [](const auto &tri, int v0, int v1) -> int {
                            for (int loc = 0; loc < 3; ++loc)
                                if (tri[loc] == v0 && tri[(loc + 1) % 3] == v1)
                                    return loc;
                            return -1;
                        };
                        auto neighborTris = vec2i::uniform(-1);
                        auto line = ses.pack(dim_c<2>, "inds", li).reinterpret_bits(int_c);

                        {
                            if (auto no = etab.query(line); no >= 0) {
                                // tri
                                auto triNo = sfi[no];
                                auto tri = sfs.pack(dim_c<3>, sfindsOffset, triNo).reinterpret_bits(int_c);
                                auto loc = findLineIdInTri(tri, line[0], line[1]);
                                if (loc == -1) {
                                    printf("ridiculous, this edge <%d, %d> does not belong to tri <%d, %d, %d>\n", line[0],
                                        line[1], tri[0], tri[1], tri[2]);
                                    return;
                                }
                                sfs(sfFeIndsOffset + loc, triNo) = li;
                                // edge
                                neighborTris[0] = triNo;
                            }
                        }
                        vec2i rline{line[1], line[0]};
                        {
                            if (auto no = etab.query(rline); no >= 0) {
                                // tri
                                auto triNo = sfi[no];
                                auto tri = sfs.pack(dim_c<3>, sfindsOffset, triNo).reinterpret_bits(int_c);
                                auto loc = findLineIdInTri(tri, rline[0], rline[1]);
                                if (loc == -1) {
                                    printf("ridiculous, this edge <%d, %d> does not belong to tri <%d, %d, %d>\n", rline[0],
                                        rline[1], tri[0], tri[1], tri[2]);
                                    return;
                                }
                                sfs(sfFeIndsOffset + loc, triNo) = li;
                                // edge
                                neighborTris[1] = triNo;
                            }
                        }
                        ses.tuple(dim_c<2>, seFeIndsOffset, li) = neighborTris.reinterpret_bits(float_c);
                    });
            }
            /// @brief compute fp neighbors
            /// @note  surface vertex index is not necessarily consecutive, thus hashing
            {
                bcht<int, int, true, universal_hash<int>, 32> vtab{svs.get_allocator(), svs.size()};
                Vector<int> svi{etab.get_allocator(), svs.size()}; // surftri indices corresponding to edges in the table
                // svs
                pol(range(svs.size()), [vtab = proxy<space>(vtab), svs = proxy<space>({}, svs),
                                        svi = proxy<space>(svi)] __device__(int vi) mutable {
                    int vert = reinterpret_bits<int>(svs("inds", vi));
                    if (auto no = vtab.insert(vert); no >= 0)
                        svi[no] = vi;
                });
                //
                pol(range(sfs.size()), [vtab = proxy<space>(vtab), sfs = proxy<space>({}, sfs),
                                        svi = proxy<space>(svi)] __device__(int ti) mutable {
                    auto neighborIds = vec3i::uniform(-1);
                    auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
                    for (int i = 0; i != 3; ++i)
                        if (auto no = vtab.query(tri[i]); no >= 0) {
                            neighborIds[i] = svi[no];
                        }
                    sfs.tuple(dim_c<3>, "fp_inds", ti) = neighborIds.reinterpret_bits(float_c);
                });
            }
        }




        void apply() override {
            using namespace zs;

            auto surf = get_input<ZenoParticles>("zssurf");
            // auto bvh_thickness = get_input2<float>("bvh_thickness");
            if(!surf->hasAuxData(ZenoParticles::s_surfTriTag))
                throw std::runtime_error("the input zsparticles has no surface tris");
            if(!surf->hasAuxData(ZenoParticles::s_surfEdgeTag))
                throw std::runtime_error("the input zsparticles has no surface lines");
            if(!surf->hasAuxData(ZenoParticles::s_surfVertTag))
                throw std::runtime_error("the input zsparticles has no surface lines");
            auto& tris  = (*surf)[ZenoParticles::s_surfTriTag];
            auto& lines = (*surf)[ZenoParticles::s_surfEdgeTag];
            auto& points = (*surf)[ZenoParticles::s_surfVertTag];

            if(!tris.hasProperty("inds") || tris.getChannelSize("inds") != 3){
                throw std::runtime_error("the tris has no inds channel");
            }

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

#if 0

            auto bvh_thickness = (T)3 * compute_average_edge_length(cudaExec,verts,"x",tris);

            // std::cout << "bvh_thickness : " << bvh_thickness << std::endl；

            tris.append_channels(cudaExec,{{"ff_inds",3},{"fe_inds",3},{"fp_inds",3}});
            lines.append_channels(cudaExec,{{"fe_inds",2},{"ep_inds",2}});
            if(!compute_ff_neigh_topo(cudaExec,verts,tris,"ff_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_neigh_topo fail");
            if(!compute_fe_neigh_topo(cudaExec,verts,lines,tris,"fe_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_neigh_topo fail");
            if(!compute_fp_neigh_topo(cudaExec,verts,points,tris,"fp_inds",bvh_thickness))
                throw std::runtime_error("ZSInitTopoConnect::compute_face_point_neigh_topo fail");
#else
            compute_surface_neighbors(cudaExec,tris,lines,points);
#endif

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
            const auto& tris    = (*zsparticles)[ZenoParticles::s_surfTriTag];
            const auto& points  = (*zsparticles)[ZenoParticles::s_surfVertTag];
            const auto& verts = zsparticles->getParticles();

            if(!tris.hasProperty("fp_inds") || tris.getChannelSize("fp_inds") != 3) {
                throw std::runtime_error("call ZSInitSurfaceTopology first before VisualizeSurfaceMesh");
            }

            auto nm_points = points.size();
            auto nm_tris = tris.size();

            // transfer the data from gpu to cpu
            constexpr auto cuda_space = execspace_e::cuda;
            auto cudaPol = cuda_exec(); 

            auto surf_verts_buffer = typename ZenoParticles::particles_t({{"x",3}},points.size(),zs::memsrc_e::device,0);
            auto surf_tris_buffer  = typename ZenoParticles::particles_t({{"inds",3}},tris.size(),zs::memsrc_e::device,0);
            // copy the verts' pos data to buffer
            cudaPol(zs::range(points.size()),
                [verts = proxy<cuda_space>({},verts),points = proxy<cuda_space>({},points),surf_verts_buffer = proxy<cuda_space>({},surf_verts_buffer)] ZS_LAMBDA(int pi) mutable {
                    auto v_idx = reinterpret_bits<int>(points("inds",pi));
                    surf_verts_buffer.template tuple<3>("x",pi) = verts.template pack<3>("x",v_idx);
            }); 
            // copy the tris topo to buffer
            TILEVEC_OPS::copy<3>(cudaPol,tris,"fp_inds",surf_tris_buffer,"inds");

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
                [&sverts,surf_verts_buffer = proxy<omp_space>({},surf_verts_buffer)] (int vi) mutable {
                    auto v = surf_verts_buffer.template pack<3>("x",vi);
                    sverts[vi] = zeno::vec3f(v[0],v[1],v[2]);
            });

            ompPol(zs::range(stris.size()),
                [&stris,surf_tris_buffer = proxy<omp_space>({},surf_tris_buffer)] (int ti) mutable {
                    auto t = surf_tris_buffer.template pack<3>("inds",ti).reinterpret_bits(int_c);
                    stris[ti] = zeno::vec3i(t[0],t[1],t[2]);
            });

            set_output("prim",std::move(sprim));
        }
    };

    ZENDEFNODE(VisualizeSurfaceMesh, {{{"ZSParticles"}},
                                {{"prim"}},
                                {},
                                {"ZSGeometry"}});


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



struct VisualizeCollision : zeno::INode {

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
        // if(!zsparticles->hasBvh(ZenoParticles::s_surfTriTag)) {
        //     throw std::runtime_error("the input zsparticles has no surface tris's spacial structure");
        // }
        // if(!zsparticles->hasBvh(ZenoParticles::s_surfEdgeTag)) {
        //     throw std::runtime_error("the input zsparticles has no surface edge's spacial structure");
        // }
        // if(!zsparticles->hasBvh(ZenoParticles::s_surfVertTag))  {
        //     throw std::runtime_error("the input zsparticles has no surface vert's spacial structure");
        // }

        const auto& verts = zsparticles->getParticles();

        auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
        auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

        // auto& stBvh = zsparticles->bvh(ZenoParticles::s_surfTriTag);
        // auto& seBvh = zsparticles->bvh(ZenoParticles::s_surfEdgeTag);

        dtiles_t sttemp(tris.get_allocator(),
            {
                {"nrm",3},
                {"x",3}
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
                {"nrm",3},
                {"x",3}
            },points.size()
        );

        dtiles_t fp_buffer(points.get_allocator(),
            {
                {"inds",4},
                {"area",1},
                {"inverted",1}
            },points.size() * MAX_FP_COLLISION_PAIRS);
        dtiles_t ee_buffer(lines.get_allocator(),
            {
                {"inds",4},
                {"area",1},
                {"inverted",1},
                {"abary",2},
                {"bbary",2}
            },lines.size());

        dtiles_t gh_buffer(points.get_allocator(),
            {
                {"inds",4},
                {"H",12*12},
                {"grad",12}
            },points.size() * MAX_FP_COLLISION_PAIRS + lines.size());


        dtiles_t vtemp(verts.get_allocator(),
            {
                {"x",3},
                {"dir",3},
            },verts.size());


        auto in_collisionEps = get_input2<float>("in_collisionEps");
        auto out_collisionEps = get_input2<float>("out_collisionEps");

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        // calculate facet-point collision pairs and force

        COLLISION_UTILS::do_facet_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
            verts,"x",
            points,
            lines,
            tris,
            sttemp,
            setemp,
            fp_buffer,
            in_collisionEps,out_collisionEps);


        std::vector<zs::PropertyTag> cv_tags{{"xs",3},{"xe",3}};
        auto cv_buffer = typename ZenoParticles::particles_t(cv_tags,points.size() * MAX_FP_COLLISION_PAIRS,zs::memsrc_e::device,0);
        std::vector<zs::PropertyTag> cv_pt_tags{{"p",3},{"t0",3},{"t1",3},{"t2",3}};
        auto cv_pt_buffer = typename ZenoParticles::particles_t(cv_pt_tags,points.size() * MAX_FP_COLLISION_PAIRS,zs::memsrc_e::device,0);


        cudaPol(zs::range(points.size()),
            [fp_buffer = proxy<space>({},fp_buffer),verts = proxy<space>({},verts),
                cv_buffer = proxy<space>({},cv_buffer),
                cv_pt_buffer = proxy<space>({},cv_pt_buffer),
                points = proxy<space>({},points)] ZS_LAMBDA(int pi) mutable {
                    for(int i = 0;i != MAX_FP_COLLISION_PAIRS;++i) {
                        auto inds = fp_buffer.template pack<4>("inds",pi * MAX_FP_COLLISION_PAIRS + i).reinterpret_bits(int_c);
                        bool contact = true;
                        auto pvert = zs::vec<T,3>::zeros();
                        for(int j = 0;j != 4;++j)
                            if(inds[j] < 0)
                                contact = false;
                        if(contact) {
                            pvert = verts.template pack<3>("x",inds[0]);
                            auto t0 = verts.template pack<3>("x",inds[1]);
                            auto t1 = verts.template pack<3>("x",inds[2]);
                            auto t2 = verts.template pack<3>("x",inds[3]);
                            auto center = (t0 + t1 + t2) / (T)3.0;

                            cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = center;

                            cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = t0;
                            cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = t1;
                            cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = t2;                                
                        }else{
                            cv_buffer.template tuple<3>("xs",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            cv_buffer.template tuple<3>("xe",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            
                            cv_pt_buffer.template tuple<3>("p",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            cv_pt_buffer.template tuple<3>("t0",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            cv_pt_buffer.template tuple<3>("t1",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;
                            cv_pt_buffer.template tuple<3>("t2",MAX_FP_COLLISION_PAIRS * pi + i) = pvert;                                
                        }
                    }
        });
        
        cv_buffer = cv_buffer.clone({zs::memsrc_e::host});
        auto collisionFacetVis = std::make_shared<zeno::PrimitiveObject>();
        auto& cv_verts = collisionFacetVis->verts;
        auto& cv_lines = collisionFacetVis->lines;
        cv_verts.resize(points.size() * 2 * MAX_FP_COLLISION_PAIRS);
        cv_lines.resize(points.size() * MAX_FP_COLLISION_PAIRS);

        auto ompPol = omp_exec();  
        constexpr auto omp_space = execspace_e::openmp;

        ompPol(zs::range(cv_buffer.size()),
            [cv_buffer = proxy<omp_space>({},cv_buffer),&cv_verts,&cv_lines] (int pi) mutable {
                auto xs = cv_buffer.template pack<3>("xs",pi);
                auto xe = cv_buffer.template pack<3>("xe",pi);
                cv_verts[pi * 2 + 0] = zeno::vec3f(xs[0],xs[1],xs[2]);
                cv_verts[pi * 2 + 1] = zeno::vec3f(xe[0],xe[1],xe[2]);
                cv_lines[pi] = zeno::vec2i(pi * 2 + 0,pi * 2 + 1);
        });

        set_output("collisionFacetVis",std::move(collisionFacetVis));

        cv_pt_buffer = cv_pt_buffer.clone({zs::memsrc_e::host});
        auto colPointFacetPairVis = std::make_shared<zeno::PrimitiveObject>();
        auto& cv_pt_verts = colPointFacetPairVis->verts;
        auto& cv_pt_tris = colPointFacetPairVis->tris;

        cv_pt_verts.resize(cv_pt_buffer.size() * 4);
        cv_pt_tris.resize(cv_pt_buffer.size());

        ompPol(zs::range(cv_pt_buffer.size()),
            [&cv_pt_verts,&cv_pt_tris,cv_pt_buffer = proxy<omp_space>({},cv_pt_buffer)] (int pi) mutable {
                cv_pt_verts[pi * 4 + 0] = cv_pt_buffer.template pack<3>("p",pi).to_array();
                cv_pt_verts[pi * 4 + 1] = cv_pt_buffer.template pack<3>("t0",pi).to_array();
                cv_pt_verts[pi * 4 + 2] = cv_pt_buffer.template pack<3>("t1",pi).to_array();
                cv_pt_verts[pi * 4 + 3] = cv_pt_buffer.template pack<3>("t2",pi).to_array();

                cv_pt_tris[pi] = zeno::vec3i(pi * 4 + 1,pi * 4 + 2,pi * 4 + 3);
        });


        set_output("colPointFacetPairVis",std::move(colPointFacetPairVis));


        COLLISION_UTILS::evaluate_fp_collision_grad_and_hessian(
            cudaPol,
            verts,"x",
            fp_buffer,
            gh_buffer,0,
            in_collisionEps,out_collisionEps,
            (T)1.0,
            (T)1.0,(T)1.0);

        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"x");
        TILEVEC_OPS::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());

        TILEVEC_OPS::assemble_range(cudaPol,gh_buffer,"grad","inds",vtemp,"dir",0,fp_buffer.size());

        auto scale = get_input2<float>("fp_scale");

        // auto ompPol = omp_exec();  
        // constexpr auto omp_space = execspace_e::openmp;
        
        auto nodalForceVis = std::make_shared<zeno::PrimitiveObject>();
        auto& spverts = nodalForceVis->verts;
        spverts.resize(vtemp.size() * 2);
        auto& splines = nodalForceVis->lines;
        splines.resize(vtemp.size());

        // auto scale = get_input2<float>("scale");

        vtemp = vtemp.clone({zs::memsrc_e::host});
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

        // calculate edge edge collision pairs and face
        COLLISION_UTILS::do_edge_edge_collision_detection(cudaPol,
            verts,"x",
            points,lines,tris,
            sttemp,setemp,
            ee_buffer,
            in_collisionEps,out_collisionEps);        
        std::vector<zs::PropertyTag> cv_ee_tags{{"a0",3},{"a1",3},{"b0",3},{"b1",3},{"abary",2},{"bbary",2}};
        auto cv_ee_buffer = typename ZenoParticles::particles_t(cv_ee_tags,setemp.size(),zs::memsrc_e::device,0);

        cudaPol(zs::range(ee_buffer.size()),
                [ee_buffer = proxy<space>({},ee_buffer),verts = proxy<space>({},verts),
            cv_ee_buffer = proxy<space>({},cv_ee_buffer)] ZS_LAMBDA(int ei) mutable {
                auto inds = ee_buffer.template pack<4>("inds",ei).reinterpret_bits(int_c);
                bool collide = true;
                if(inds[0] < 0 || inds[1] < 0 || inds[2] < 0 || inds[3] < 0)
                    collide = false;
                if(collide) {
                    auto abary = ee_buffer.template pack<2>("abary",ei);
                    auto bbary = ee_buffer.template pack<2>("bbary",ei);

                    // printf("Found edge collision pair %d %d %d %d %f %f %f %f\n",inds[0],inds[1],inds[2],inds[3],
                    //     (float)abary[0],(float)abary[1],(float)bbary[0],(float)bbary[1]);

                    // printf("find collision pairs : %d %d %d %d with bary %f %f %f %f\n",inds[0],inds[1],inds[2],inds[3],
                    //     (float)abary[0],(float)abary[1],(float)bbary[0],(float)bbary[1]);
                    cv_ee_buffer.template tuple<3>("a0",ei) = verts.template pack<3>("x",inds[0]);
                    cv_ee_buffer.template tuple<3>("a1",ei) = verts.template pack<3>("x",inds[1]);
                    cv_ee_buffer.template tuple<3>("b0",ei) = verts.template pack<3>("x",inds[2]);
                    cv_ee_buffer.template tuple<3>("b1",ei) = verts.template pack<3>("x",inds[3]);
                    cv_ee_buffer.template tuple<2>("abary",ei) = abary;
                    cv_ee_buffer.template tuple<2>("bbary",ei) = bbary;
                }else {
                    cv_ee_buffer.template tuple<3>("a0",ei) = zs::vec<T,3>::zeros();
                    cv_ee_buffer.template tuple<3>("a1",ei) = zs::vec<T,3>::zeros();
                    cv_ee_buffer.template tuple<3>("b0",ei) = zs::vec<T,3>::zeros();
                    cv_ee_buffer.template tuple<3>("b1",ei) = zs::vec<T,3>::zeros();
                    cv_ee_buffer.template tuple<2>("abary",ei) = zs::vec<T,2>((T)1.0,0.0);
                    cv_ee_buffer.template tuple<2>("bbary",ei) = zs::vec<T,2>((T)1.0,0.0);
                }
        });

        cv_ee_buffer = cv_ee_buffer.clone({zs::memsrc_e::host});

        // auto ompPol = omp_exec();  
        // constexpr auto omp_space = execspace_e::openmp;

        auto collisionEdgeVis = std::make_shared<zeno::PrimitiveObject>();
        auto& ee_verts = collisionEdgeVis->verts;
        auto& ee_lines = collisionEdgeVis->lines;
        ee_verts.resize(cv_ee_buffer.size() * 2);
        ee_lines.resize(cv_ee_buffer.size());


        ompPol(zs::range(cv_ee_buffer.size()),
            [cv_ee_buffer = proxy<omp_space>({},cv_ee_buffer),&ee_verts,&ee_lines] (int eei) mutable {
                auto a0 = cv_ee_buffer.template pack<3>("a0",eei);
                auto a1 = cv_ee_buffer.template pack<3>("a1",eei);
                auto b0 = cv_ee_buffer.template pack<3>("b0",eei);
                auto b1 = cv_ee_buffer.template pack<3>("b1",eei);     
                
                auto abary = cv_ee_buffer.template pack<2>("abary",eei);
                auto bbary = cv_ee_buffer.template pack<2>("bbary",eei);

                // auto ac = (a0 + a1) / (T)2.0;
                // auto bc = (b0 + b1) / (T)2.0;

                auto ac = abary[0] * a0 + abary[1] * a1;
                auto bc = bbary[0] * b0 + bbary[1] * b1;

                ee_verts[eei * 2 + 0] = zeno::vec3f(ac[0],ac[1],ac[2]);
                ee_verts[eei * 2 + 1] = zeno::vec3f(bc[0],bc[1],bc[2]);
                ee_lines[eei] = zeno::vec2i(eei * 2 + 0,eei * 2 + 1);
        });

        set_output("collisionEdgeVis",std::move(collisionEdgeVis));

        auto colEdgetPairVis = std::make_shared<zeno::PrimitiveObject>();
        auto& cv_ee_verts = colEdgetPairVis->verts;
        auto& cv_ee_lines = colEdgetPairVis->lines;

        cv_ee_verts.resize(cv_ee_buffer.size() * 4);
        cv_ee_lines.resize(cv_ee_buffer.size() * 2);

        ompPol(zs::range(cv_ee_buffer.size()),
            [&cv_ee_verts,&cv_ee_lines,cv_ee_buffer = proxy<omp_space>({},cv_ee_buffer)] (int eei) mutable {
                cv_ee_verts[eei * 4 + 0] = cv_ee_buffer.template pack<3>("a0",eei).to_array();
                cv_ee_verts[eei * 4 + 1] = cv_ee_buffer.template pack<3>("a1",eei).to_array();
                cv_ee_verts[eei * 4 + 2] = cv_ee_buffer.template pack<3>("b0",eei).to_array();
                cv_ee_verts[eei * 4 + 3] = cv_ee_buffer.template pack<3>("b1",eei).to_array();

                cv_ee_lines[eei * 2 + 0] = zeno::vec2i(eei * 4 + 0,eei * 4 + 1);
                cv_ee_lines[eei * 2 + 1] = zeno::vec2i(eei * 4 + 2,eei * 4 + 3);
        });


        set_output("colEdgePairVis",std::move(colEdgetPairVis)); 


        dtiles_t ee_vtemp(verts.get_allocator(),
            {
                {"x",3},
                {"dir",3},
            },verts.size());

        COLLISION_UTILS::evaluate_ee_collision_grad_and_hessian(cudaPol,
            verts,"x",
            ee_buffer,
            gh_buffer,fp_buffer.size(),
            in_collisionEps,out_collisionEps,
            1.0,
            1.0,1.0);

        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",ee_vtemp,"x");
        TILEVEC_OPS::fill(cudaPol,ee_vtemp,"dir",(T)0.0);
        TILEVEC_OPS::assemble_range(cudaPol,gh_buffer,"grad","inds",ee_vtemp,"dir",fp_buffer.size(),ee_buffer.size());

        auto EENodalForceVis = std::make_shared<zeno::PrimitiveObject>();
        auto& ee_spverts = EENodalForceVis->verts;
        ee_spverts.resize(ee_vtemp.size() * 2);
        auto& ee_splines = EENodalForceVis->lines;
        ee_splines.resize(ee_vtemp.size());

        scale = get_input2<float>("ee_scale");

        ee_vtemp = ee_vtemp.clone({zs::memsrc_e::host});   
        ompPol(zs::range(ee_vtemp.size()),
            [ee_vtemp = proxy<space>({},ee_vtemp),&ee_spverts,&ee_splines,scale] (int vi) mutable {
                auto xs = ee_vtemp.template pack<3>("x",vi);
                auto dir = ee_vtemp.template pack<3>("dir",vi);

                auto xe = xs + scale * dir;

                ee_spverts[vi * 2 + 0] = xs.to_array();
                ee_spverts[vi * 2 + 1] = xe.to_array();
                ee_splines[vi] = zeno::vec2i(vi * 2 + 0,vi * 2 + 1);               
        });             

        set_output("EENodalForceVis",std::move(EENodalForceVis));     
    }

};

ZENDEFNODE(VisualizeCollision, {{"ZSParticles",{"float","fp_scale","1.0"},{"float","ee_scale","1.0"},{"float","in_collisionEps"},{"float","out_collisionEps"}},
                                  {
                                        "collisionFacetVis",
                                        "colPointFacetPairVis",
                                        "FPNodalForceVis",
                                        "collisionEdgeVis",
                                        "colEdgePairVis",
                                        "EENodalForceVis",
                                    },
                                  {
                                  },
                                  {"ZSGeometry"}});



}