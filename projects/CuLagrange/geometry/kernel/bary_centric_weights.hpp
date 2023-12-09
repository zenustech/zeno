#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"

#include <iostream>

#include "geo_math.hpp"
#include "tiled_vector_ops.hpp"
#include "..\fem\Ccds.hpp"

namespace zeno {

    template <typename TileVecT, int codim = 3>
    zs::Vector<zs::AABBBox<3, typename TileVecT::value_type>>
    get_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT &vtemp,
                            const zs::SmallString &xTag,
                            const typename ZenoParticles::particles_t &eles,
                            zs::wrapv<codim>, int voffset) {
        using namespace zs;
        using T = typename TileVecT::value_type;
        using bv_t = AABBBox<3, T>;
        static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
        constexpr auto space = execspace_e::cuda;
        zs::Vector<bv_t> ret{eles.get_allocator(), eles.size()};
        pol(range(eles.size()), [eles = proxy<space>({}, eles),
                                bvs = proxy<space>(ret),
                                vtemp = proxy<space>({}, vtemp),
                                codim_v = wrapv<codim>{}, xTag,
                                voffset] ZS_LAMBDA(int ei) mutable {
            constexpr int dim = RM_CVREF_T(codim_v)::value;
            auto inds =
                eles.template pack<dim>("inds", ei).template reinterpret_bits<int>() +
                voffset;
            auto x0 = vtemp.template pack<3>(xTag, inds[0]);
            bv_t bv{x0, x0};
            for (int d = 1; d != dim; ++d)
            merge(bv, vtemp.template pack<3>(xTag, inds[d]));
            bvs[ei] = bv;
        });
        return ret;
    }

    template<typename T>
    constexpr T volume(
        const zs::vec<T,3>& p0,
        const zs::vec<T,3>& p1,
        const zs::vec<T,3>& p2,
        const zs::vec<T,3>& p3) {
        zs::vec<T,4,4> m{};
        for(int i = 0;i < 3;++i){
            m(i,0) = p0[i];
            m(i,1) = p1[i];
            m(i,2) = p2[i];
            m(i,3) = p3[i];
        }
        m(3,0) = m(3,1) = m(3,2) = m(3,3) = 1;
        return (T)zs::determinant(m.template cast<double>())/6;
    }

    template<typename T>
    constexpr T area(
        const zs::vec<T,3>& p0,
        const zs::vec<T,3>& p1,
        const zs::vec<T,3>& p2
    ) {
        auto p01 = p0 - p1;
        auto p02 = p0 - p2;
        auto p12 = p1 - p2;
        T a = p01.length();
        T b = p02.length();
        T c = p12.length();
        T s = (a + b + c)/2;
        T A2 = s*(s-a)*(s-b)*(s-c);
        if(A2 > zs::limits<T>::epsilon()*10)
            return zs::sqrt(A2);
        else
            return 0;
    }    


    template<typename T>
    constexpr zs::vec<T,3> compute_vertex_triangle_barycentric_weights(const zs::vec<T,3>& p,
        const zs::vec<T,3>& t0,
        const zs::vec<T,3>& t1,
        const zs::vec<T,3>& t2) {
            constexpr auto eps = 1e-6;
            const auto& v1 = t0;
            const auto& v2 = t1;
            const auto& v3 = t2;
            const auto& v4 = p;

            auto x13 = v1 - v3;
            auto x23 = v2 - v3;
            auto x43 = v4 - v3;
            auto A00 = x13.dot(x13);
            auto A01 = x13.dot(x23);
            auto A11 = x23.dot(x23);
            auto b0 = x13.dot(x43);
            auto b1 = x23.dot(x43);
            auto detA = A00 * A11 - A01 * A01;

            zs::vec<T,3> bary{};

            bary[0] = ( A11 * b0 - A01 * b1) / detA;
            bary[1] = (-A01 * b0 + A00 * b1) / detA;
            bary[2] = 1 - bary[0] - bary[1];

            return bary;
    }

    template<typename T>
    constexpr zs::vec<T,4> compute_vertex_tetrahedron_barycentric_weights(const zs::vec<T,3>& p,
        const zs::vec<T,3>& p0,
        const zs::vec<T,3>& p1,
        const zs::vec<T,3>& p2,
        const zs::vec<T,3>& p3
    ) {
        #if 1
        auto vol = volume(p0,p1,p2,p3);
        auto vol0 = volume(p,p1,p2,p3);
        auto vol1 = volume(p0,p,p2,p3);      
        auto vol2 = volume(p0,p1,p,p3);
        auto vol3 = volume(p0,p1,p2,p);
        #else
        auto vol = LSL_GEO::volume<T>(p0,p1,p2,p3);
        auto vol0 = LSL_GEO::volume<T>(p,p1,p2,p3);
        auto vol1 = LSL_GEO::volume<T>(p0,p,p2,p3);      
        auto vol2 = LSL_GEO::volume<T>(p0,p1,p,p3);
        auto vol3 = LSL_GEO::volume<T>(p0,p1,p2,p);
        #endif
        return zs::vec<T,4>{vol0/vol,vol1/vol,vol2/vol,vol3/vol};
    }

    template<typename T>
    constexpr bool compute_vertex_prism_barycentric_weights(const zs::vec<T,3>& p,
        const zs::vec<T,3>& a0,
        const zs::vec<T,3>& a1,
        const zs::vec<T,3>& a2,
        const zs::vec<T,3>& b0,
        const zs::vec<T,3>& b1,
        const zs::vec<T,3>& b2,
        T& toc,
        zs::vec<T,6>& bary,
        const T& eta = (T)0.1) {

        auto v0 = b0 - a0;
        auto v1 = b1 - a1;
        auto v2 = b2 - a2;

        toc = (T)1.0;
        auto is_intersected = accd::ptccd(p,a0,a1,a2,zs::vec<T,3>::zeros(),v0,v1,v2,(T)eta,(T)0,toc);
        if(!is_intersected)
            return is_intersected;

        auto c0 = a0 + toc * v0;
        auto c1 = a1 + toc * v1;
        auto c2 = a2 + toc * v2;

        auto intersected_bary = compute_vertex_triangle_barycentric_weights(p,c0,c1,c2);
        
        for(int i = 0;i != 3;++i) {
            bary[i] = (1 - toc) * intersected_bary[i];
            bary[i + 3] = toc * intersected_bary[i];
        }

        // printf("find a vertex enclose by prism with tri_bary :  %f %f %f and toc : %f\n",
        //     (float)intersected_bary[0],(float)intersected_bary[1],(float)intersected_bary[2],(float)toc);

        return is_intersected;
    }

    template <typename Pol,typename VTileVec,typename ETileVec,typename EmbedTileVec,typename BCWTileVec>
    constexpr void compute_barycentric_weights(Pol& pol,const VTileVec& verts,
        const ETileVec& quads,const EmbedTileVec& everts,
        const zs::SmallString& x_tag,BCWTileVec& bcw,
        const zs::SmallString& elm_tag,const zs::SmallString& weight_tag,
        float bvh_thickness,
        int fitting_in) {

        // std::cout << "COMPUTE BARYCENTRIC_WEIGHTS BEGIN" << std::endl;

        static_assert(zs::is_same_v<typename BCWTileVec::value_type,typename EmbedTileVec::value_type>,"precision not match");
        static_assert(zs::is_same_v<typename VTileVec::value_type,typename ETileVec::value_type>,"precision not match");        
        static_assert(zs::is_same_v<typename VTileVec::value_type,typename BCWTileVec::value_type>,"precision not match"); 
        using T = typename VTileVec::value_type; 
        using bv_t = zs::AABBBox<3, T>;
        
        using namespace zs;

        // auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // std::cout << "TRY RETRIEVE BOUNDING VOLUMES" << std::endl;

        // std::cout << "QUADS : " << quads.getPropertySize(elm_tag) << "\t" << quads.size() << std::endl;
        const int mem = (int)quads.memspace();
        const int did = quads.devid();


        auto bvs = retrieve_bounding_volumes(pol,verts,quads,wrapv<4>{},bvh_thickness,x_tag);


        auto tetsBvh = LBvh<3, int,T>{};


        tetsBvh.build(pol,bvs);
        // std::cout << "FINISH BUILDING TETS BVG" << std::endl;

        int numEmbedVerts = everts.size();
        pol(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw),elm_tag] ZS_LAMBDA(int vi) mutable {
                bcw(elm_tag,vi) = reinterpret_bits<T>(int(-1));
            });

        pol(zs::range(numEmbedVerts),
            [verts = proxy<space>({},verts),eles = proxy<space>({},quads),bcw = proxy<space>({},bcw),
                    everts = proxy<space>({},everts),tetsBvh = proxy<space>(tetsBvh),
                    x_tag,elm_tag,weight_tag,fitting_in] ZS_LAMBDA (int vi) mutable {
                const auto& p = everts.template pack<3>(x_tag,vi);
                T closest_dist = 1e6;
                bool found = false;
                // if(vi == 10820)
                    // printf("check to locate vert %d using bvh with pos = %f %f %f\n",vi,(float)p[0],(float)p[1],(float)p[2]);

                // auto dst_bv = bv_t{get_bounding_box(dst )}
                tetsBvh.iter_neighbors(p,[&](int ei){
                    // printf("test %d v's neighbor element %d ei\n",vi,ei);
                    if(found)
                        return;
                    // if(vi == 10820)
                    auto inds = eles.template pack<4>(elm_tag, ei).template reinterpret_bits<int>();
                    auto p0 = verts.template pack<3>(x_tag,inds[0]);
                    auto p1 = verts.template pack<3>(x_tag,inds[1]);
                    auto p2 = verts.template pack<3>(x_tag,inds[2]);
                    auto p3 = verts.template pack<3>(x_tag,inds[3]);

                    auto ws = compute_vertex_tetrahedron_barycentric_weights(p,p0,p1,p2,p3);

                    T epsilon = zs::limits<T>::epsilon();
                    if(ws[0] > epsilon && ws[1] > epsilon && ws[2] > epsilon && ws[3] > epsilon){
                        bcw(elm_tag,vi) = reinterpret_bits<T>(ei);
                        bcw.template tuple<4>(weight_tag,vi) = ws;
                        found = true;
                        return;
                    }
                    if(!fitting_in)
                        return;
                    zs::vec<T,3> bary{};

                    if(ws[0] < 0){
                        // T dist = compute_dist_2_facet(p,p1,p2,p3);
                        T dist = LSL_GEO::get_vertex_triangle_distance(p1,p2,p3,p,bary);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<T>(ei);
                            bcw.template tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                    if(ws[1] < 0){
                        T dist = LSL_GEO::get_vertex_triangle_distance(p0,p2,p3,p,bary);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<T>(ei);
                            bcw.template tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                    if(ws[2] < 0){
                        T dist = LSL_GEO::get_vertex_triangle_distance(p0,p1,p3,p,bary);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<T>(ei);
                            bcw.template tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                    if(ws[3] < 0){
                        T dist = LSL_GEO::get_vertex_triangle_distance(p0,p1,p2,p,bary);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<T>(ei);
                            bcw.template tuple<4>(weight_tag,vi) = ws;
                        }
                    }

                    if(!fitting_in){
                        printf("bind vert %d to %d under non-fitting-in mode\n",vi,ei);
                        // return;
                    }


                });// finish iter the neighbor tets
        });
    }

    template <typename Pol,
        typename VTileVec,
        typename VNrmTileVec,
        typename BoundaryTileVec,
        typename HalfEdgeTileVec,
        typename TriTileVec,
        typename CellTileVec,
        typename T = VTileVec::value_type>
    constexpr void compute_boundary_edge_cells_and_vertex_normal(Pol& pol,const VTileVec& verts,const zs::SmallString& xtag,
        VNrmTileVec& vertex_nrm_buffer,
        const BoundaryTileVec& boundary_halfedges,
        const HalfEdgeTileVec& halfedges,
        const TriTileVec& tris,
        CellTileVec& boundary_cell_buffer,
        const T& thickness,
        const T& extend_thickness) {
            using namespace zs;
            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            constexpr auto exec_tag = wrapv<space>{};

            TILEVEC_OPS::fill(pol,vertex_nrm_buffer,"nrm",(T)0.0);
            pol(zs::range(tris.size()),[
                exec_tag = exec_tag,
                tris = tris.begin("inds",dim_c<3>,int_c),
                verts = verts.begin(xtag,dim_c<3>),
                xtag = zs::SmallString(xtag),
                nrmOffset = vertex_nrm_buffer.getPropertyOffset("nrm"),
                vertex_nrm_buffer = view<space>(vertex_nrm_buffer)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris[ti];
                    auto nrm = LSL_GEO::facet_normal(verts[tri[0]],verts[tri[1]],verts[tri[2]]);
                    auto w = LSL_GEO::area(verts[tri[0]],verts[tri[1]],verts[tri[2]]);
                    for(int i = 0;i != 3;++i)
                        for(int d = 0;d != 3;++d)
                            atomic_add(exec_tag,&vertex_nrm_buffer(nrmOffset + d,tri[i]),w * nrm[d]);
            });
            TILEVEC_OPS::normalized_channel<3>(pol,vertex_nrm_buffer,"nrm");

            TILEVEC_OPS::fill(pol,vertex_nrm_buffer,"enrm",(T)0.0);
            pol(zs::range(boundary_halfedges.size()),[
                exec_tag = exec_tag,
                boundary_halfedges = boundary_halfedges.begin("he_inds",dim_c<1>,int_c),
                verts = proxy<space>({},verts),
                enrmOffset = vertex_nrm_buffer.getPropertyOffset("enrm"),
                vertex_nrm_buffer = proxy<space>({},vertex_nrm_buffer),
                xtag = zs::SmallString(xtag),
                tris = tris.begin("inds",dim_c<3>,int_c),
                halfedges = proxy<space>({},halfedges)] ZS_LAMBDA(int bei) mutable {
                    auto hi = boundary_halfedges[bei];
                    auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));

                    auto tri = tris[ti];
                    auto edge = zs::vec<int,2>{tri[local_vertex_id],tri[(local_vertex_id + 1) % 3]};

                    auto E = verts.pack(dim_c<3>,xtag,edge[1]) - verts.pack(dim_c<3>,xtag,edge[0]);
                    zs::vec<T,3> denrm[2] = {};
                    for(int i = 0;i != 2;++i)
                        denrm[i] = E.cross(vertex_nrm_buffer.pack(dim_c<3>,"nrm",edge[i]));
                    
                    for(int i = 0;i != 2;++i)
                        for(int d = 0;d != 3;++d)
                            atomic_add(exec_tag,&vertex_nrm_buffer(enrmOffset + d,edge[i]),denrm[i][d]);
            });
            TILEVEC_OPS::normalized_channel<3>(pol,vertex_nrm_buffer,"enrm");

            pol(zs::range(boundary_halfedges.size()),[
                boundary_halfedges = boundary_halfedges.begin("he_inds",dim_c<1>,int_c),
                boundary_cell_buffer = proxy<space>({},boundary_cell_buffer),
                thickness = thickness,
                extend_thickness = extend_thickness,
                vertex_nrm_buffer = proxy<space>({},vertex_nrm_buffer),
                xtag = zs::SmallString(xtag),
                verts = proxy<space>({},verts),
                tris = tris.begin("inds",dim_c<3>,int_c),
                halfedges = proxy<space>({},halfedges)] ZS_LAMBDA(int cell_id) mutable {
                    auto hi = boundary_halfedges[cell_id];
                    auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    auto local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));

                    auto tri = tris[ti];
                    auto edge = zs::vec<int,2>{tri[local_vertex_id],tri[(local_vertex_id + 1) % 3]};

                    zs::vec<T,3> snrm[2] = {};
                    zs::vec<T,3> senrm[2] = {};
                    zs::vec<T,3> epos[2] = {};

                    for(int i = 0;i != 2;++i) {
                        epos[i] = verts.pack(dim_c<3>,xtag,edge[i]);
                        snrm[i] = vertex_nrm_buffer.pack(dim_c<3>,"nrm",edge[i]);
                        senrm[i] = vertex_nrm_buffer.pack(dim_c<3>,"enrm",edge[i]);
                    }

                    auto E = (epos[1] - epos[0]).normalized();

                    zs::vec<T,3> cell_vertices[8] = {};

                    cell_vertices[0] = epos[1] - thickness * snrm[1] + 0.01 * thickness * E;
                    cell_vertices[1] = epos[1] + thickness * snrm[1] + 0.01 * thickness * E;
                    cell_vertices[2] = epos[0] - thickness * snrm[0] - 0.01 * thickness * E;
                    cell_vertices[3] = epos[0] + thickness * snrm[0] - 0.01 * thickness * E;


                    cell_vertices[0 + 4] = cell_vertices[0] + senrm[1] * extend_thickness;
                    cell_vertices[1 + 4] = cell_vertices[1] + senrm[1] * extend_thickness;
                    cell_vertices[2 + 4] = cell_vertices[2] + senrm[0] * extend_thickness;
                    cell_vertices[3 + 4] = cell_vertices[3] + senrm[0] * extend_thickness;

                    for(int i = 0;i != 8;++i)
                        boundary_cell_buffer.tuple(dim_c<3>,"x",cell_id * 8 + i) = cell_vertices[i];
            });

    }


};




