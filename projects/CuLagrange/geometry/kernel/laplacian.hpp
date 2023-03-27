#pragma once

#include "Structures.hpp"
#include "topology.hpp"

namespace zeno {
    template<typename T>
    constexpr T doublearea(T a,T b,T c) {
        T s = (a + b + c)/2;
        return 2*zs::sqrt(s*(s-a)*(s-b)*(s-c));
    }

    template<typename T>
    constexpr T area(T a,T b,T c) {
        return doublearea(a,b,c)/2;
    }

    template<typename T>
    constexpr T volume(zs::vec<T, 6> l) {
        T u = l(0);
        T v = l(1);
        T w = l(2);
        T U = l(3);
        T V = l(4);
        T W = l(5);
        T X = (w - U + v)*(U + v + w);
        T x = (U - v + w)*(v - w + U);
        T Y = (u - V + w)*(V + w + u);
        T y = (V - w + u)*(w - u + V);
        T Z = (v - W + u)*(W + u + v);
        T z = (W - u + v)*(u - v + W);
        T a = zs::sqrt(x*Y*Z);
        T b = zs::sqrt(y*Z*X);
        T c = zs::sqrt(z*X*Y);
        T d = zs::sqrt(x*y*z);
        T vol = zs::sqrt(
        (-a + b + c + d)*
        ( a - b + c + d)*
        ( a + b - c + d)*
        ( a + b + c - d))/
        (192.*u*v*w);

        return vol;
    }

    template<typename T>
    constexpr void dihedral_angle_intrinsic(const zs::vec<T, 6>& l,const zs::vec<T, 4>& s,zs::vec<T, 6>& theta,zs::vec<T, 6>& cos_theta) {
        zs::vec<T, 6> H_sqr{};
        H_sqr[0] = (1./16.) * (4.*l(3)*l(3)*l(0)*l(0) - zs::sqr((l(1)*l(1) + l(4)*l(4)) - (l(2)*l(2) + l(5)*l(5))));
        H_sqr[1] = (1./16.) * (4.*l(4)*l(4)*l(1)*l(1) - zs::sqr((l(2)*l(2) + l(5)*l(5)) - (l(3)*l(3) + l(0)*l(0))));
        H_sqr[2] = (1./16.) * (4.*l(5)*l(5)*l(2)*l(2) - zs::sqr((l(3)*l(3) + l(0)*l(0)) - (l(4)*l(4) + l(1)*l(1))));
        H_sqr[3] = (1./16.) * (4.*l(0)*l(0)*l(3)*l(3) - zs::sqr((l(4)*l(4) + l(1)*l(1)) - (l(5)*l(5) + l(2)*l(2))));
        H_sqr[4] = (1./16.) * (4.*l(1)*l(1)*l(4)*l(4) - zs::sqr((l(5)*l(5) + l(2)*l(2)) - (l(0)*l(0) + l(3)*l(3))));
        H_sqr[5] = (1./16.) * (4.*l(2)*l(2)*l(5)*l(5) - zs::sqr((l(0)*l(0) + l(3)*l(3)) - (l(1)*l(1) + l(4)*l(4))));

        cos_theta(0) = (H_sqr(0) - s(1)*s(1) - s(2)*s(2)) / (-2.*s(1) * s(2));
        cos_theta(1) = (H_sqr(1) - s(2)*s(2) - s(0)*s(0)) / (-2.*s(2) * s(0));
        cos_theta(2) = (H_sqr(2) - s(0)*s(0) - s(1)*s(1)) / (-2.*s(0) * s(1));
        cos_theta(3) = (H_sqr(3) - s(3)*s(3) - s(0)*s(0)) / (-2.*s(3) * s(0));
        cos_theta(4) = (H_sqr(4) - s(3)*s(3) - s(1)*s(1)) / (-2.*s(3) * s(1));
        cos_theta(5) = (H_sqr(5) - s(3)*s(3) - s(2)*s(2)) / (-2.*s(3) * s(2));

        //TODO the theta here might be invalid, might be a hidden bug
        theta(0) = zs::acos(cos_theta(0));  
        theta(1) = zs::acos(cos_theta(1)); 
        theta(2) = zs::acos(cos_theta(2)); 
        theta(3) = zs::acos(cos_theta(3)); 
        theta(4) = zs::acos(cos_theta(4)); 
        theta(5) = zs::acos(cos_theta(5));       
    }


    template<int MAX_NEIGHS,
        typename Pol,
        typename PosTileVec,
        typename SrcTileVec,
        typename HalfEdgeTileVec,
        typename PointTileVec,
        typename EdgeTileVec,
        typename TriTileVec,
        typename DstTileVec>
    void compute_smooth_laplacian(Pol& pol,
        const PosTileVec& verts,const zs::SmallString& xTag,
        const SrcTileVec& src,const zs::SmallString& srcTag,
        const HalfEdgeTileVec& halfEdges,
        const PointTileVec& points,
        const EdgeTileVec& edges,
        const TriTileVec& tris,
        DstTileVec& dst,const zs::SmallString& dstTag) {
            using T = typename SrcTileVec::value_type;
            using namespace zs;
            constexpr auto space = Pol::exec_tag::value;
            int space_dim = src.getPropertySize(srcTag);

            pol(range(points.size()),[
                    verts = proxy<space>({},verts),xTag,
                    src = proxy<space>({},src),srcTag,
                    half_edges = proxy<space>({},halfEdges),
                    points = proxy<space>({},points),
                    edges = proxy<space>({},edges),
                    tris = proxy<space>({},tris),
                    dst = proxy<space>({},dst),dstTag,space_dim]
                        ZS_LAMBDA(int pi) mutable {
                auto vidx = reinterpret_bits<int>(points("inds",pi));
                auto he_idx = reinterpret_bits<int>(points("he_inds",pi));
                zs::vec<int,MAX_NEIGHS> pneighs = get_one_ring_neigh_points<MAX_NEIGHS>(he_idx,half_edges);
                zs::vec<int,MAX_NEIGHS> eneighs = get_one_ring_neigh_edges<MAX_NEIGHS>(he_idx,half_edges);
                T ws = (T)0.0;
                for(int i = 0;i != MAX_NEIGHS;++i) {
                    auto npi = pneighs[i];
                    if(npi < 0)
                        break;
                    auto nvidx = reinterpret_bits<int>(points("inds",npi));
                    auto w = (T)0.0;
                    // compute cotangent weight
                    {
                        auto li = eneighs[i];
                        auto ne = edges.pack(dim_c<2>,"inds",li).reinterpret_bits(int_c);
                        auto fe_inds = edges.pack(dim_c<2>,"fe_inds",li).reinterpret_bits(int_c);

                        auto t0 = fe_inds[0];
                        auto t1 = fe_inds[1];

                        zs::vec<T,3> l{};
                        zs::vec<T,3> l2{};
                        zs::vec<T,3> vs[3] = {};

                        for(int j = 0;j != 2;++j) {
                            if(fe_inds[j] < 0)
                                break;
                            auto tri = tris.pack(dim_c<3>,"inds",fe_inds[j]).reinterpret_bits(int_c);
                            int k = 0;
                            for(k = 0;k != 3;++k) {
                                if((tri[k] == ne[0] && tri[(k+1)%3] == ne[1]) || (tri[k] == ne[1] && tri[(k+1)%3] == ne[0]))
                                    break;
                            }
                            if(k == 3) {
                                printf("invalid fe_inds detected");
                            }else{
                                for(int d = 0;d != 3;++d)
                                    vs[d] = verts.pack(dim_c<3>,xTag,tri[(k + d) % 3]);
                                for(int d = 0;d != 3;++d){
                                    l2[d] = (vs[d] - vs[(d+1) % 3]).l2Norm();
                                    l[d] = zs::sqrt(l2[d]);
                                }

                                auto dblA = doublearea(l[0],l[1],l[2]);
                                auto C = (l2[2] + l2[1] - l2[0])/dblA/(T)4.0;
                                w += C;
                            }
                            
                        }
                    }
                    ws += w;
                    for(int i = 0;i != space_dim;++i)
                        dst(dstTag,i,pi) += src(srcTag,i,pi) * w;
                }
                for(int i = 0;i != space_dim;++i)
                    dst(dstTag,i,pi) /= ws;
            });            
    }

    template<int MAX_NEIGHS,
        typename Pol,
        typename SrcTileVec,
        typename HalfEdgeTileVec,
        typename PointTileVec,
        typename EdgeTileVec,
        typename TriTileVec,
        typename DstTileVec>
    void compute_smooth(Pol& pol,
        const SrcTileVec& src,const zs::SmallString& srcTag,
        const HalfEdgeTileVec& halfEdges,
        const PointTileVec& points,
        const EdgeTileVec& edges,
        const TriTileVec& tris,
        DstTileVec& dst,const zs::SmallString& dstTag) {
            using T = typename SrcTileVec::value_type;
            using namespace zs;
            constexpr auto space = Pol::exec_tag::value;
            int space_dim = src.getPropertySize(srcTag);
            pol(range(points.size()),[
                    src = proxy<space>({},src),srcTag,
                    half_edges = proxy<space>({},halfEdges),
                    points = proxy<space>({},points),
                    edges = proxy<space>({},edges),
                    tris = proxy<space>({},tris),
                    dst = proxy<space>({},dst),dstTag,space_dim]
                        ZS_LAMBDA(int pi) mutable {
                auto vidx = reinterpret_bits<int>(points("inds",pi));
                auto he_idx = reinterpret_bits<int>(points("he_inds",pi));
                zs::vec<int,MAX_NEIGHS> pneighs = get_one_ring_neigh_points<MAX_NEIGHS>(he_idx,half_edges);
                T ws = (T)0.0;

                for(int i = 0;i != MAX_NEIGHS;++i) {
                    auto npi = pneighs[i];
                    if(npi < 0)
                        break;
                    auto nvidx = reinterpret_bits<int>(points("inds",npi));
                    auto w = (T)1.0;
                    ws += w;
                    for(int d = 0;d != space_dim;++d)
                        dst(dstTag,d,pi) += w * src(srcTag,d,pi);
                }
                for(int d = 0;d != space_dim;++d)
                    dst(dstTag,d,pi) /= ws;
            });
    }

    template<int MAX_NEIGHS,
        typename Pol,
        typename SrcTileVec,
        typename HalfEdgeTileVec,
        typename PointTileVec,
        typename EdgeTileVec,
        typename TriTileVec,
        typename DstTileVec>
    void compute_smooth_corrective(Pol& pol,
            const SrcTileVec& src,const zs::SmallString& srcTag,
            const HalfEdgeTileVec& halfEdges,
            const PointTileVec& points,
            const EdgeTileVec& edges,
            const TriTileVec& tris,
            DstTileVec& dst,const zs::SmallString& dstTag) {

    }


    template <int simplex_size,typename Pol,typename ETileVec,typename VTileVec,typename ETmpTileVec>
    void compute_cotmatrix(Pol &pol,const ETileVec &eles,
        const VTileVec &verts, const zs::SmallString& xTag, 
        ETmpTileVec& etemp, const zs::SmallString& HTag) {

        static_assert(zs::is_same_v<typename ETileVec::value_type,typename VTileVec::value_type>,"precision not match");
        static_assert(zs::is_same_v<typename ETileVec::value_type,typename ETmpTileVec::value_type>,"precision not match");   

        using T = typename VTileVec::value_type;

        using namespace zs;
        static_assert(simplex_size >= 3 && simplex_size <=4, "invalid co-dimension!\n");
        constexpr auto space = Pol::exec_tag::value;

        #if ZS_ENABLE_CUDA && defined(__CUDACC__)
            static_assert(space == execspace_e::cuda,
                    "specified policy and compiler not match");
        #else
            static_assert(space != execspace_e::cuda,
                    "specified policy and compiler not match");
        #endif

        if(!verts.hasProperty(xTag)){
            printf("the verts buffer does not contain specified channel\n");
        }   

        // if(!etemp.hasProperty(HTag)){
        //     printf("the etemp buffer does not contain specified channel\n");
        // }  

        etemp.append_channels(pol,{{HTag,simplex_size*simplex_size}});

        // zs::Vector<T> C{eles.get_allocator(),eles.size()*simplex_size*(simplex_size-1)/2};

        // compute cotangent entries
        // fmt::print("COMPUTE COTANGENT ENTRIES\n");
        int nm_elms = etemp.size();
        pol(zs::range(etemp.size()),
            [eles = proxy<space>({},eles),verts = proxy<space>({},verts),
            etemp = proxy<space>({},etemp),xTag,HTag,nm_elms] ZS_LAMBDA(int ei) mutable {
                constexpr int ne = simplex_size*(simplex_size-1)/2;
                auto inds = eles.template pack<simplex_size>("inds",ei).template reinterpret_bits<int>();
                
                using IV = zs::vec<int,ne*2>;
                using TV = zs::vec<T, ne>;

                TV C;
                IV edges;
                // printf("check_0\n");
                // compute the cotangent entris
                if constexpr (simplex_size == 3){
                    edges = IV{1,2,2,0,0,1};
                    zs::vec<T,3> l;
                    zs::vec<T,3> l2;
                    for(size_t i = 0;i != ne;++i) {
                        l[i] = (verts.pack<3>(xTag,inds[edges[i*2+0]]) - verts.pack<3>(xTag,inds[edges[i*2+1]])).norm();
                        l2[i] = l[i] * l[i];
                    }
                    auto dblA = doublearea(l[0],l[1],l[2]);// check here, double area
                    for(size_t i = 0;i != ne;++i)
                        C[i] = (l2[edges[2*i+0]] + l2[edges[2*i+1]] - l2[3 - edges[2*i+0] - edges[2*i+1]])/dblA/4.0;
                }
                if constexpr (simplex_size == 4){
                    // printf("check_1\n");
                    edges = IV{1,2,2,0,0,1,3,0,3,1,3,2};
                    zs::vec<T,ne> l{};
                    l[0] = (verts.pack<3>(xTag,inds[3]) - verts.pack<3>(xTag,inds[0])).length();
                    l[1] = (verts.pack<3>(xTag,inds[3]) - verts.pack<3>(xTag,inds[1])).length();
                    l[2] = (verts.pack<3>(xTag,inds[3]) - verts.pack<3>(xTag,inds[2])).length();
                    l[3] = (verts.pack<3>(xTag,inds[1]) - verts.pack<3>(xTag,inds[2])).length();
                    l[4] = (verts.pack<3>(xTag,inds[2]) - verts.pack<3>(xTag,inds[0])).length();
                    l[5] = (verts.pack<3>(xTag,inds[0]) - verts.pack<3>(xTag,inds[1])).length();
                    // for(int i = 0;i != ne;++i)
                    //     l[i] = (verts.pack<3>(xTag,inds[edges[i*2+0]]) - verts.pack<3>(xTag,inds[edges[i*2+1]])).norm();
                    // printf("check_2\n");
                    zs::vec<T, 4> s{ 
                        area(l[1],l[2],l[3]),
                        area(l[0],l[2],l[4]),
                        area(l[0],l[1],l[5]),
                        area(l[3],l[4],l[5])};
                    // printf("check_3\n");
                    zs::vec<T,ne> cos_theta{},theta{};
                    dihedral_angle_intrinsic(l,s,theta,cos_theta);
                    // printf("check_4\n");
                    T vol = eles("vol",ei);
                    // T vol_cmp = volume(l);
                    // if(fabs(vol_cmp - vol) > 1e-6)
                        // printf("VOL_ERROR<%d> : %f\n",ei,(float)fabs(vol_cmp - vol));
                    zs::vec<T, 6> sin_theta{};
                    #if 0
                    sin_theta(0) = vol / ((2./(3.*l(0))) * s(1) * s(2));
                    sin_theta(1) = vol / ((2./(3.*l(1))) * s(2) * s(0));
                    sin_theta(2) = vol / ((2./(3.*l(2))) * s(0) * s(1));
                    sin_theta(3) = vol / ((2./(3.*l(3))) * s(3) * s(0));
                    sin_theta(4) = vol / ((2./(3.*l(4))) * s(3) * s(1));
                    sin_theta(5) = vol / ((2./(3.*l(5))) * s(3) * s(2));
                    #else
                    for(size_t i = 0;i !=ne; ++i)
                        sin_theta(i) = zs::sin(theta(i));
                    #endif
                    C = (1./6.) * l * cos_theta / sin_theta;
                }

                constexpr int simplex_size2 = simplex_size*simplex_size;
                etemp.template tuple<simplex_size2>(HTag,ei) = zs::vec<T,simplex_size2>::zeros();


                for(size_t i = 0;i != ne;++i){
                    int source = edges(i*2 + 0);
                    int dest = edges(i*2 + 1);
                    etemp(HTag,simplex_size*source + dest,ei) -= C(i); 
                    etemp(HTag,simplex_size*dest + source,ei) -= C(i); 
                    etemp(HTag,simplex_size*source + source,ei) += C(i); 
                    etemp(HTag,simplex_size*dest + dest,ei) += C(i); 
                }

                auto L = etemp.template pack<simplex_size,simplex_size>(HTag,ei);
        });

        // fmt::print("FINISH COMPUTING COTANGENT ENTRIES\n");

    }
};