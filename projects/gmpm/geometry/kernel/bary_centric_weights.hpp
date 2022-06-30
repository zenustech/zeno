#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"

namespace zeno {
    template<typename T>
    constexpr T compute_dist_2_facet(const zs::vec<T,3>& vp,const zs::vec<T,3>& v0,const zs::vec<T,3>& v1,const zs::vec<T,3>& v2){
        auto v012 = (v0 + v1 + v2) / 3;
        auto v01 = (v0 + v1) / 2;
        auto v02 = (v0 + v2) / 2;
        auto v12 = (v1 + v2) / 2;

        T dist = 1e6;
        T tdist = (v012 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v01 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v02 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v12 - vp).norm();
        dist = tdist < dist ? tdist : dist;

        tdist = (v0 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v1 - vp).norm();
        dist = tdist < dist ? tdist : dist;
        tdist = (v2 - vp).norm();
        dist = tdist < dist ? tdist : dist;

        return dist;        
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
    constexpr zs::vec<T,4> compute_barycentric_weights(const zs::vec<T,3>& p,
        const zs::vec<T,3>& p0,
        const zs::vec<T,3>& p1,
        const zs::vec<T,3>& p2,
        const zs::vec<T,3>& p3
    ) {
        auto vol = volume(p0,p1,p2,p3);
        auto vol0 = volume(p,p1,p2,p3);
        auto vol1 = volume(p0,p,p2,p3);      
        auto vol2 = volume(p0,p1,p,p3);
        auto vol3 = volume(p0,p1,p2,p);
        return zs::vec<T,4>{vol0/vol,vol1/vol,vol2/vol,vol3/vol};
    }

    template <typename T,typename Pol>
    constexpr void compute_barycentric_weights(Pol& pol,const typename ZenoParticles::particles_t& verts,
        const typename ZenoParticles::particles_t& quads,const typename ZenoParticles::particles_t& everts,
        const zs::SmallString& x_tag, typename ZenoParticles::particles_t& bcw,
        const zs::SmallString& elm_tag,const zs::SmallString& weight_tag,int fitting_in) {

        using namespace zs;

        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;


        T bvh_thickness = 0;
        auto bvs = retrieve_bounding_volumes(cudaExec,verts,quads,wrapv<4>{},bvh_thickness,x_tag);
        auto tetsBvh = LBvh<3,32, int,T>{};
        tetsBvh.build(cudaExec,bvs);

        int numEmbedVerts = everts.size();
        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw),elm_tag] ZS_LAMBDA(int vi) mutable {
                bcw(elm_tag,vi) = reinterpret_bits<T>(int(-1));
            });

        cudaExec(zs::range(numEmbedVerts),
            [verts = proxy<space>({},verts),eles = proxy<space>({},quads),bcw = proxy<space>({},bcw),
                    everts = proxy<space>({},everts),tetsBvh = proxy<space>(tetsBvh),x_tag,elm_tag,weight_tag,fitting_in] ZS_LAMBDA (int vi) mutable {
                const auto& p = everts.pack<3>(x_tag,vi);
                T closest_dist = 1e6;
                bool found = false;
                tetsBvh.iter_neighbors(p,[&](int ei){
                    if(found)
                        return;
                    auto inds = eles.pack<4>(elm_tag, ei).reinterpret_bits<int>();
                    auto p0 = verts.pack<3>(x_tag,inds[0]);
                    auto p1 = verts.pack<3>(x_tag,inds[1]);
                    auto p2 = verts.pack<3>(x_tag,inds[2]);
                    auto p3 = verts.pack<3>(x_tag,inds[3]);

                    auto ws = compute_barycentric_weights(p,p0,p1,p2,p3);

                    T epsilon = zs::limits<T>::epsilon();
                    if(ws[0] > epsilon && ws[1] > epsilon && ws[2] > epsilon && ws[3] > epsilon){
                        bcw(elm_tag,vi) = reinterpret_bits<float>(ei);
                        bcw.tuple<4>(weight_tag,vi) = ws;
                        found = true;
                        return;
                    }
                    if(!fitting_in)
                        return;

                    if(ws[0] < 0){
                        T dist = compute_dist_2_facet(p,p1,p2,p3);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                    if(ws[1] < 0){
                        T dist = compute_dist_2_facet(p,p0,p2,p3);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                    if(ws[2] < 0){
                        T dist = compute_dist_2_facet(p,p0,p1,p3);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                    if(ws[3] < 0){
                        T dist = compute_dist_2_facet(p,p0,p1,p2);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw(elm_tag,vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>(weight_tag,vi) = ws;
                        }
                    }
                });// finish iter the neighbor tets
        });
    }

};




