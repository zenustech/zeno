#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
// #include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "../../ZenoFX/LinearBvh.h"

namespace zeno{

using T = float;
using vec3 = zs::vec<T,3>;
using vec4 = zs::vec<T,4>;
using mat3 = zs::vec<T,3,3>;
using mat4 = zs::vec<T,4,4>;

struct ZSComputeBaryCentricWeights : INode {
    static constexpr T ComputeDistToTriangle(const vec3& vp,const vec3& v0,const vec3& v1,const vec3& v2){
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

    static constexpr T ComputeVolume(
        const vec3& p0,
        const vec3& p1,
        const vec3& p2,
        const vec3& p3
    ) {
        mat4 m{};
        for(size_t i = 0;i < 3;++i){
            m(i,0) = p0[i];
            m(i,1) = p1[i];
            m(i,2) = p2[i];
            m(i,3) = p3[i];
        }
        m(3,0) = m(3,1) = m(3,2) = m(3,3) = 1;
        return zs::determinant(m)/6;
    }

    static constexpr T ComputeArea(
        const vec3& p0,
        const vec3& p1,
        const vec3& p2
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

    static constexpr vec4 ComputeBaryCentricCoordinate(const vec3& p,
        const vec3& p0,
        const vec3& p1,
        const vec3& p2,
        const vec3& p3
    ) {
        auto vol = ComputeVolume(p0,p1,p2,p3);
        auto vol0 = ComputeVolume(p,p1,p2,p3);
        auto vol1 = ComputeVolume(p0,p,p2,p3);      
        auto vol2 = ComputeVolume(p0,p1,p,p3);
        auto vol3 = ComputeVolume(p0,p1,p2,p);
        return vec4{vol0/vol,vol1/vol,vol2/vol,vol3/vol};
    }
    void apply() override {
        using namespace zs;
        auto zsvolume = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");
        // the bvh of zstets
        // auto lbvh = get_input<zeno::LBvh>("lbvh");
        auto thickness = get_param<float>("bvh_thickness");
        auto fitting_in = get_param<int>("fitting_in");

        auto bvh_channel = get_param<std::string>("bvh_channel");
        auto tag = get_param<std::string>("tag");

        const auto& verts = zsvolume->getParticles();
        const auto& eles = zsvolume->getQuadraturePoints();

        const auto& everts = zssurf->getParticles();
        const auto& e_eles = zssurf->getQuadraturePoints();

        auto &bcw = (*zsvolume)[tag];
        bcw = typename ZenoParticles::particles_t({{"inds",1},{"w",4},{"cnorm",1}},everts.size(),zs::memsrc_e::device,0);


        auto cudaExec = zs::cuda_exec();
        const auto numFEMVerts = verts.size();
        const auto numFEMEles = eles.size();
        const auto numEmbedVerts = bcw.size();
        const auto numEmbedEles = e_eles.size();

        constexpr auto space = zs::execspace_e::cuda;

        // fmt::print("NM_ELES : {}\n",numFEMEles);
        // cudaExec(zs::range(numFEMEles),
        //     [eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {
        //         auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
        //         printf("ELE<%d> : %d\t%d\t%d\t%d\n",ei,inds[0],inds[1],inds[2],inds[3]);
        //     });


        auto bvs = retrieve_bounding_volumes(cudaExec,verts,eles,wrapv<4>{},thickness,bvh_channel);
        auto tetsBvh = zsvolume->bvh(ZenoParticles::s_elementTag);

        tetsBvh.build(cudaExec,bvs);




        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw)] ZS_LAMBDA(int vi) mutable{
                bcw("inds",vi) = reinterpret_bits<float>(int(-1));
            });


        cudaExec(zs::range(numEmbedVerts),
            [verts = proxy<space>({},verts),eles = proxy<space>({},eles),bcw = proxy<space>({},bcw),everts = proxy<space>({},everts),tetsBvh = proxy<space>(tetsBvh),fitting_in] ZS_LAMBDA (int vi) mutable {
                const auto& p = everts.pack<3>("x",vi);
                T closest_dist = 1e6;
                bool found = false;
                tetsBvh.iter_neighbors(p,[&](int ei){

                    if(found)
                        return;

                    auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
                    auto p0 = verts.pack<3>("x",inds[0]);
                    auto p1 = verts.pack<3>("x",inds[1]);
                    auto p2 = verts.pack<3>("x",inds[2]);
                    auto p3 = verts.pack<3>("x",inds[3]);

                    auto ws = ComputeBaryCentricCoordinate(p,p0,p1,p2,p3);

                    // if(vi == 0){
                    //     printf("ITER_V<%d> ON E<%d> : W[%f %f %f %f]\n",
                    //         vi,ei,(float)ws[0],(float)ws[1],(float)ws[2],(float)ws[3]
                    //     );
                    // }

                    T epsilon = zs::limits<T>::epsilon();
                    if(ws[0] > -epsilon && ws[1] > -epsilon && ws[2] > -epsilon && ws[3] > -epsilon){
                        bcw("inds",vi) = reinterpret_bits<float>(ei);
                        bcw.tuple<4>("w",vi) = ws;
                        found = true;
                        return;
                    }
                    
                    if(!fitting_in)
                        return;

                    if(ws[0] < 0){
                        T dist = ComputeDistToTriangle(p,p1,p2,p3);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw("inds",vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>("w",vi) = ws;
                        }
                    }
                    if(ws[1] < 0){
                        T dist = ComputeDistToTriangle(p,p0,p2,p3);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw("inds",vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>("w",vi) = ws;
                        }
                    }
                    if(ws[2] < 0){
                        T dist = ComputeDistToTriangle(p,p0,p1,p3);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw("inds",vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>("w",vi) = ws;
                        }
                    }
                    if(ws[3] < 0){
                        T dist = ComputeDistToTriangle(p,p0,p1,p2);
                        if(dist < closest_dist){
                            closest_dist = dist;
                            bcw("inds",vi) = reinterpret_bits<float>(ei);
                            bcw.tuple<4>("w",vi) = ws;
                        }
                    }
                });

                // if(!found && !fitting_in){
                //     if(vi == 0)
                // }
            }
        );

        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw),fitting_in] ZS_LAMBDA (int vi) mutable {
                auto idx = reinterpret_bits<int>(bcw("inds",vi));
                if(idx < 0 && fitting_in){
                    printf("INVALID INTERPOLATED VERTS<%d> : %d\n",vi,idx);
                }
        });

        auto e_dim = e_eles.getChannelSize("inds");

        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw)] ZS_LAMBDA (int vi) mutable {
                using T = typename RM_CVREF_T(bcw)::value_type;
                bcw("cnorm",vi) = (T)0.;
        });

        zs::Vector<T> edgeCount(bcw.get_allocator(),bcw.size());
        cudaExec(zs::range(bcw.size()),[edgeCount = proxy<space>(edgeCount)]
            ZS_LAMBDA(int vi) mutable{
                using T = typename RM_CVREF_T(bcw)::value_type;
                edgeCount[vi] = (T)0.;
        });

        if(e_dim !=3 && e_dim !=4){
            throw std::runtime_error("INVALID EMBEDDED PRIM TOPO");
        }
        cudaExec(zs::range(numEmbedEles),
            [everts = proxy<space>({},everts),e_eles = proxy<space>({},e_eles),bcw = proxy<space>({},bcw),e_dim,execTag = wrapv<space>{},edgeCount = proxy<space>(edgeCount)]
                ZS_LAMBDA (int ti) mutable {
                    using T = typename RM_CVREF_T(bcw)::value_type;
                    if(e_dim == 3){// for triangle mesh
                        auto inds = e_eles.pack<3>("inds",ti).reinterpret_bits<int>(); 
                        auto p0 = everts.pack<3>("x",inds[0]);
                        auto p1 = everts.pack<3>("x",inds[1]);
                        auto p2 = everts.pack<3>("x",inds[2]);

                        auto A = ComputeArea(p0,p1,p2)/3;
                        auto l01 = (p0 - p1).norm();
                        auto l02 = (p0 - p2).norm();
                        auto l12 = (p1 - p2).norm();

                        if(l12 > zs::limits<T>::epsilon() * 10){
                            atomic_add(execTag,&bcw("cnorm",inds[0]),(T)(2*A/l12));
                            atomic_add(execTag,&edgeCount[inds[0]],(T)1.0);
                        }
                        if(l02 > zs::limits<T>::epsilon() * 10){
                            atomic_add(execTag,&bcw("cnorm",inds[1]),(T)(2*A/l02));
                            atomic_add(execTag,&edgeCount[inds[1]],(T)1.0);
                        }
                        if(l01 > zs::limits<T>::epsilon() * 10){
                            atomic_add(execTag,&bcw("cnorm",inds[2]),(T)(2*A/l01));
                            atomic_add(execTag,&edgeCount[inds[2]],(T)1.0);
                        }

                        // bcw("area",inds[0]) += aA;
                        // bcw("area",inds[1]) += aA;
                        // bcw("area",inds[2]) += aA;
                    }else if(e_dim == 4){// for tet mesh
                        auto inds = e_eles.pack<4>("inds",ti).reinterpret_bits<int>();
                        auto p0 = everts.pack<3>("x",inds[0]);
                        auto p1 = everts.pack<3>("x",inds[1]);
                        auto p2 = everts.pack<3>("x",inds[2]);
                        auto p3 = everts.pack<3>("x",inds[3]);

                        auto V = zs::abs(ComputeVolume(p0,p1,p2,p3));
                        auto A012 = ComputeArea(p0,p1,p2);
                        auto A023 = ComputeArea(p0,p2,p3);
                        auto A013 = ComputeArea(p0,p1,p3);
                        auto A123 = ComputeArea(p1,p2,p3); 

                        if(A123 > limits<T>::epsilon()){
                            atomic_add(execTag,&bcw("cnorm",inds[0]),(T)(3*V/A123));
                            atomic_add(execTag,&edgeCount[inds[0]],(T)1.0);
                        }
                        if(A023 > limits<T>::epsilon()){
                            atomic_add(execTag,&bcw("cnorm",inds[1]),(T)(3*V/A023));
                            atomic_add(execTag,&edgeCount[inds[1]],(T)1.0);
                        }
                        if(A013 > limits<T>::epsilon()){
                            atomic_add(execTag,&bcw("cnorm",inds[2]),(T)(3*V/A013));
                            atomic_add(execTag,&edgeCount[inds[2]],(T)1.0);
                        }
                        if(A012 > limits<T>::epsilon()){
                            atomic_add(execTag,&bcw("cnorm",inds[3]),(T)(3*V/A012));
                            atomic_add(execTag,&edgeCount[inds[3]],(T)1.0);
                        }
                    }
        });

        cudaExec(zs::range(numEmbedVerts),
            [bcw = proxy<space>({},bcw),edgeCount = proxy<space>(edgeCount)] 
                ZS_LAMBDA(int vi) mutable{
                    if(edgeCount[vi] > 0)
                        bcw("cnorm",vi) /= edgeCount[vi];
        });


        set_output("zsvolume", zsvolume);
    }
};

ZENDEFNODE(ZSComputeBaryCentricWeights, {{{"interpolator","zsvolume"}, {"embed surf", "zssurf"}},
                            {{"interpolator on gpu", "zsvolume"}},
                            {{"float","bvh_thickness","0"},{"int","fitting_in","1"},{"string","tag","skin_bw"},{"string","bvh_channel","x"}},
                            {"FEM"}});


struct ZSInterpolateEmbedPrim : zeno::INode {
    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zsvolume");
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto tag = get_param<std::string>("tag");
        auto outAttr = get_param<std::string>("outAttr");

        auto cudaExec = zs::cuda_exec();

        auto &everts = zssurf->getParticles();
        // everts.append_channels(cudaExec, {{outAttr, 3}});
        
        const auto& verts = zstets->getParticles();
        const auto& eles = zstets->getQuadraturePoints();
        const auto& bcw = (*zstets)[tag];

        const auto nmEmbedVerts = bcw.size();

        if(everts.size() != nmEmbedVerts)
            throw std::runtime_error("INPUT SURF SIZE AND BCWS SIZE DOES NOT MATCH");


        constexpr auto space = zs::execspace_e::cuda;

        cudaExec(zs::range(nmEmbedVerts),
            [outAttr = zs::SmallString{outAttr},verts = proxy<space>({},verts),eles = proxy<space>({},eles),bcw = proxy<space>({},bcw),everts = proxy<space>({},everts)] ZS_LAMBDA (int vi) mutable {
                const auto& ei = bcw.pack<1>("inds",vi).reinterpret_bits<int>()[0];
                if(ei < 0)
                    return;
                const auto& w = bcw.pack<4>("w",vi);

                everts.tuple<3>(outAttr,vi) = vec3::zeros();

                for(size_t i = 0;i < 4;++i){
                    // const auto& idx = eles.pack<4>("inds",ei).reinterpret_bits<int>()[i];
                    const auto idx = reinterpret_bits<int>(eles("inds", i, ei));
                    everts.tuple<3>(outAttr,vi) = everts.pack<3>(outAttr,vi) + w[i] * verts.pack<3>("x", idx);
                }
#if 0
                if(vi == 100){
                    auto vert = everts.pack<3>(outAttr,vi);
                    printf("V<%d>->E<%d>(%f,%f,%f,%f) :\t%f\t%f\t%f\n",vi,ei,w[0],w[1],w[2],w[3],vert[0],vert[1],vert[2]);
                }
#endif
        });
        set_output("zssurf",zssurf);
    }
};

ZENDEFNODE(ZSInterpolateEmbedPrim, {{{"zsvolume"}, {"embed primitive", "zssurf"}},
                            {{"embed primitive", "zssurf"}},
                            {{"string","outAttr","x"},{"string","tag","skin_bw"}},
                            {"FEM"}});


} // namespace zeno