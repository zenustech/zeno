#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
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

#include "kernel/laplace_matrix.hpp"

namespace zeno {

struct ZSEvalBBW : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;
    struct HarmonicSystem {
        template<typename Pol> 
        void project(Pol& pol,const zs::SmallString& btag,const tiles_t& verts,const zs::SmallString& tag, dtiles_t& vtemp,int wdim) const {
            using namespace zs;
            constexpr execspace_e space = execspace_e::cuda;
            pol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),btag,tag,wdim]
                        ZS_LAMBDA(int vi) mutable {
                    if(verts(btag,vi) > 0.0)
                        return;
                    for(size_t d = 0;d != wdim;++d)
                        vtemp(tag,d,vi) = (T)0.0;
                });
        }   

        // the right hand-side are all zeros;
        template<typename Pol>
        void rhs(Pol& pol,const zs::SmallString& tag,const zs::SmallString& rTag,dtiles_t& vtemp,int wdim) const {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            const auto numVerts = verts.size();
            const auto numEles = eles.size();
            // b -> 0
            pol(range(numVerts),
                [vtemp = proxy<space>({},vtemp),rTag,wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d != wdim;++d)
                        vtemp(rTag,d,vi) = (T)0.0;
                });
        }

        template<typename Pol>
        void multiply(Pol& pol,const zs::SmallString& dxTag,
                    const zs::SmallString& bTag,
                    dtiles_t& vtemp,
                    const dtiles_t& etemp,int wdim) const {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            constexpr auto execTag = wrapv<space>{};
            const auto numVerts = verts.size();
            const auto numEles = eles.size();

            // b -> 0
            pol(range(numVerts),
                [execTag,vtemp = proxy<space>({},vtemp),bTag,wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d != wdim;++d)
                        vtemp(bTag,d,vi) = (T)0.0;
                });
            // compute Ldx->b
            pol(range(numEles),[execTag,etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles),dxTag,bTag,wdim]
                ZS_LAMBDA(int ei) mutable {
                    constexpr int cdim = 4;
                    auto inds = eles.pack<cdim>("inds",ei).reinterpret_bits<int>();
                    zs::vec<T,cdim> temp{};
                    for(size_t d = 0;d != wdim;++d){
                        for(int vi = 0;vi != cdim;++vi)
                            temp[vi] = vtemp(dxTag,d,inds[vi]);

                        auto He = etemp.pack<cdim,cdim>("L",ei);
                        temp = He * temp;
                        for(int vi = 0;vi != cdim;++vi)
                            atomic_add(execTag,&vtemp(bTag,d,inds[vi]),temp[vi]);
                    }
            });

            // compute MLdx->b
            // the vtemp contains a nodal-volume
            pol(range(numEles),[vtemp = proxy<space>({},vtemp),bTag,wdim]
                ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d != wdim;++d)
                        vtemp(bTag,d,vi) /= vtemp("vol",vi);
            });
            // compute LMLdx->b
            pol(range(numEles),[execTag,etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles),dxTag,bTag,wdim]
                ZS_LAMBDA(int ei) mutable {
                    constexpr int cdim = 4;
                    auto inds = eles.pack<cdim>("inds",ei).reinterpret_bits<int>();
                    zs::vec<T,cdim> temp{};
                    for(size_t d = 0;d != wdim;++d){
                        for(int vi = 0;vi != cdim;++vi)
                            temp[vi] = vtemp(dxTag,d,inds[vi]);

                        auto He = etemp.pack<cdim,cdim>("L",ei);
                        temp = He * temp;
                        for(int vi = 0;vi != cdim;++vi)
                            atomic_add(execTag,&vtemp(bTag,d,inds[vi]),temp[vi]);
                    }
            });
        }
        // for biharmonic equation, using laplace diagonal preconditioner seems reasonable
        template <typename Pol>
        void precondition(Pol &pol, const zs::SmallString srcTag,
                        const zs::SmallString dstTag,dtiles_t& vtemp,int wdim) const {
            using namespace zs;
            constexpr execspace_e space = execspace_e::cuda;
            // precondition
            pol(zs::range(verts.size()),
                [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts), srcTag, dstTag, wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d < wdim;++d)
                        vtemp(dstTag,d,vi) = vtemp("P",vi) * vtemp(srcTag,d,vi);
                });
        }

        HarmonicSystem(const tiles_t& verts,const tiles_t& eles) : verts{verts},eles{eles} {}

        const tiles_t &verts;
        const tiles_t &eles;
    };

    template<int pack_dim = 3>
    T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
            const zs::SmallString tag0, const zs::SmallString tag1) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vertData.get_allocator(), 1};
        res.setVal(0);
        cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
            tag1] __device__(int pi) mutable {
                auto v0 = data.pack<pack_dim>(tag0, pi);
                auto v1 = data.pack<pack_dim>(tag1, pi);

                atomic_add(exec_cuda, res.data(), v0.dot(v1));
            });
        return res.getVal();
    }
    T infNorm(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
                const zs::SmallString tag = "dir") {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vertData.get_allocator(), 1};
        res.setVal(0);
        cudaPol(range(vertData.size()),
                [data = proxy<space>({}, vertData), res = proxy<space>(res),
                tag] __device__(int pi) mutable {
                auto v = data.pack<3>(tag, pi);
                atomic_max(exec_cuda, res.data(), v.abs().max());
                });
        return res.getVal();
    }

    template<typename Equation,typename VTileVec,typename ETileVec>
    void pcg_with_fixed_solve(zs::CudaExecutionPolicy &cudaPol,const Equation& A,
            const VTileVec& verts,const ETileVec& eles,int wdim,
            const zs::SmallString& wtag,
            dtiles_t& vtemp,dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),tag = zs::SmallString(wtag),wdim]
                    ZS_LAMBDA(int vi) mutable{
                        for(size_t d = 0;d != wdim;++d)
                            vtemp("x",d,vi) = verts(tag,d,vi);
            });
            A.rhs(cudaPol,"x","b",vtemp,wdim);
            A.multiply(cudaPol,"x","temp",vtemp,etemp,wdim);
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d != wdim;++d)
                        vtemp("r",d,vi) = vtemp("b",d,vi) - vtemp("temp",d,vi);
                });

            A.project(cudaPol,"btag",verts,"r",vtemp,wdim);
            A.precondition(cudaPol,"r","q",vtemp,wdim);
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    vtemp("p",vi) = vtemp("q",vi);
            });

            T zTrk = dot(cudaPol,vtemp,"r","q");
            if(std::isnan(zTrk)){
                T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
                T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
                T gn = std::sqrt(dot(cudaPol,vtemp,"grad","grad"));
                T Pn = std::sqrt(dot<9>(cudaPol,vtemp,"P","P"));

                fmt::print("NAN zTrk Detected r: {} q: {}, gn:{} Pn:{}\n",rn,qn,gn,Pn);
                throw std::runtime_error("NAN zTrk");
            }
            if(zTrk < 0){
                T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
                T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
                fmt::print("\t#Begin invalid zTrk found  with zTrk {} and b{} and r {} and q {}\n",
                    zTrk, infNorm(cudaPol, vtemp, "b"),rn,qn);

                fmt::print("FOUND NON_SPD P\n");
                cudaPol(zs::range(vtemp.size()),
                    [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi){
                        auto P = vtemp("P",vi);
                        if(P < 0) 
                            printf("NON_SPD_P<%d> %f\n",vi,(float)P);
                    });
                throw std::runtime_error("INVALID zTrk");
            }

            auto residualPreconditionedNorm = std::sqrt(zTrk);
            // auto localTol = std::min(0.5 * residualPreconditionedNorm, 1.0);
            auto localTol = 0.1 * residualPreconditionedNorm;
            // if(newtonIter < 10)
            //     localTol = 0.5 * residualPreconditionedNorm;
            int iter = 0;            
            for (; iter != 1000; ++iter) {
                if (iter % 50 == 0)
                    fmt::print("cg iter: {}, norm: {} zTrk: {} localTol: {}\n", iter,
                                residualPreconditionedNorm,zTrk,localTol);
                if(zTrk < 0){
                    T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
                    T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
                    fmt::print("\t# invalid zTrk found in {} iters with zTrk {} and r {} and q {}\n",
                        iter, zTrk,rn,qn);

                    fmt::print("FOUND NON_SPD P\n");
                    cudaPol(zs::range(vtemp.size()),
                        [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi){
                            auto P = vtemp("P",vi);
                            if(P < 0) 
                                printf("NON_SPD_P<%d> %f\n",vi,(float)P);
                    });
                    throw std::runtime_error("INVALID zTrk");
                }

                if (residualPreconditionedNorm <= localTol){ // this termination criterion is dimensionless
                    fmt::print("finish with cg iter: {}, norm: {} zTrk: {}\n", iter,
                                residualPreconditionedNorm,zTrk);
                    break;
                }
                A.multiply(cudaPol, "p", "temp",vtemp,etemp,wdim);
                A.project(cudaPol,"btag",verts, "temp",vtemp,wdim);

                T alpha = zTrk / dot(cudaPol, vtemp, "temp", "p");
                cudaPol(range(verts.size()), [verts = proxy<space>({}, verts),
                                    vtemp = proxy<space>({}, vtemp),
                                    alpha,wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d != wdim;++d){
                        vtemp("x", d, vi) += alpha * vtemp("p", vi);
                        vtemp("r", d, vi) -= alpha * vtemp("temp", vi);
                    }
                });
                A.precondition(cudaPol, "r", "q",vtemp,wdim);
                auto zTrkLast = zTrk;
                zTrk = dot(cudaPol, vtemp, "q", "r");
                auto beta = zTrk / zTrkLast;
                cudaPol(range(verts.size()), [vtemp = proxy<space>({}, vtemp),beta] ZS_LAMBDA(int vi) mutable {
                    vtemp("p", vi) = vtemp("q", vi) + beta * vtemp("p", vi);
                });
                residualPreconditionedNorm = std::sqrt(zTrk);            
            }
            fmt::print("FINISH SOLVING PCG with cg_iter = {}\n",iter);  
    }

    template<typename Equation,typename VTileVec,typename ETileVec>
    void active_set_with_fixed_solve(zs::CudaExecutionPolicy &cudaPol,const Equation& A,
            const VTileVec& verts,const ETileVec& eles,int wdim,
            const zs::SmallString& wtag,
            dtiles_t& vtemp,dtiles_t& etemp) {

        pcg_with_fixed_solve(cudaPol,A,verts,eles,wdim,wtag,vtemp,etemp);
        // only enforce the positive weights
        // initialize the active_set as empty, only the lower bound constraint is needed
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),wdim] ZS_LAMBDA(int vi) mutable{
                for(size_t d = 0;d < wdim;++d)
                    vtemp("w_old",d,vi) = limits<T>::max();
                    vtemp("as_lx",vi) = false;
            });

        constexpr T solution_res = (T)1e-8;

        while(true) {
            int new_as_lx = 0;
            // check the violations
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),wtag.wdim] ZS_LAMBDA(int vi) mutable{
                    for(size_t d = 0;d < wdim;++d)
                        vtemp("temp",d,vi) = vtemp(wtag,d,vi) - vtemp("w_old",d,vi);
                }
            )
            auto diffnorm = std::sqrt(dot(vtemp,"temp","temp"));
            if(diffnorm < solution_res)
                break;

            // mark the new constrained nodes
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),new_as_lx,wtag,wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d < wdim;++d){
                        if(vtemp(wtag,d,vi) < limits<T>::min()){
                            vtemp(wtag,d,vi) = limits<T>::min()*2;
                            vtemp("btag",d,vi) = (T)1.0;
                        }
                    }
                });

            pcg_with_fixed_solve(cudaPol,A,verts,eles,wdim,wtag,vtemp,etemp);


            // relieve the constrained nodes with negative lagrangian
            

        }
    }     

    virtual void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zstets");
        auto wtag = get_param<std::string>("wtag");
        auto btag = get_param<std::string>("btag");

        auto& verts = zstets->getParticles();
        auto& eles = zstets->getQuadraturePoints();

        if(!verts.hasProperty(wtag)){
            fmt::print("The input tets does not contain specified property:{}\n",wtag);
        }
        if(!verts.hasProperty(btag)){
            fmt::print("The input tets does not contain specified property:{}\n",btag);
        }

        int wdim = verts.getChannelSize(wtag);
        if(wdim == 0){
            throw std::runtime_error("The input tets does not contain bbw channel\n");
        }
        
        int cdim = eles.getChannelSize("inds");
        if(cdim != 4){
            throw std::runtime_error("Only tetrahedra is currently supported\n");
        }

        static dtiles_t etemp{eles.get_allocator(),{{"L",cdim*cdim}},eles.size()};
        static dtiles_t vtemp{verts.get_allocator(),{
            {"x",wdim},
            {"b",wdim},
            {"P",1},
            {"temp",wdim},
            {"r",wdim},
            {"p",wdim},
            {"q",wdim}
        },verts.size()};

        etemp.resize(eles.size());
        vtemp.resize(verts.size());

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        compute_cotmatrix(cudaPol,eles,verts,"x",etemp,"L",zs::wrapv<4>{});
        // compute the residual
        HarmonicSystem A{verts,eles};
        // compute preconditioner
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp),
                verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                    vtemp("P", vi) = (T)0.0;
        });

        cudaPol(zs::range(eles.size()),
                    [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles)]
                        ZS_LAMBDA(int ei) mutable{
             constexpr int cdim = 4;
             auto inds = eles.template pack<4>("inds",ei).template reinterpret_bits<int>();
             auto H = etemp.pack<cdim,cdim>("L",ei);
             for(int vi = 0;vi != cdim;++vi)
                atomic_add(exec_cuda,&vtemp("P",inds[vi]),(T)H(vi,vi));
        });

        cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable{
                vtemp("P",vi) = 1./vtemp("P",vi);
        });

        // Solve BBW using active-set method
        active_set_with_fixed_solve(cudaPol,A,verts,eles,wdim,zs::SmallString(wtag),vtemp,etemp);

        cudaPol(zs::range(verts.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),tag = zs::SmallString(wtag),wdim] ZS_LAMBDA(int vi) mutable {
                    for(size_t d = 0;d != wdim;++d)
                        verts(tag,d,vi) = vtemp("x",d,vi);
        });

        set_output("zstets",zstets);        
    }

};

};