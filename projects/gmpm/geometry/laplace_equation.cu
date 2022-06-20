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

struct ZSSolveLaplaceEquaOnTets : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;
    struct LaplaceSystem {
        template<typename Pol> 
        void project(Pol& pol,const zs::SmallString& btag,tiles_t& verts,const zs::SmallString& tag, dtiles_t& vtemp) {
            using namespace zs;
            constexpr execspace_e space = execspace_e::cuda;
            pol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),btag,tag]
                        ZS_LAMBDA(int vi) mutable {
                    if(verts(btag,vi) > zs::limits<T>::epsilon())
                        vtemp(tag,vi) = (T)0.0;
                    // vtemp(tag,vi) = (T)0.0;
                });
        }    

        // the right hand-side are all zeros;
        template<typename Pol>
        void rhs(Pol& pol,const zs::SmallString& tag,const zs::SmallString& rTag,dtiles_t& vtemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            const auto numVerts = verts.size();
            const auto numEles = eles.size();
            // b -> 0
            pol(range(numVerts),
                [vtemp = proxy<space>({},vtemp),rTag] ZS_LAMBDA(int vi) mutable {
                    vtemp(rTag,vi) = (T)0.0;
                });
        }

        template<int codim>
        void prepare_preconditioner(zs::CudaExecutionPolicy &pol,const zs::SmallString& HTag,dtiles_t& etemp,const zs::SmallString& PTag,dtiles_t& vtemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            constexpr auto execTag = wrapv<space>{};
            const auto numVerts = verts.size();
            const auto numEles = eles.size();

            pol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp),
                    verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                        vtemp("P", vi) = (T)0.0;
            });

            pol(zs::range(eles.size()),
                        [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles),codim_v = wrapv<codim>{}]
                            ZS_LAMBDA(int ei) mutable{
                constexpr int cdim = RM_CVREF_T(codim_v)::value;
                auto inds = eles.template pack<4>("inds",ei).template reinterpret_bits<int>();
                auto H = etemp.pack<cdim,cdim>("L",ei);
                for(int vi = 0;vi != cdim;++vi)
                    atomic_add(exec_cuda,&vtemp("P",inds[vi]),(T)H(vi,vi));
            });

            pol(zs::range(verts.size()),
                [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable{
                    vtemp("P",vi) = 1./vtemp("P",vi);
                    // vtemp("P",vi) = 1.0;
            });            
        }

        template<int codim>
        void multiply(zs::CudaExecutionPolicy &pol,const zs::SmallString& dxTag,
                    const zs::SmallString& bTag,
                    dtiles_t& vtemp,
                    const dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            constexpr auto execTag = wrapv<space>{};
            const auto numVerts = verts.size();
            const auto numEles = eles.size();

            // b -> 0
            pol(range(numVerts),
                [execTag,vtemp = proxy<space>({},vtemp),bTag] ZS_LAMBDA(int vi) mutable {
                    vtemp(bTag,vi) = (T)0.0;
                });
            // compute Adx->b
            pol(range(numEles),[execTag,etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles),dxTag,bTag,cdim_v = wrapv<codim>{}]
                ZS_LAMBDA(int ei) mutable {
                    constexpr int cdim = RM_CVREF_T(cdim_v)::value;
                    auto inds = eles.template pack<cdim>("inds",ei).template reinterpret_bits<int>();
                    zs::vec<T,cdim> temp{};
                    for(int vi = 0;vi != cdim;++vi)
                        temp[vi] = vtemp(dxTag,inds[vi]);

                    auto He = etemp.pack<cdim,cdim>("L",ei);
                    temp = He * temp;
                    for(int vi = 0;vi != cdim;++vi)
                        atomic_add(execTag,&vtemp(bTag,inds[vi]),temp[vi]);
            });
        }

        template <typename Pol>
        void precondition(Pol &pol, const zs::SmallString srcTag,
                        const zs::SmallString dstTag,dtiles_t& vtemp) {
        using namespace zs;
        constexpr execspace_e space = execspace_e::cuda;
        // precondition
        pol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
            srcTag, dstTag] ZS_LAMBDA(int vi) mutable {
                vtemp(dstTag, vi) =
                    vtemp("P", vi) * vtemp(srcTag, vi);
            });
        }

        LaplaceSystem(const tiles_t& verts,const tiles_t& eles) : verts{verts},eles{eles} {}

        const tiles_t &verts;
        const tiles_t &eles;
    };

    template<int pack_dim = 1>
    T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
            const zs::SmallString tag0, const zs::SmallString tag1) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vertData.get_allocator(), 1};
        res.setVal(0);
        cudaPol(range(vertData.size()),
                [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
                tag1] __device__(int pi) mutable {
                    auto v0 = data.pack<pack_dim>(tag0,pi);
                    auto v1 = data.pack<pack_dim>(tag1,pi);
                    atomic_add(exec_cuda, res.data(), v0.dot(v1));
                });
        return res.getVal();
    }
    template<int pack_dim = 1>
    T infNorm(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
                const zs::SmallString tag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vertData.get_allocator(), 1};
        res.setVal(0);
        cudaPol(range(vertData.size()),
                [data = proxy<space>({}, vertData), res = proxy<space>(res),
                tag] __device__(int pi) mutable {
                auto v = data.pack<pack_dim>(tag, pi);
                atomic_max(exec_cuda, res.data(), v.abs().max());
                });
        return res.getVal();
    }

    virtual void apply() override {
        using namespace zs;
        auto zspars = get_input<ZenoParticles>("ZSParticles");
        // specify the name of a nodal attribute
        auto attr = get_param<std::string>("tag");
        auto& verts = zspars->getParticles();
        auto accuracy = get_param<float>("accuracy");
        //  make sure the input zspars has specified attributes
        if(!verts.hasProperty(attr)){
            fmt::print("the input zspars does not contain specified channel:{}\n",attr);
            throw std::runtime_error("the input zspars does not contain specified channel");
        }

        if(!verts.hasProperty("btag")){
            fmt::print("the input zspars does not contain 'btag' channel\n");
            throw std::runtime_error("the input zspars does not contain specified channel");
        }

        const auto& eles = zspars->getQuadraturePoints();
        auto cdim = eles.getChannelSize("inds");
        if(cdim != 4 && cdim != 3){
            fmt::print("INVALID SIMPLEX SIZE : {}\n",cdim);
            throw std::runtime_error("ZSSolveLaplaceEquaOnTets: invalid simplex size");
        }

        static dtiles_t etemp{eles.get_allocator(),{{"L",cdim*cdim}},eles.size()};
        static dtiles_t vtemp{verts.get_allocator(),{
            {"x",1},
            {"b",1},
            {"P",1},
            {"temp",1},
            {"r",1},
            {"p",1},
            {"q",1}
        },verts.size()};
        
        etemp.resize(eles.size());
        vtemp.resize(verts.size());

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        // compute per-element laplace operator
        // fmt::print("COMPUTE COTMATRIX\n");
        if(cdim == 4)
            compute_cotmatrix(cudaPol,eles,verts,"x",etemp,"L",zs::wrapv<4>{});
        else
            compute_cotmatrix(cudaPol,eles,verts,"x",etemp,"L",zs::wrapv<3>{});
        fmt::print("FINISH COMPUTE COTMATRIX\n");
        // compute the residual
        LaplaceSystem A{verts,eles};
        // compute preconditioner
        fmt::print("prepare preconditioner\n");
        if(cdim == 4)
            A.prepare_preconditioner<4>(cudaPol,"L",etemp,"P",vtemp);
        else
            A.prepare_preconditioner<3>(cudaPol,"L",etemp,"P",vtemp);
        fmt::print("finish setup preconditioner\n");
        // initial guess
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),tag = zs::SmallString(attr)] ZS_LAMBDA(int vi) mutable {
                vtemp("x",vi) = verts(tag,vi);
            }
        );

        // if(cdim == 3){
        //     cudaPol(zs::range(etemp.size()),
        //         [etemp = proxy<space>({},etemp)] ZS_LAMBDA(int ei) mutable {
        //             auto L = etemp.pack<3,3>("L",ei);
        //             T Ln = L.norm();
        //             if(isnan(Ln)){
        //                 printf("NAN LN<%d> : \n%f %f %f\n%f %f %f\n%f %f %f\n",ei,
        //                     (float)L(0,0),(float)L(0,1),(float)L(0,2),
        //                     (float)L(1,0),(float)L(1,1),(float)L(1,2),
        //                     (float)L(2,0),(float)L(2,1),(float)L(2,2));
        //             }
        //         });

        //     auto Ln = dot<9>(cudaPol,vtemp,"P","P");
        //     auto Pn = dot(cudaPol,vtemp,"P","P");

        //     fmt::print("Ln : {}\t Pn : {}\n",Ln,Pn);
        // }

        // Solve Laplace Equation Using PCG
        {
            // set the initial guess of the solution subject to boundary condition
            // eval the right hand side
            A.rhs(cudaPol,"x","b",vtemp);
            if(cdim == 4)
                A.multiply<4>(cudaPol,"x","temp",vtemp,etemp);
            else
                A.multiply<3>(cudaPol,"x","temp",vtemp,etemp);

            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    vtemp("r",vi) = vtemp("b",vi) - vtemp("temp",vi);
                });

            A.project(cudaPol,"btag",verts,"r",vtemp);
            A.precondition(cudaPol,"r","q",vtemp);

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
            auto localTol = accuracy * residualPreconditionedNorm;
            // if(newtonIter < 10)
            //     localTol = 0.5 * residualPreconditionedNorm;
            int iter = 0;            
            for (; iter != 1000; ++iter) {
                // if (iter % 50 == 0){
                //     // recalculate the residual every 50 iterations
                //     fmt::print("cg iter: {}, norm: {} zTrk: {} localTol: {}\n", iter,
                //                 residualPreconditionedNorm,zTrk,localTol);
                // }
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
                    fmt::print("finish with cg iter: {}, norm: {} zTrk: {}\n",iter,
                                residualPreconditionedNorm,zTrk);
                    break;
                }
                if(cdim == 4)
                    A.multiply<4>(cudaPol, "p", "temp",vtemp,etemp);
                else
                    A.multiply<3>(cudaPol, "p", "temp",vtemp,etemp);
                A.project(cudaPol,"btag",verts, "temp",vtemp);

                T alpha = zTrk / dot(cudaPol, vtemp, "temp", "p");

                cudaPol(range(verts.size()), [verts = proxy<space>({}, verts),
                                    vtemp = proxy<space>({}, vtemp),
                                    alpha] ZS_LAMBDA(int vi) mutable {
                    vtemp("x", vi) += alpha * vtemp("p", vi);
                    vtemp("r", vi) -= alpha * vtemp("temp", vi);
                });
                // recalcute the residual every 50 iterations
                if(iter % 51 == 50){
                    A.rhs(cudaPol,"x","b",vtemp);
                    if(cdim == 4)
                        A.multiply<4>(cudaPol,"x","temp",vtemp,etemp);
                    else
                        A.multiply<3>(cudaPol,"x","temp",vtemp,etemp);
                    cudaPol(zs::range(vtemp.size()),
                        [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                            vtemp("r",vi) = vtemp("b",vi) - vtemp("temp",vi);
                        });
                    A.project(cudaPol,"btag",verts,"r",vtemp);
                }             
                A.precondition(cudaPol, "r", "q",vtemp);
                A.project(cudaPol,"btag",verts,"q",vtemp);
                auto zTrkLast = zTrk;
                zTrk = dot(cudaPol, vtemp, "q", "r");
                if (iter % 50 == 0){
                    fmt::print("cg iter: {}, norm: {} zTrk: {} localTol: {}\n", iter,
                                residualPreconditionedNorm,zTrk,localTol);
                }

                auto beta = zTrk / zTrkLast;
                cudaPol(range(verts.size()), [vtemp = proxy<space>({}, vtemp),beta] ZS_LAMBDA(int vi) mutable {
                    vtemp("p", vi) = vtemp("q", vi) + beta * vtemp("p", vi);
                });
                residualPreconditionedNorm = std::sqrt(zTrk);
                ++iter;
            }
            fmt::print("FINISH SOLVING PCG with cg_iter = {}\n",iter);  
        }// end cg step

        cudaPol(zs::range(verts.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),tag = zs::SmallString(attr)] ZS_LAMBDA(int vi) mutable {
                    verts(tag,vi) = vtemp("x",vi);
        });

        set_output("ZSParticles",zspars);
    }
};

ZENDEFNODE(ZSSolveLaplaceEquaOnTets, {
                                    {"ZSParticles"},
                                    {"ZSParticles"},
                                    {
                                        {"string","tag","T"},{"float","accuracy","1e-6"}
                                    },
                                    {"FEM"}
});


// the biharmonic hessian can be eval as LML, where M is a diagonal matrix with diagonal entries the inverse of nodal volume
struct ZSSolveBiHarmonicEquaOnTets : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;

    struct HarmonicSystem {
        template<typename Pol> 
        void project(Pol& pol,const zs::SmallString& btag,tiles_t& verts,const zs::SmallString& tag, dtiles_t& vtemp) {
            using namespace zs;
            constexpr execspace_e space = execspace_e::cuda;
            pol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),btag,tag]
                        ZS_LAMBDA(int vi) mutable {
                    if(verts(btag,vi) > 0.0)
                        return;
                    vtemp(tag,vi) = (T)0.0;
                });
        }   

        // the right hand-side are all zeros;
        template<typename Pol>
        void rhs(Pol& pol,const zs::SmallString& tag,const zs::SmallString& rTag,dtiles_t& vtemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            const auto numVerts = verts.size();
            const auto numEles = eles.size();
            // b -> 0
            pol(range(numVerts),
                [vtemp = proxy<space>({},vtemp),rTag] ZS_LAMBDA(int vi) mutable {
                    vtemp(rTag,vi) = (T)0.0;
                });
        }

        template<typename Pol>
        void multiply(Pol& pol,const zs::SmallString& dxTag,
                    const zs::SmallString& bTag,
                    dtiles_t& vtemp,
                    const dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            constexpr auto execTag = wrapv<space>{};
            const auto numVerts = verts.size();
            const auto numEles = eles.size();

            // b -> 0
            pol(range(numVerts),
                [execTag,vtemp = proxy<space>({},vtemp),bTag] ZS_LAMBDA(int vi) mutable {
                    vtemp(bTag,vi) = (T)0.0;
                });
            // compute Ldx->b
            pol(range(numEles),
                [execTag,etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles),dxTag,bTag]
                ZS_LAMBDA(int ei) mutable {
                    // constexpr int cdim = RM_CVREF_T(codim_v)::value;
                    constexpr int cdim = 4;
                    auto inds = eles.pack<cdim>("inds",ei).reinterpret_bits<int>();
                    zs::vec<T,cdim> temp{};
                    for(int vi = 0;vi != cdim;++vi)
                        temp[vi] = vtemp(dxTag,inds[vi]);

                    auto He = etemp.pack<cdim,cdim>("L",ei);
                    temp = He * temp;
                    for(int vi = 0;vi != cdim;++vi)
                        atomic_add(execTag,&vtemp(bTag,inds[vi]),temp[vi]);
            });

            // compute MLdx->b
            // the vtemp contains a nodal-volume
            pol(range(numEles),[vtemp = proxy<space>({},vtemp),bTag]
                ZS_LAMBDA(int vi) mutable {
                    vtemp(bTag,vi) /= vtemp("vol",vi);
            });
            // compute LMLdx->b
            pol(range(numEles),[execTag,etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles),dxTag,bTag]
                ZS_LAMBDA(int ei) mutable {
                    constexpr int cdim = 4;
                    auto inds = eles.pack<cdim>("inds",ei).reinterpret_bits<int>();
                    zs::vec<T,cdim> temp{};
                    for(int vi = 0;vi != cdim;++vi)
                        temp[vi] = vtemp(dxTag,inds[vi]);

                    auto He = etemp.pack<cdim,cdim>("L",ei);
                    temp = He * temp;
                    for(int vi = 0;vi != cdim;++vi)
                        atomic_add(execTag,&vtemp(bTag,inds[vi]),temp[vi]);
            });
        }
        // for biharmonic equation, using laplace diagonal preconditioner seems reasonable
        template <typename Pol>
        void precondition(Pol &pol, const zs::SmallString srcTag,
                        const zs::SmallString dstTag,dtiles_t& vtemp) {
            using namespace zs;
            constexpr execspace_e space = execspace_e::cuda;
            // precondition
            pol(zs::range(verts.size()),
                [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
                srcTag, dstTag] ZS_LAMBDA(int vi) mutable {
                    vtemp(dstTag, vi) =
                        vtemp("P", vi) * vtemp(srcTag, vi);
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

    virtual void apply() override {
        using namespace zs;
        auto zspars = get_input<ZenoParticles>("ZSParticles");
        auto attr = get_param<std::string>("tag");
        auto k = get_param<int>("k");
        auto& verts = zspars->getParticles();

        if(!verts.hasProperty(attr)){
            fmt::print("the input zspars does not contain specified channel:{}\n",attr);
            throw std::runtime_error("the input zspars does not contain specified channel");
        }

        if(!verts.hasProperty("btag")){
            fmt::print("the input zspars does not contain 'btag' channel\n");
            throw std::runtime_error("the input zspars does not contain specified channel");
        }

        if(k != 1 && k == 2){

        }

        const auto& eles = zspars->getQuadraturePoints();
        auto cdim = eles.getChannelSize("inds");
        if(cdim != 4 || cdim != 3)
            throw std::runtime_error("ZSSolveLaplaceEquaOnTets: invalid simplex size");

        static dtiles_t etemp{eles.get_allocator(),{{"L",cdim*cdim}},eles.size()};
        static dtiles_t vtemp{verts.get_allocator(),{
            {"x",1},
            {"b",1},
            {"P",1},
            {"temp",1},
            {"r",1},
            {"p",1},
            {"q",1},
            {"vol",1}
        },verts.size()};
        
        etemp.resize(eles.size());
        vtemp.resize(verts.size());

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        
        // compute laplace matrix
        if(cdim == 4)
            compute_cotmatrix(cudaPol,eles,verts,"x",etemp,"L",zs::wrapv<4>{});
        else //(cdim == 3)
            compute_cotmatrix(cudaPol,eles,verts,"x",etemp,"L",zs::wrapv<3>{});
        
        // compute nodal volume
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
            vtemp("vol",vi) = 0.0;
        });
        cudaPol(zs::range(eles.size()),
            [eles = proxy<space>({},eles),vtemp = proxy<space>({},vtemp),cdim] ZS_LAMBDA(int ei) mutable {
                auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                for(size_t i = 0;i < cdim;++i)
                    atomic_add(exec_cuda,&vtemp("vol",inds[i]),(T)eles("vol",ei)/(T)cdim);
        });
        HarmonicSystem A{verts,eles};
        // compute diagonal precondiner
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
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA (int vi) mutable {
                vtemp("P",vi) = 1./zs::sqrt(vtemp("vol",vi))/vtemp("P",vi);
        });

        {
            // set the initial guess of the solution subject to boundary condition
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),tag = zs::SmallString(attr)] ZS_LAMBDA(int vi) mutable{
                    vtemp("x",vi) = verts(tag,vi);
                }
            );
            // eval the right hand side
            A.rhs(cudaPol,"x","b",vtemp);
            A.multiply(cudaPol,"x","temp",vtemp,etemp);
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    vtemp("r",vi) = vtemp("b",vi) - vtemp("temp",vi);
                });

            A.project(cudaPol,"btag",verts,"r",vtemp);
            A.precondition(cudaPol,"r","q",vtemp);
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
                A.multiply(cudaPol, "p", "temp",vtemp,etemp);
                A.project(cudaPol,"btag",verts, "temp",vtemp);

                T alpha = zTrk / dot(cudaPol, vtemp, "temp", "p");
                cudaPol(range(verts.size()), [verts = proxy<space>({}, verts),
                                    vtemp = proxy<space>({}, vtemp),
                                    alpha] ZS_LAMBDA(int vi) mutable {
                    vtemp("x", vi) += alpha * vtemp("p", vi);
                    vtemp("r", vi) -= alpha * vtemp("temp", vi);
                });
                A.precondition(cudaPol, "r", "q",vtemp);
                auto zTrkLast = zTrk;
                zTrk = dot(cudaPol, vtemp, "q", "r");
                auto beta = zTrk / zTrkLast;
                cudaPol(range(verts.size()), [vtemp = proxy<space>({}, vtemp),beta] ZS_LAMBDA(int vi) mutable {
                    vtemp("p", vi) = vtemp("q", vi) + beta * vtemp("p", vi);
                });
                residualPreconditionedNorm = std::sqrt(zTrk);
            }
            fmt::print("FINISH SOLVING PCG with cg_iter = {}\n",iter);  
        }// end cg step        


        cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts),tag = zs::SmallString(attr)] ZS_LAMBDA(int vi) mutable {
                verts(tag,vi) = vtemp("x",vi);
        });

        set_output("ZSParticles",zspars);
    }

};

ZENDEFNODE(ZSSolveBiHarmonicEquaOnTets, {
                                    {"ZSParticles"},
                                    {"ZSParticles"},
                                    {
                                        {"string","tag","T"}
                                    },
                                    {"FEM"}
});



};