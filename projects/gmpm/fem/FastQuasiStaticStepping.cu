#include "../Structures.hpp"
#include "../Utils.hpp"
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

namespace zeno {
struct FastQuasiStaticStepping : INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;

    struct FastFEMSystem {
        template <typename Pol, typename Model>
        T energy(Pol &pol, const Model &model, const zs::SmallString tag, dtiles_t& vtemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            Vector<T> res{verts.get_allocator(), 1};
            res.setVal(0);
            //   elastic potential
            pol(range(eles.size()), [verts = proxy<space>({}, verts),
                                    eles = proxy<space>({}, eles),
                                    vtemp = proxy<space>({}, vtemp),
                                    res = proxy<space>(res), tag, model = model,volf = volf] 
                                    ZS_LAMBDA (int ei) mutable {
                auto DmInv = eles.pack<3, 3>("IB", ei);
                auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
                vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                            vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
                mat3 F{};
                {
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];
                auto x3x0 = xs[3] - xs[0];
                auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                F = Ds * DmInv;
                }
                auto psi = model.psi(F);
                auto vole = eles("vol", ei);

                T gpsi = 0;
                for(int i = 0;i != 4;++i)
                    gpsi += (-volf.dot(xs[i])/4); 

                atomic_add(exec_cuda, &res[0], (T)(vole * (psi + gpsi)));
            });
        // Bone Driven Potential Energy
            T lambda = model.lam;
            T mu = model.mu;
            auto nmEmbedVerts = b_verts.size();
            if(b_bcws.size() != b_verts.size()){
                fmt::print("B_BCWS_SIZE = {}\t B_VERTS_SIZE = {}\n",b_bcws.size(),b_verts.size());
                throw std::runtime_error("B_BCWS SIZE AND B_VERTS SIZE NOT MATCH");
            }
            pol(range(nmEmbedVerts), [vtemp = proxy<space>({},vtemp),
                eles = proxy<space>({},eles),
                b_verts = proxy<space>({},b_verts),
                bcws = proxy<space>({},b_bcws),lambda,mu,tag,res = proxy<space>(res),bone_driven_weight = bone_driven_weight]
                ZS_LAMBDA(int vi) mutable {
                    auto ei = reinterpret_bits<int>(bcws("inds",vi));
                    if(ei < 0)
                        return;
                    auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                    auto w = bcws.pack<4>("w",vi);

                    auto tpos = vec3::zeros();
                    for(size_t i = 0;i != 4;++i)
                        tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
                    auto pdiff = tpos - b_verts.pack<3>("x",vi);

                    T stiffness = 2.0066 * mu + 1.0122 * lambda;
                    T bpsi = (0.5 * bcws("cnorm",vi) * stiffness * bone_driven_weight * eles("vol",ei)) * pdiff.l2NormSqr();
                    atomic_add(exec_cuda, &res[0], (T)bpsi);
            });

            return res.getVal();
        }

        template <typename Model>
        void gradient(zs::CudaExecutionPolicy& cudaPol,
                                        const Model& model,
                                        const zs::SmallString tag, 
                                        dtiles_t& vtemp,
                                        dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                        etemp = proxy<space>({}, etemp),
                                        bcws = proxy<space>({},b_bcws),
                                        b_verts = proxy<space>({},b_verts),
                                        verts = proxy<space>({}, verts),
                                        eles = proxy<space>({}, eles),tag, model, volf = volf] ZS_LAMBDA (int ei) mutable {
                auto DmInv = eles.pack<3, 3>("IB", ei);
                auto dFdX = dFdXMatrix(DmInv);
                auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
                vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                                vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
                mat3 F{};
                {
                    auto x1x0 = xs[1] - xs[0];
                    auto x2x0 = xs[2] - xs[0];
                    auto x3x0 = xs[3] - xs[0];
                    auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                    F = Ds * DmInv;
                }
                auto P = model.first_piola(F);
                auto vole = eles("vol", ei);
                auto vecP = flatten(P);
                auto dFdXT = dFdX.transpose();
                auto vf = -vole * (dFdXT * vecP);
                auto mg = volf * vole / 4;
                for (int i = 0; i != 4; ++i) {
                    auto vi = inds[i];
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_cuda, &vtemp("grad", d, vi), vf(i * 3 + d) + mg(d));
                }

            });

            if(b_bcws.size() != b_verts.size()){
                fmt::print("B_BCWS_SIZE = {}\t B_VERTS_SIZE = {}\n",b_bcws.size(),b_verts.size());
                throw std::runtime_error("B_BCWS SIZE AND B_VERTS SIZE NOT MATCH");
            }

            T stiffness = 2.0066 * model.mu + 1.0122 * model.lam;
            auto nmEmbedVerts = b_verts.size();
            cudaPol(zs::range(nmEmbedVerts),
                [bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                            eles = proxy<space>({},eles),stiffness,tag,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
                auto ei = reinterpret_bits<int>(bcws("inds",vi));
                if(ei < 0)
                    return;
                auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                auto w = bcws.pack<4>("w",vi);
                auto tpos = vec3::zeros();
                for(size_t i = 0;i != 4;++i)
                    tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
                auto pdiff = tpos - b_verts.pack<3>("x",vi);

                for(size_t i = 0;i != 4;++i){
                    auto tmp = pdiff * (-stiffness * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)); 
                    // tmp = pdiff * (-lambda * bcws("cnorm",vi) * bone_driven_weight * w[i]);
                    for(size_t d = 0;d != 3;++d)
                        atomic_add(exec_cuda,&vtemp("grad",d,inds[i]),(T)tmp[d]);
                }
            });
        }

        template <typename Model>
        void laplacian(zs::CudaExecutionPolicy& cudaPol,
                                const Model& model,
                                const zs::SmallString tag, 
                                const zs::SmallString Htag,
                                dtiles_t& vtemp,
                                dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            T stiffness = 2.0066 * model.mu + 1.0122 * model.lam;    
            cudaPol(zs::range(eles.size()),
                [vtemp = proxy<space>({}, vtemp),etemp = proxy<space>({}, etemp),
                    bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),
                    verts = proxy<space>({},verts),eles = proxy<space>({},eles),tag,
                    Htag,stiffness,bone_driven_weight = bone_driven_weight]
                        ZS_LAMBDA(int ei) mutable {
                auto DmInv = eles.pack<3, 3>("IB", ei);
                auto dFdX = dFdXMatrix(DmInv);
                auto vol = eles("vol",ei);
                etemp.pack<12,12>(Htag,ei) = stiffness * vol * dFdX.transpose() * dFdX;            
            });   

            cudaPol(zs::range(b_bcws.size()),
                    [bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    eles = proxy<space>({},eles),stiffness,tag,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
                auto ei = reinterpret_bits<int>(bcws("inds",vi));
                if(ei < 0)
                    return;
                auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                auto w = bcws.pack<4>("w",vi);

                for(int i = 0;i != 4;++i)
                    for(int j = 0;j != 4;++j){
                        T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi) * eles("vol",ei);
                        for(int d = 0;d != 3;++d){
                            atomic_add(exec_cuda,&etemp("He",(i * 3 + d) * 12 + j * 3 + d,ei),alpha);
                        }
                    }

            });                                         
        }

        template <typename Model>
        void hessian(zs::CudaExecutionPolicy& cudaPol,
                                                const Model& model,
                                                const zs::SmallString xTag,
                                                const zs::SmallString HTag, 
                                                dtiles_t& vtemp,
                                                dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            // fmt::print("check here 0");
            cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                            etemp = proxy<space>({}, etemp),
                                            bcws = proxy<space>({},b_bcws),
                                            b_verts = proxy<space>({},b_verts),
                                            verts = proxy<space>({}, verts),
                                            eles = proxy<space>({}, eles),tag = xTag,HTag, model, volf = volf] ZS_LAMBDA (int ei) mutable {
                auto DmInv = eles.pack<3, 3>("IB", ei);
                auto dFdX = dFdXMatrix(DmInv);
                auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
                vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                                vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
                mat3 F{};
                {
                    auto x1x0 = xs[1] - xs[0];
                    auto x2x0 = xs[2] - xs[0];
                    auto x3x0 = xs[3] - xs[0];
                    auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                    F = Ds * DmInv;
                }
                auto vole = eles("vol", ei);
                auto dFdXT = dFdX.transpose();

                auto Hq = model.first_piola_derivative(F, true_c);
                auto H = dFdXT * Hq * dFdX * vole;

                etemp.tuple<12 * 12>(HTag, ei) = H;

            });
            T stiffness = 2.0066 * model.mu + 1.0122 * model.lam;   
            cudaPol(zs::range(b_bcws.size()),
                    [bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    eles = proxy<space>({},eles),stiffness,HTag,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
                auto ei = reinterpret_bits<int>(bcws("inds",vi));
                if(ei < 0)
                    return;
                auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                auto w = bcws.pack<4>("w",vi);

                for(int i = 0;i != 4;++i)
                    for(int j = 0;j != 4;++j){
                        T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi) * eles("vol",ei);
                        for(int d = 0;d != 3;++d){
                            atomic_add(exec_cuda,&etemp(HTag,(i * 3 + d) * 12 + j * 3 + d,ei),alpha);
                        }
                    }

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
                vtemp.tuple<3>(dstTag, vi) =
                    vtemp.pack<3, 3>("P", vi) * vtemp.pack<3>(srcTag, vi);
                // vtemp.tuple<3>(dstTag, vi) = vtemp.pack<3>(srcTag, vi);
            });
        }

        template <typename Pol>
        void multiply(Pol &pol, const zs::SmallString dxTag,
                    const zs::SmallString bTag,
                    const zs::SmallString HTag,
                    dtiles_t& vtemp,
                    const dtiles_t& etemp) {
            using namespace zs;
            constexpr execspace_e space = execspace_e::cuda;
            constexpr auto execTag = wrapv<space>{};
            const auto numVerts = verts.size();
            const auto numEles = eles.size();
            // dx -> b
            pol(range(numVerts),
                [execTag, vtemp = proxy<space>({}, vtemp), bTag] ZS_LAMBDA(
                    int vi) mutable { vtemp.tuple<3>(bTag, vi) = vec3::zeros(); });
            // elastic energy
            pol(range(numEles), [execTag, etemp = proxy<space>({}, etemp),
                                vtemp = proxy<space>({}, vtemp),
                                eles = proxy<space>({}, eles), dxTag, bTag, HTag] ZS_LAMBDA(int ei) mutable {
                constexpr int dim = 3;
                constexpr auto dimp1 = dim + 1;
                auto inds = eles.pack<dimp1>("inds", ei).reinterpret_bits<int>();
                zs::vec<T, dimp1 * dim> temp{};
                for (int vi = 0; vi != dimp1; ++vi)
                for (int d = 0; d != dim; ++d) {
                    temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
                }
                auto He = etemp.pack<dim * dimp1, dim * dimp1>(HTag, ei);

                temp = He * temp;

                for (int vi = 0; vi != dimp1; ++vi)
                for (int d = 0; d != dim; ++d) {
                    atomic_add(execTag, &vtemp(bTag, d, inds[vi]), temp[vi * dim + d]);
                }
            });
        }

        FastFEMSystem(const tiles_t &verts, const tiles_t &eles, const tiles_t &b_bcws, const tiles_t& b_verts,T bone_driven_weight,vec3 volf)
            : verts{verts}, eles{eles}, b_bcws{b_bcws}, b_verts{b_verts}, bone_driven_weight{bone_driven_weight},volf{volf}{}

        const tiles_t &verts;
        const tiles_t &eles;
        const tiles_t &b_bcws;  // the barycentric interpolation of embeded bones 
        const tiles_t &b_verts; // the position of embeded bones

        T bone_driven_weight;
        vec3 volf;

    };

    template<typename Equation,typename Model>
    constexpr void backtracking_line_search(zs::CudaExecutionPolicy &cudaPol,Equation& A,Model& models,int max_line_search,T armijo,
            const zs::SmallString& dtag,const zs::SmallString& gtag,const zs::SmallString& xtag,T init_step,dtiles_t& vtemp) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        T dg = dot(cudaPol,vtemp,gtag,dtag);
        T E0;
        match([&](auto &elasticModel) {
            E0 = A.energy(cudaPol, elasticModel, xtag,vtemp);
        })(models.getElasticModel());
        T E{E0};
        int line_search = 0;
        std::vector<T> armijo_buffer(max_line_search);
        T step = init_step;
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),xtag,dtag,step] ZS_LAMBDA(int vi) mutable {
                vtemp.pack<3>(xtag,vi) += step * vtemp.pack<3>(dtag,vi);
            });

        do {
            match([&](auto &elasticModel) {
            E = A.energy(cudaPol,elasticModel,xtag,vtemp);
            })(models.getElasticModel());
            // fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
            // fmt::print("Armijo : {} < {}\n",(E - E0)/alpha,dg);
            armijo_buffer[line_search] = (E - E0)/step;
            // test Armojo condition
            if(((double)E - (double)E0) < (double)armijo * (double)dg * (double)step)
                break;
            step /= 2;
            cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),step,xtag,dtag] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple<3>(xtag, vi) = vtemp.pack<3>(xtag, vi) - step * vtemp.pack<3>(dtag, vi);
            });
            ++line_search;
        } while (line_search < max_line_search);
        // return line_search;
    }

    template<typename Equation,typename Model>
    constexpr void solve_equation_using_pcg(zs::CudaExecutionPolicy &cudaPol,Equation& A,Model& models,const zs::SmallString& btag,const zs::SmallString& xtag,const zs::SmallString& Ptag,dtiles_t& vtemp,
            zs::SmallString Htag,dtiles_t& etemp) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

    }

    static T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> ret{res.get_allocator(), 1};
        auto sid = cudaPol.getStreamid();
        auto procid = cudaPol.getProcid();
        auto &context = Cuda::context(procid);
        auto stream = (cudaStream_t)context.streamSpare(sid);
        std::size_t temp_bytes = 0;
        cub::DeviceReduce::Reduce(nullptr, temp_bytes, res.data(), ret.data(),
                                res.size(), std::plus<T>{}, (T)0, stream);
        Vector<std::max_align_t> temp{res.get_allocator(),
                                    temp_bytes / sizeof(std::max_align_t) + 1};
        cub::DeviceReduce::Reduce(temp.data(), temp_bytes, res.data(), ret.data(),
                                res.size(), std::plus<T>{}, (T)0, stream);
        context.syncStreamSpare(sid);
        return (T)ret.getVal();
    }
    template<int pack_dim = 3>
    T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
            const zs::SmallString tag0, const zs::SmallString tag1,int offset0 = 0,int offset1 = 0) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vertData.get_allocator(), vertData.size()},ret{vertData.get_allocator(),1};
        cudaPol(range(vertData.size()),
            [data = proxy<space>({},vertData),res = proxy<space>(res),tag0,tag1,offset0,offset1] ZS_LAMBDA(int pi) mutable {
                res[pi] = (T)0.;
                for(int i = 0;i < pack_dim;++i)
                    res[pi] += data(tag0,offset0*pack_dim + i,pi) * data(tag1,offset1*pack_dim + i,pi);
            });
        //zs::reduce(cudaPol,std::begin(res),std:end(res),std::begin(ret), (T)0);
        //return (T)ret.getVal();
        return reduce(cudaPol, res);
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
        auto zstets = get_input<ZenoParticles>("ZSParticles");
        auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec<3,T>>();
        auto zsbones = get_input<ZenoParticles>("driven_bones");            // driven bones

        auto armijo = get_param<float>("armijo");
        auto curvature = get_param<float>("wolfe");
        auto cg_res = get_param<float>("cg_res");                           // cg_res for inner loop of quasi-newton solver
        auto btl_res = get_param<float>("btl_res");                         // a termination criterion for line search
        
        auto epsilon = get_param<float>("epsilon");
        auto rel_epsilon = get_param<float>("rel_epsilon");
        
        auto models = zstets->getModel();           
        auto& verts = zstets->getParticles();
        auto& eles = zstets->getQuadraturePoints();

        auto tag = get_param<std::string>("driven_tag");                    // tag channel where the bones are binded
        auto bone_driven_weight = get_param<float>("bone_driven_weight");   // the weight of bone-driven potential
        auto nm_newton_iters = get_param<int>("nm_newton_iters");
        auto quasi_newton_window_size = get_param<int>("window_size");

        auto volf = vec3::from_array(gravity * models.density);

        static dtiles_t vtemp{verts.get_allocator(),
            {
                {"grad", 3},
                {"gradp",3},
                {"P", 9},
                {"dir", 3},
                {"xn", 3},
                {"xn0", 3},
                {"xp",3},
                {"temp", 3},
                {"r", 3},
                {"p", 3},
                {"q", 3},
                {"fx", quasi_newton_window_size},
                {"s", 3 * quasi_newton_window_size},
                {"y", 3 * quasi_newton_window_size}
            },verts.size()};
        // buffer storage for laplace matrix
        static dtiles_t etemp{eles.get_allocator(),{{"L", 12 * 12},{"H",12 * 12}},eles.size()};  
        vtemp.resize(verts.size());
        etemp.resize(eles.size());
        FastFEMSystem A{verts,eles,(*zstets)[tag],zsbones->getParticles(),bone_driven_weight,volf};

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();   
        
        match([&](auto &elasticModel){
            A.laplacian(cudaPol,elasticModel,"xn","L",vtemp,etemp);
        })(models.getElasticModel());

        // build preconditioner for fast cg convergence
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp),
                verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                    vtemp.tuple<9>("P", vi) = mat3::zeros();
        });        
        cudaPol(zs::range(eles.size()),
            [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles)]
                ZS_LAMBDA (int ei) mutable {
                constexpr int dim = 3;
                constexpr auto dimp1 = dim + 1;
                auto inds = 
                    eles.template pack<dimp1>("inds",ei).template reinterpret_bits<int>();
                auto He = etemp.pack<dim * dimp1,dim * dimp1>("L",ei);

                for (int vi = 0; vi != dimp1; ++vi) {
                #if 1
                    for (int i = 0; i != dim; ++i)
                    for (int j = i; j != dim; ++j){ 
                        atomic_add(exec_cuda, &vtemp("P", i * dim + j, inds[vi]),He(vi * dim + i, vi * dim + j));
                    //   atomic_add(exec_cuda, &vtemp("P", j * dim + i, inds[vi]),He(vi * dim + i, vi * dim + j));
                    }
                #else
                    for (int j = 0; j != dim; ++j) {
                        atomic_add(exec_cuda, &vtemp("P", j * dim + j, inds[vi]),
                                He(vi * dim + j, vi * dim + j));
                    }
                #endif
                }
        });
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp),
                verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                    constexpr int dim = 3;
                    for (int i = 0; i != dim; ++i)
                        for (int j = i+1; j != dim; ++j){ 
                            vtemp("P", j * dim + i, vi) = vtemp("P", i * dim + j, vi);
                    //   atomic_add(exec_cuda, &vtemp("P", j * dim + i, inds[vi]),He(vi * dim + i, vi * dim + j));
                    }
        });

        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp)] __device__(int vi) mutable {
                    // we need to use double-precision inverse here, when the P matrix is nearly singular or has very large coeffs
                    vtemp.tuple<9>("P",vi) = inverse(vtemp.pack<3,3>("P",vi).cast<double>());
        });

        // solve the problem using quasi-newton solver
        T fx;
        match([&](auto &elasticModel){
            fx = A.energy(cudaPol,elasticModel,"xn",vtemp);
        })(models.getElasticModel());

        match([&](auto &elasticModel){
            A.gradient(cudaPol,elasticModel,"xn",vtemp,etemp);
        })(models.getElasticModel());

        T gn = std::sqrt(dot(cudaPol,vtemp,"grad","grad"));
        T xn = std::sqrt(dot(cudaPol,vtemp,"xn","xn"));

        if(gn > epsilon && gn > xn * rel_epsilon) {
            int k = 0;
            T step = 1. / gn;
            // solve for cg newton dir might be better?
            cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    vtemp.tuple<3>("dir",vi) = -vtemp.pack<3,3>("P",vi) * vtemp.pack<3>("grad",vi);
            }); 

            int nm_corr = 0;
            std::vector<T> m_alpha(quasi_newton_window_size);
            std::vector<T> m_ys(quasi_newton_window_size);

            while(k < nm_newton_iters) {
                // copy the x and grad
                cudaPol(zs::range(vtemp.size()),
                    [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                        vtemp.tuple<3>("xp",vi) = vtemp.pack<3>("xn",vi);
                        vtemp.tuple<3>("gradp",vi) = vtemp.pack<3>("grad",vi);
                });
                // do line search along the searching direction using armijo condition.../ consider wolfe only when the spd is not enforced
                backtracking_line_search(cudaPol,A,models,10,armijo,"dir","grad","xn",step,vtemp);
                T gn = std::sqrt(dot(cudaPol,vtemp,"grad","grad"));
                T xn = std::sqrt(dot(cudaPol,vtemp,"xn","xn"));
                // gradient termination criterion test
                if(gn <= epsilon || gn <= epsilon * xn)
                    break;
                // add correction to hessian approximation
                cudaPol(zs::range(vtemp.size()),
                    [vtemp = proxy<space>({},vtemp),ws = quasi_newton_window_size,k] ZS_LAMBDA(int vi) mutable {
                        for(int i = 0;i != 3;++i){
                            vtemp("s",(k % ws)*3 + i,vi) = vtemp("xn",i,vi) - vtemp("xp",i,vi);
                            vtemp("y",(k % ws)*3 + i,vi) = vtemp("grad",i,vi) - vtemp("gradp",i,vi);
                            // vtemp.tuple<3>("s",k % ws,vi) = vtemp.pack<3>("xn",vi) - vtemp.pack<3>("xp",vi);
                            // vtemp.tuple<3>("y",k % ws,vi) = vtemp.pack<3>("grad",vi) - vtemp.pack<3>("gradp",vi);
                        }
                });
                // some problem use atomic add
                m_ys[k % quasi_newton_window_size] = dot(cudaPol,vtemp,"s","y",k % quasi_newton_window_size,k % quasi_newton_window_size);
                ++nm_corr;
                // apply Hv 
                // recursively compute d = -H*g
                {
                    // Loop1
                    // m_dir = -m_g
                    cudaPol(zs::range(vtemp.size()),
                        [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                            vtemp.tuple<3>("temp",vi) = -vtemp.pack<3>("grad",vi);
                        });
                    // point to the most recent correction buffer
                    int j = (k+1) % quasi_newton_window_size;
                    for(int i = 0;i < nm_corr;++i){
                        // moving backward
                        j = (j + quasi_newton_window_size - 1) % quasi_newton_window_size;
                        m_alpha[j] = dot(cudaPol,vtemp,"s","temp",k % quasi_newton_window_size) / m_ys[j];
                        cudaPol(zs::range(vtemp.size()),
                            [vtemp = proxy<space>({},vtemp),alpha = m_alpha[j],ws = quasi_newton_window_size,k]
                                ZS_LAMBDA(int vi) mutable {
                                    for(int i = 0;i != 3;++i)
                                        vtemp("temp",i,vi) -= alpha * vtemp("y",(k % ws)*3 + i,vi);
                        });
                    }

                    // solve laplace equation using cg, do not have to be that accurate?
                    solve_equation_using_pcg(cudaPol,A,models,"temp","dir","P",vtemp,"L",etemp);

                    // Loop 2
                    for(int i = 0;i < nm_corr;++i){
                        T beta = dot(cudaPol,vtemp,"y","dir",j) / m_ys[j];
                        cudaPol(zs::range(vtemp.size()),
                            [vtemp = proxy<space>({},vtemp),offset = k % quasi_newton_window_size,alpha = m_alpha[j],beta,j] ZS_LAMBDA(int vi) mutable{
                                for(int i = 0;i != 3;++i)
                                    vtemp("dir",i,vi) += (alpha - beta) * vtemp("s",j*3 + i,vi);
                            });
                        j = (j+1) % quasi_newton_window_size;
                    }
                }

                step = 1.;
                ++k;
            }
        }
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                verts.pack<3>("x",vi) = vtemp.pack<3>("xn",vi);
        });
        set_output("ZSParticles", zstets);
    }
};


ZENDEFNODE(FastQuasiStaticStepping, {{"ZSParticles","driven_bones","gravity"},
                                  {"ZSParticles"},
                                  {{"float","armijo","0.1"},{"float","wolfe","0.9"},
                                    {"float","cg_res","0.1"},{"float","btl_res","0.0001"},{"float","epsilon","1e-5"},
                                    {"float","rel_epsilon","1e-3"},
                                    {"string","driven_tag","bone_bw"},{"float","bone_driven_weight","0.0"},
                                    {"int","nm_newton_iters","20"},{"int","window_size","8"}
                                    },
                                  {"FEM"}});

};