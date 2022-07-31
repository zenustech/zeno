#include "../Structures.hpp"
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
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "../geometry/linear_system/mfcg.hpp"

namespace zeno {


struct FleshQuasiStaticStepping : INode {
  using T = float;
  using dtiles_t = zs::TileVector<T,32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<T, 3>;
  using mat3 = zs::vec<T, 3, 3>;
  struct FEMSystem {

    constexpr auto dFAdF(const mat3& A) {
        zs::vec<T,9,9> M{};
        M(0,0) = M(1,1) = M(2,2) = A(0,0);
        M(3,0) = M(4,1) = M(5,2) = A(0,1);
        M(6,0) = M(7,1) = M(8,2) = A(0,2);

        M(0,3) = M(1,4) = M(2,5) = A(1,0);
        M(3,3) = M(4,4) = M(5,5) = A(1,1);
        M(6,3) = M(7,4) = M(8,5) = A(1,2);

        M(0,6) = M(1,7) = M(2,8) = A(2,0);
        M(3,6) = M(4,7) = M(5,8) = A(2,1);
        M(6,6) = M(7,7) = M(8,8) = A(2,2);

        return M;        
    }


    template <typename Pol, typename Model>
    T energy(Pol &pol, const Model &model, const zs::SmallString tag, dtiles_t& vtemp,dtiles_t& etemp) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
    //   elastic potential
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               etemp = proxy<space>({},etemp),
                               res = proxy<space>(res), tag, model = model,volf = volf] 
                               ZS_LAMBDA (int ei) mutable {
        auto DmInv = eles.template pack<3, 3>("IB", ei);
        auto inds = eles.template pack<4>("inds", ei).template reinterpret_bits<int>();
        vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                      vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
        mat3 FAct{};
        {
          auto x1x0 = xs[1] - xs[0];
          auto x2x0 = xs[2] - xs[0];
          auto x3x0 = xs[3] - xs[0];
          auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                         x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
          FAct = Ds * DmInv;

          FAct = FAct * etemp.template pack<3,3>("ActInv",ei);

        //   if(ei == 0) {
        //     printf("FAct in energy : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
        //         (float)FAct(0,0),(float)FAct(0,1),(float)FAct(0,2),
        //         (float)FAct(1,0),(float)FAct(1,1),(float)FAct(1,2),
        //         (float)FAct(2,0),(float)FAct(2,1),(float)FAct(2,2));
        //   }
        }

        auto psi = model.psi(FAct);
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
              // if(eles("vol",ei) < 0)
              //     printf("WARNING INVERT TET DETECTED<%d> %f\n",ei,(float)eles("vol",ei));
              T bpsi = (0.5 * bcws("cnorm",vi) * stiffness * bone_driven_weight * eles("vol",ei)) * pdiff.l2NormSqr();
                // bpsi = (0.5 * bcws("cnorm",vi) * lambda * bone_driven_weight) * pdiff.dot(pdiff);
                // the cnorm here should be the allocated volume of point in embeded tet 
              atomic_add(exec_cuda, &res[0], (T)bpsi);
      });

      return res.getVal();
    }

    template <typename Model>
    void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                                            const Model& model,
                                            const zs::SmallString tag, 
                                            dtiles_t& vtemp,
                                            dtiles_t& etemp) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // fmt::print("check here 0");
        PCG::fill<3>(cudaPol,vtemp,"grad",zs::vec<T,3>::zeros());
        PCG::fill<144>(cudaPol,etemp,"He",zs::vec<T,144>::zeros());
        cudaPol(zs::range(eles.size()), [this,
                                        vtemp = proxy<space>({}, vtemp),
                                        etemp = proxy<space>({}, etemp),
                                        bcws = proxy<space>({},b_bcws),
                                        b_verts = proxy<space>({},b_verts),
                                        verts = proxy<space>({}, verts),
                                        eles = proxy<space>({}, eles),tag, model, volf = volf] ZS_LAMBDA (int ei) mutable {
            auto DmInv = eles.template pack<3, 3>("IB", ei);
            auto dFdX = dFdXMatrix(DmInv);
            auto inds = eles.template pack<4>("inds", ei).template reinterpret_bits<int>();
            vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                            vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
            mat3 FAct{};
            {
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];
                auto x3x0 = xs[3] - xs[0];
                auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                            x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                FAct = Ds * DmInv;

                FAct = FAct * etemp.template pack<3,3>("ActInv",ei);

                // if(ei == 0) {
                //     printf("FAct in gH : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
                //         (float)FAct(0,0),(float)FAct(0,1),(float)FAct(0,2),
                //         (float)FAct(1,0),(float)FAct(1,1),(float)FAct(1,2),
                //         (float)FAct(2,0),(float)FAct(2,1),(float)FAct(2,2));
                    
                //     auto Act =  etemp.template pack<3,3>("ActInv",ei);

                //     printf("Act in gH : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
                //         (float)Act(0,0),(float)Act(0,1),(float)Act(0,2),
                //         (float)Act(1,0),(float)Act(1,1),(float)Act(1,2),
                //         (float)Act(2,0),(float)Act(2,1),(float)Act(2,2));                        
                // }

                // auto ActInv_check = etemp.template pack<3,3>("ActInv",ei);
                // for(int i = 0;i != 3;++i)
                //     ActInv_check(i,i) -= 1.0;
                // if(ActInv_check.norm() > 1){
                //     auto ActInv = etemp.template pack<3,3>("ActInv",ei);
                //     printf("wierd ActInv<%d> in gH : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",ei,
                //         (float)ActInv(0,0),(float)ActInv(0,1),(float)ActInv(0,2),
                //         (float)ActInv(1,0),(float)ActInv(1,1),(float)ActInv(1,2),
                //         (float)ActInv(2,0),(float)ActInv(2,1),(float)ActInv(2,2));  
                // }

            }

            auto dFActdF = dFAdF(etemp.template pack<3,3>("ActInv",ei));

            auto P = model.first_piola(FAct);
            auto vole = eles("vol", ei);
            auto vecP = flatten(P);
            vecP = dFActdF.transpose() * vecP;
            auto dFdXT = dFdX.transpose();
            auto vf = -vole * (dFdXT * vecP);

            auto mg = volf * vole / 4;
            for (int i = 0; i != 4; ++i) {
                auto vi = inds[i];
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_cuda, &vtemp("grad", d, vi), vf(i * 3 + d) + mg(d));
            }

            auto Hq = model.first_piola_derivative(FAct, true_c);
            auto dFdAct_dFdX = dFActdF * dFdX; 
            // dFdAct_dFdX = dFdX; 
            auto H = dFdAct_dFdX.transpose() * Hq * dFdAct_dFdX * vole;

            etemp.tuple<12 * 12>("He", ei) = H;


            // auto Hn = H.norm();
            // if(isnan(Hn)){
            //     auto Hqn = Hq.norm();
            //     auto dFdXn = dFdAct_dFdX.norm();
            //     printf("elm<%d>_Hn : %f %f %f\n",ei,(float)Hn,(float)dFdXn,(float)Hqn);
            //     printf("FAct<%d> in gH : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",ei,
            //         (float)FAct(0,0),(float)FAct(0,1),(float)FAct(0,2),
            //         (float)FAct(1,0),(float)FAct(1,1),(float)FAct(1,2),
            //         (float)FAct(2,0),(float)FAct(2,1),(float)FAct(2,2));

            //     auto Act = etemp.template pack<3,3>("ActInv",ei);
            //     printf("Act<%d> in gH : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",ei,
            //         (float)Act(0,0),(float)Act(0,1),(float)Act(0,2),
            //         (float)Act(1,0),(float)Act(1,1),(float)Act(1,2),
            //         (float)Act(2,0),(float)Act(2,1),(float)Act(2,2));                
            // }

        });


        // fmt::print("check here 1\n");
        T lambda = model.lam;
        T mu = model.mu;
        if(b_bcws.size() != b_verts.size()){
            fmt::print("B_BCWS_SIZE = {}\t B_VERTS_SIZE = {}\n",b_bcws.size(),b_verts.size());
            throw std::runtime_error("B_BCWS SIZE AND B_VERTS SIZE NOT MATCH");
        }

        // fmt::print("check here 2\n");

        auto nmEmbedVerts = b_verts.size();
        cudaPol(zs::range(nmEmbedVerts),
            [bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                eles = proxy<space>({},eles),lambda,mu,tag,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
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

                    for(size_t i = 0;i != 4;++i){
                        auto tmp = pdiff * (-stiffness * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)); 
                        // tmp = pdiff * (-lambda * bcws("cnorm",vi) * bone_driven_weight * w[i]);
                        for(size_t d = 0;d != 3;++d)
                            atomic_add(exec_cuda,&vtemp("grad",d,inds[i]),(T)tmp[d]);
                    }
                    for(int i = 0;i != 4;++i)
                        for(int j = 0;j != 4;++j){
                            T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi) * eles("vol",ei);
                            // alpha = lambda * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi);
                            // if(ei == 11221)
                            //   if(i == 3 && j == 3)
                            //     printf("alpha : %f\n",alpha);
                            for(int d = 0;d != 3;++d){
                                // etemp("He",(i * 3 + d) * 12 + j * 3 + d,ei) += alpha;
                                // if(isnan(alpha)){
                                //     printf("nan alpha<%d,%d,%d> %f %f %f %f %f\n",vi,i,j,(float)lambda,(float)bone_driven_weight,(float)w[i],(float)w[j],(float)bcws("cnorm",vi));
                                // }
                                atomic_add(exec_cuda,&etemp("He",(i * 3 + d) * 12 + j * 3 + d,ei),alpha);
                            }
                        }

        });

    }

    FEMSystem(const tiles_t &verts, const tiles_t &eles, const tiles_t &b_bcws, const tiles_t& b_verts,T bone_driven_weight,vec3 volf)
        : verts{verts}, eles{eles}, b_bcws{b_bcws}, b_verts{b_verts}, bone_driven_weight{bone_driven_weight},volf{volf}{}

    const tiles_t &verts;
    const tiles_t &eles;
    const tiles_t &b_bcws;  // the barycentric interpolation of embeded bones 
    const tiles_t &b_verts; // the position of embeded bones

    T bone_driven_weight;
    vec3 volf;
  };

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec<3,T>>();
    auto armijo = get_param<float>("armijo");
    auto curvature = get_param<float>("wolfe");
    auto cg_res = get_param<float>("cg_res");
    auto btl_res = get_param<float>("btl_res");
    auto models = zstets->getModel();
    auto& verts = zstets->getParticles();
    auto& eles = zstets->getQuadraturePoints();
    auto zsbones = get_input<ZenoParticles>("driven_bones");
    auto tag = get_param<std::string>("driven_tag");
    auto muscle_id_tag = get_param<std::string>("muscle_id_tag");
    auto bone_driven_weight = get_param<float>("bone_driven_weight");
    auto newton_res = get_param<float>("newton_res");

    auto volf = vec3::from_array(gravity * models.density);

    auto nm_acts = get_input<zeno::ListObject>("Acts")->arr.size();
    // fmt::print("number of activations : {}\n",nm_acts);
    auto act_ = get_input<zeno::ListObject>("Acts")->getLiterial<float>();
    // initialize on host qs[i] = qs_[i]->get<zeno::vec4f>();

    constexpr auto host_space = zs::execspace_e::openmp;
    auto ompExec = zs::omp_exec();
    auto act_buffer = dtiles_t{{{"act",1}},nm_acts,zs::memsrc_e::host};
    ompExec(range(act_buffer.size()),
        [act_buffer = proxy<host_space>({},act_buffer),act_] (int i) mutable{
            act_buffer("act",i) = act_[i];
            // fmt::print("act<{}> : {}\n",i,act_buffer("act",i));
    });
    act_buffer = act_buffer.clone({zs::memsrc_e::device, 0});

    if((!eles.hasProperty(muscle_id_tag)) || (eles.getChannelSize(muscle_id_tag) != 1)){
        fmt::print("the quadrature has no muscle id tag : {} {}\n",muscle_id_tag,eles.getChannelSize(muscle_id_tag));
        throw std::runtime_error("the quadrature has no muscle id tag");
    }

    static dtiles_t vtemp{verts.get_allocator(),
                          {{"grad", 3},
                           {"P", 9},
                           {"bou_tag",1},
                           {"dir", 3},
                           {"xn", 3},
                           {"xn0", 3},
                           {"temp", 3},
                           {"r", 3},
                           {"p", 3},
                           {"q", 3}},
                          verts.size()};
    static dtiles_t etemp{eles.get_allocator(), {{"He", 12 * 12},{"inds",4},{"ActInv",3*3}}, eles.size()};
    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    FEMSystem A{verts,eles,(*zstets)[tag],zsbones->getParticles(),bone_driven_weight,volf};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    PCG::copy<4>(cudaPol,eles,"inds",etemp,"inds");


    if(!eles.hasProperty("fiber")){
        fmt::print("The input flesh should have fiber orientations\n");
        throw std::runtime_error("The input flesh should have fiber orientations");
    }

    if(eles.getChannelSize("fiber") != 3){
        fmt::print("The input fiber  has wrong channel size\n");
        throw std::runtime_error("The input fiber has wrong channel size");
    }

    // apply muscle activation
    cudaPol(range(etemp.size()),
        [eles = proxy<space>({},eles),etemp = proxy<space>({},etemp),act_buffer = proxy<space>({},act_buffer),muscle_id_tag = SmallString(muscle_id_tag)] ZS_LAMBDA(int ei) mutable {
            auto act = eles.template pack<3>("act",ei);
            auto fiber = eles.template pack<3>("fiber",ei);
            
            auto nfiber = fiber.norm();
            auto ID = eles(muscle_id_tag,ei);
            if(nfiber < 0.5 || ID < -1e-6){ // if there is no local fiber orientaion, use the default act and fiber
                fiber = vec3{1.0,0.0,0.0};
                act = vec3{1.0,1.0,1.0};
            }else{
                // a test
                int id = (int)ID;
                float a = 1. - act_buffer("act",id);
                act = vec3{1,zs::sqrt(1./a),zs::sqrt(1./a)};
                fiber /= nfiber;// in case there is some floating-point error

                // printf("use act[%d] : %f\n",id,(float)a);
            }

            vec3 dir[3];
            dir[0] = fiber;
            auto tmp = vec3{1.0,0.0,0.0};
            dir[1] = dir[0].cross(tmp);
            if(dir[1].length() < 1e-3) {
                tmp = vec3{0.0,1.0,0.0};
                dir[1] = dir[0].cross(tmp);
            }

            dir[1] = dir[1] / dir[1].length();
            dir[2] = dir[0].cross(dir[1]);

            auto R = mat3{};
            for(int i = 0;i < 3;++i)
                for(int j = 0;j < 3;++j)
                    R(i,j) = dir[j][i];

            auto Act = mat3::zeros();
            Act(0,0) = act[0];
            Act(1,1) = act[1];
            Act(2,2) = act[2];

            Act = R * Act * R.transpose();

            // if(ei == 0) {
            //     printf("Act : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
            //         (float)Act(0,0),(float)Act(0,1),(float)Act(0,2),
            //         (float)Act(1,0),(float)Act(1,1),(float)Act(1,2),
            //         (float)Act(2,0),(float)Act(2,1),(float)Act(2,2));                        
            // }


            etemp.template tuple<9>("ActInv",ei) = zs::inverse(Act);

            // if(ei == 0) {
            //     Act = etemp.template pack<3,3>("ActInv",ei);
            //     printf("Act : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
            //         (float)Act(0,0),(float)Act(0,1),(float)Act(0,2),
            //         (float)Act(1,0),(float)Act(1,1),(float)Act(1,2),
            //         (float)Act(2,0),(float)Act(2,1),(float)Act(2,2));  

            //     // auto dFActdF = dFAdF(eles.template pack<3,3>("ActInv",ei));
            //     // printf("dFActdF : \n%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\")

            // }

    });

    // setup initial guess
    PCG::copy<3>(cudaPol,verts,verts.hasProperty("init_x") ? "init_x" : "x",vtemp,"xn");    
    PCG::fill<1>(cudaPol,vtemp,"bou_tag",zs::vec<T,1>::zeros());

    for(int newtonIter = 0;newtonIter != 1000;++newtonIter){
      match([&](auto &elasticModel) {
        A.computeGradientAndHessian(cudaPol, elasticModel,"xn",vtemp,etemp);
      })(models.getElasticModel());

    // auto Hn = PCG::dot<144>(cudaPol,etemp,"He","He");
    // fmt::print("Hn : {}\n",(float)Hn);    

    // break;

    //  Prepare Preconditioning
      PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"He",etemp,"P",vtemp);

      // if the grad is too small, return the result
      // Solve equation using PCG
      PCG::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());
      PCG::pcg_with_fixed_sol_solve<3,4>(cudaPol,vtemp,etemp,"dir","bou_tag","grad","P","inds","He",cg_res,1000,50);
      PCG::project<3>(cudaPol,vtemp,"dir","bou_tag");
      PCG::project<3>(cudaPol,vtemp,"grad","bou_tag");
      T res = PCG::inf_norm<3>(cudaPol, vtemp, "dir");// this norm is independent of descriterization

      if (res < newton_res) {
        fmt::print("\t# newton optimizer reach desired resolution in {} iters with residual {}\n",
                   newtonIter, res);
        break;
      }
      T dg = PCG::dot<3>(cudaPol,vtemp,"grad","dir");
      if(fabs(dg) < btl_res){
        fmt::print("\t# newton optimizer reach stagnation point in {} iters with residual {}\n",
        newtonIter, res);
        break;
      }
      if(dg < 0){
          T gradn = std::sqrt(PCG::dot<3>(cudaPol,vtemp,"grad","grad"));
          T dirn = std::sqrt(PCG::dot<3>(cudaPol,vtemp,"dir","dir"));
          fmt::print("invalid dg = {} grad = {} dir = {}\n",dg,gradn,dirn);
          throw std::runtime_error("INVALID DESCENT DIRECTION");
      }
      T alpha = 1.;
      PCG::copy<3>(cudaPol,vtemp,"xn",vtemp,"xn0");
      T E0;
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel, "xn0",vtemp,etemp);
      })(models.getElasticModel());

      dg = -dg;

      T E{E0};
    //   Backtracking Linesearch
      int max_line_search = 10;
      int line_search = 0;
      std::vector<T> armijo_buffer(max_line_search);
      do {
        PCG::add<3>(cudaPol,vtemp,"xn0",(T)1.0,"dir",alpha,"xn");
        match([&](auto &elasticModel) {
          E = A.energy(cudaPol, elasticModel, "xn",vtemp,etemp);
        })(models.getElasticModel());
        armijo_buffer[line_search] = (E - E0)/alpha;
        // test Armojo condition
        if (((double)E - (double)E0) < (double)armijo * (double)dg * (double)alpha)
          break;
        alpha /= 2;
        ++line_search;
      } while (line_search < max_line_search);
      if(line_search == max_line_search){
          fmt::print("LINE_SEARCH_EXCEED: %f\n",dg);
          for(size_t i = 0;i != max_line_search;++i)
            fmt::print("AB[{}]\t = {} dg = {}\n",i,armijo_buffer[i],dg);
      }

      cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                        alpha] __device__(int i) mutable {
        vtemp.tuple<3>("xn", i) =
            vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
      });
    
    }

    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts)] __device__(int vi) mutable {
              auto newX = vtemp.pack<3>("xn", vi);
              verts.tuple<3>("x", vi) = newX;
            });


    set_output("ZSParticles", zstets);
  }
};

ZENDEFNODE(FleshQuasiStaticStepping, {{"ZSParticles","driven_bones","gravity","Acts"},
                                  {"ZSParticles"},
                                  {{"float","armijo","0.1"},{"float","wolfe","0.9"},
                                    {"float","cg_res","0.1"},{"float","btl_res","0.0001"},{"float","newton_res","0.001"},
                                    {"string","driven_tag","bone_bw"},{"float","bone_driven_weight","0.0"},
                                    {"string","muscle_id_tag","ms_id_tag"}  
                                  },
                                  {"FEM"}});

}