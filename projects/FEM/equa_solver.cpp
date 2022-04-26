#include "declares.h"
#include <LBFGS.h>

namespace zeno{



// a jiggle deformer use successive 3 frames
struct Jiggle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        // the deformation without jiggling
        const auto& cpos_vec =  prim->attr<zeno::vec3f>("curPos");
        const auto& ppos_vec =  prim->attr<zeno::vec3f>("prePos");
        const auto& pppos_vec = prim->attr<zeno::vec3f>("preprePos");

        // the first time Jiggle get called
        if(!prim->has_attr("curJiggle")){
            prim->add_attr<zeno::vec3f>("curJiggle");
            prim->add_attr<zeno::vec3f>("preJiggle");
            prim->add_attr<zeno::vec3f>("prepreJiggle");
            auto& cjiggle = prim->attr<zeno::vec3f>("curJiggle");
            auto& pjiggle =   prim->attr<zeno::vec3f>("preJiggle");
            auto& ppjiggle =  prim->attr<zeno::vec3f>("prepreJiggle");

            #pragma omp parallel for
            for(size_t i = 0;i < prim->size();++i){
                cjiggle[i] = ppos_vec[i];
                pjiggle[i] = ppos_vec[i];
                ppjiggle[i] = ppos_vec[i];
            }
        }

        auto& cjiggle = prim->attr<zeno::vec3f>("curJiggle");
        auto& pjiggle =   prim->attr<zeno::vec3f>("preJiggle");
        auto& ppjiggle =  prim->attr<zeno::vec3f>("prepreJiggle");

        auto jiggleStiffness = get_input2<float>("jiggleStiffness");
        auto jiggleDamp = get_input2<float>("jiggleDamping");

        if(jiggleDamp > 0.8){
            std::cout << "input jiggle damp >= 0.8, clamp to 0.8" << std::endl;
            jiggleDamp = 0.8;
        }

        auto jiggleRate = get_input2<float>("jiggleRate");

        auto jdt = 1.0/jiggleRate;


//  try explicit integrator first
        #pragma omp parallel for
        for(size_t i = 0;i < prim->size();++i){
            const auto& cpos = cpos_vec[i];
            const auto& ppos = ppos_vec[i];
            const auto& pppos = pppos_vec[i];

            ppjiggle[i] = pjiggle[i];
            pjiggle[i] = cjiggle[i];

            auto& cj = cjiggle[i];
            const auto& pj = pjiggle[i];
            const auto& ppj = ppjiggle[i];

//          the previous velocity applied with damping
            auto jvec = (1 - jiggleDamp) * (pj - ppj)/jdt;

// compute accel
            auto tension = cpos - pjiggle[i];

            cj += jvec * jdt + 0.5 * tension * jdt * jdt;
        }


        set_output("prim",prim); 
    }
};


ZENDEFNODE(Jiggle, {
    {"prim",
        {"float","jiggleStiffness","10"},
        {"float","jiggleDamping","0.5"},
        {"float","jiggleRate","1"}
    },
    {"prim"},
    {},
    {"FEM"},
});


struct Jiggle2 : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto jprim = get_input<zeno::PrimitiveObject>("jprim");

        const auto& cpos_vec =  prim->attr<zeno::vec3f>("curPos");
        const auto& ppos_vec =  prim->attr<zeno::vec3f>("prePos");
        const auto& pppos_vec = prim->attr<zeno::vec3f>("preprePos");

        auto& j_cpos_vec = jprim->attr<zeno::vec3f>("curJiggle");
        auto& j_ppos_vec = jprim->attr<zeno::vec3f>("preJiggle");
        auto& j_pppos_vec= jprim->attr<zeno::vec3f>("prepreJiggle");

        auto jiggleStiffness = get_input2<float>("jiggleRate");
        auto characterLen = get_input2<float>("characterLen");
        auto jiggleScale = get_input2<float>("jiggleScale");
        // jiggleStiffness /= characterLen;

        auto jiggleDamp = get_input2<float>("jiggleDamping");
        // auto jiggleWeight = get_input2<float>("jiggleWeight");

        auto jiggleWs = jprim->attr<float>("jws");

        if(jiggleDamp > 0.8) {
            std::cout << "Input jiggle damp >= 0.8, clamp to 0.8" << std::endl;
            jiggleDamp = 0.8;
        }

        // auto jiggleRate = get_input2<float>("jiggleRate");
        // auto jiggle
        // auto jiggleStiffness = 1.0/jiggleRate;
        auto jdt = 1;

        #pragma omp parallel for
        for(size_t i = 0;i < prim->size();++i){
            const auto& cpos = cpos_vec[i];
            const auto& ppos = ppos_vec[i];
            const auto& pppos= pppos_vec[i];
            
            j_pppos_vec[i] = j_ppos_vec[i];
            j_ppos_vec[i] = j_cpos_vec[i];

            auto& cj = j_cpos_vec[i];
            const auto& pj = j_ppos_vec[i];
            const auto& ppj = j_pppos_vec[i];

            auto jvec = (1 - jiggleDamp) * (pj - ppj)/jdt;    

            auto tension = jiggleStiffness * (cpos - pj);
            cj += jvec * jdt + 0.5 * tension * jdt * jdt;  

            auto jw = jiggleScale * jiggleWs[i];

            cj = cpos * (1 - jw) + jw * cj;
            // if(i == 0)
            //     std::cout << "cj : " << cj[0] << "\t" << cj[1] << "\t" << cj[2] << std::endl;  
        }

        set_output("jprim",jprim); 

    }
};

ZENDEFNODE(Jiggle2, {
    {"prim","jprim",
        // {"float","jiggleWeight","1"},
        {"float","jiggleDamping","0.5"},
        {"float","jiggleRate","5"},
        {"float","characterLen","1"},
        {"float","jiggleScale","1"},
    },
    {"jprim"},
    {},
    {"FEM"},
});

struct LaplaceOperator : zeno::IObject {
    LaplaceOperator() = default;
    std::shared_ptr<PrimitiveObject> mesh;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<FEM_Scaler>> laplace_solver;
};

struct BuildLapaceOperator : zeno::INode {
    void AssignElmInterpShape(size_t nm_elms,
        const std::shared_ptr<PrimitiveObject>& interpShape,
        std::vector<std::vector<Vec3d>>& interpPs,
        std::vector<std::vector<Vec3d>>& interpWs){
            interpPs.resize(nm_elms);
            interpWs.resize(nm_elms);
            for(size_t i = 0;i < nm_elms;++i){
                interpPs[i].clear();
                interpWs[i].clear();
            }
            // std::cout << "TRY ASSIGN INTERP SHAPE" << std::endl;
            if(interpShape && interpShape->has_attr("embed_id") && interpShape->has_attr("embed_w")){
                // std::cout << "ASSIGN ATTRIBUTES" << std::endl;
                const auto& elm_ids = interpShape->attr<float>("embed_id");
                const auto& elm_ws = interpShape->attr<zeno::vec3f>("embed_w");
                const auto& pos = interpShape->verts;

                // #pragma omp parallel for 
                for(size_t i = 0;i < interpShape->size();++i){
                    auto elm_id = elm_ids[i];
                    const auto& pos = interpShape->verts[i];
                    const auto& w = elm_ws[i];
                    interpPs[elm_id].emplace_back(pos[0],pos[1],pos[2]);
                    interpWs[elm_id].emplace_back(w[0],w[1],w[2]);
                }
            }
            // if(!interpShape){
            //     std::cout << "NULLPTR FOR INTERPSHAPE" << std::endl;
            // }
            // if()
    }

    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto elmView = get_input<zeno::PrimitiveObject>("elmView");
        auto integrator = get_input<FEMIntegrator>("integrator");
        std::shared_ptr<PrimitiveObject> interpShape = has_input("skin") ? get_input<PrimitiveObject>("skin") : nullptr;
        auto res = std::make_shared<LaplaceOperator>();
        res->mesh = prim;
        Eigen::SparseMatrix<FEM_Scaler> L;
        L.resize(prim->size() * 3,prim->size() * 3);

        std::vector<Mat9x12d> elm_dFdx;
        std::vector<float> elm_stiffness;
        elm_dFdx.resize(prim->quads.size());
        elm_stiffness.resize(prim->quads.size());

        std::vector<std::vector<Vec3d>> interpPs;
        std::vector<std::vector<Vec3d>> interpWs;
        AssignElmInterpShape(prim->quads.size(),interpShape,interpPs,interpWs);

        #pragma omp parallel for
        for(size_t elm_id = 0;elm_id < prim->quads.size();++elm_id){
            const auto& elm = prim->quads[elm_id];
            Mat3x3d Dm;
            for(size_t i = 1;i < 4;++i){
                const auto& vert = prim->verts[elm[i]];
                const auto& vert0 = prim->verts[elm[0]];
                auto vi0 = vert - vert0; 
                Dm.col(i - 1) << vi0[0],vi0[1],vi0[2];
            }

            Mat3x3d DmInv = Dm.inverse();
            double m = DmInv(0,0);
            double n = DmInv(0,1);
            double o = DmInv(0,2);
            double p = DmInv(1,0);
            double q = DmInv(1,1);
            double r = DmInv(1,2);
            double s = DmInv(2,0);
            double t = DmInv(2,1);
            double u = DmInv(2,2);

            double t1 = - m - p - s;
            double t2 = - n - q - t;
            double t3 = - o - r - u; 

            elm_dFdx[elm_id] << 
                t1, 0, 0, m, 0, 0, p, 0, 0, s, 0, 0, 
                0,t1, 0, 0, m, 0, 0, p, 0, 0, s, 0,
                0, 0,t1, 0, 0, m, 0, 0, p, 0, 0, s,
                t2, 0, 0, n, 0, 0, q, 0, 0, t, 0, 0,
                0,t2, 0, 0, n, 0, 0, q, 0, 0, t, 0,
                0, 0,t2, 0, 0, n, 0, 0, q, 0, 0, t,
                t3, 0, 0, o, 0, 0, r, 0, 0, u, 0, 0,
                0,t3, 0, 0, o, 0, 0, r, 0, 0, u, 0,
                0, 0,t3, 0, 0, o, 0, 0, r, 0, 0, u;

            auto E  = elmView->attr<float>("E")[elm_id];
            auto nu = elmView->attr<float>("nu")[elm_id];

            auto lambda = ElasticModel::Enu2Lambda(E,nu);
            auto mu = ElasticModel::Enu2Mu(E,nu); 

            elm_stiffness[elm_id] = lambda + mu;         
        }

        const auto& vols = elmView->attr<float>("V");


        std::vector<Eigen::Triplet<FEM_Scaler>> triplets;
        triplets.resize(prim->quads.size() * 12 * 12);


//      Compute the Laplace Operator For Elastic Object
        #pragma omp parallel for
        for(size_t elm_id = 0;elm_id < prim->quads.size();++elm_id){
            const auto& elm = prim->quads[elm_id];
            Mat12x12d elm_H = elm_stiffness[elm_id] * vols[elm_id] * elm_dFdx[elm_id].transpose() * elm_dFdx[elm_id];

            auto interpPenalty = elmView->has_attr("embed_PC") ? elmView->attr<float>("embed_PC")[elm_id] : 0;

            if(interpPs[elm_id].size() > 0){
                for(size_t i = 0;i < interpPs[elm_id].size();++i){
                    const auto& ipos = interpPs[elm_id][i];
                    const auto& w = interpWs[elm_id][i];

                    Vec4d iw;
                    iw << w[0],w[1],w[2],1-w[0]-w[1]-w[2];

                    for(size_t j = 0;j < 4;++j)
                        for(size_t k = 0;k < 4;++k){
                            FEM_Scaler alpha = interpPenalty * iw[j] * iw[k] / interpPs[elm_id].size();
                            elm_H.block(j * 3,k*3,3,3).diagonal() += Vec3d::Constant(alpha);
                        }                    
                }
            }            

            for(size_t i = 0;i < 12;++i)
                for(size_t j = 0;j < 12;++j){
                    size_t vr_id = i / 3;
                    size_t dr_id = i % 3;
                    size_t vc_id = j / 3;
                    size_t dc_id = j % 3;
                    auto val = elm_H(i,j);
                    triplets[elm_id * 12 * 12 + (vr_id * 3 + dr_id) * 12 + vc_id * 3 + dc_id] = Eigen::Triplet<FEM_Scaler>(elm[vr_id] * 3 + dr_id,elm[vc_id] * 3 + dc_id,val);
                }
                

        }
        L.setZero();
        L.setFromTriplets(triplets.begin(),triplets.end());
        res->laplace_solver.compute(L);

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(BuildLapaceOperator,{
    {"prim","elmView","integrator","skin"
    },
    {"res"},
    {},
    {"FEM"},
});



struct SolveFEMFast : zeno::INode {
    virtual void apply() override {
        using namespace LBFGSpp;

        auto integrator = get_input<FEMIntegrator>("integrator");
        auto shape = get_input<PrimitiveObject>("shape");
        auto& cpos = shape->attr<zeno::vec3f>("curPos");
        auto& ppos = shape->attr<zeno::vec3f>("prePos");
        auto& pppos = shape->attr<zeno::vec3f>("preprePos");
        auto laplace_op = get_input<LaplaceOperator>("laplaceOp");
        for(size_t i = 0;i < shape->size();++i){
            pppos[i] = ppos[i];
            ppos[i] = cpos[i];
        }

        auto elmView = get_input<PrimitiveObject>("elmView");

        std::shared_ptr<PrimitiveObject> interpShape = has_input("skin") ? get_input<PrimitiveObject>("skin") : nullptr;
        auto max_iters = get_input2<int>("maxNRIters");
        auto max_linesearch = get_input2<int>("maxBTLs");
        auto c1 = get_input2<float>("ArmijoCoeff");
        auto c2 = get_input2<float>("CurvatureCoeff");
        auto beta = get_input2<float>("BTL_shrinkingRate");
        auto epsilon = get_input2<float>("epsilon");
        auto rel_epsilon = get_input2<float>("rel_epsilon");

        auto window_size = get_input2<int>("window_size");

        LBFGSParam<FEM_Scaler> param;
        param.m = window_size;
        param.epsilon = epsilon;
        param.epsilon_rel = rel_epsilon;
        param.max_iterations = max_iters;
        param.max_linesearch = max_linesearch;
        param.ftol = c1;
        param.wolfe = c2;

        param.check_param();


        LBFGSSolver<FEM_Scaler> solver(param);

        Eigen::VectorXd _x(shape->size() * 3);
        for(size_t i = 0;i < shape->size();++i)
            _x.segment(i*3,3) << cpos[i][0],cpos[i][1],cpos[i][2];


        FEM_Scaler _fx;
        int niter = solver.minimize(
            [&](const Eigen::VectorXd& x,Eigen::VectorXd& grad) mutable {
                // std::cout << "HELO LAMBDA TEST" << std::endl;
                for(size_t i = 0;i < shape->size();++i)
                    cpos[i] = zeno::vec3f(x[i*3 + 0],x[i*3 + 1],x[i*3 + 2]);
                return integrator->EvalObjDeriv(shape,elmView,interpShape,grad);
            },
            _x,
            _fx,// TODO: define the inverse of initial hessian approximation
            [&](const Eigen::VectorXd& b) mutable {
                auto res = laplace_op->laplace_solver.solve(b);
                return res;
            },
            true
        );

        for(size_t i = 0;i < shape->size();++i)
            cpos[i] = zeno::vec3f(_x[i*3+ 0],_x[i*3+1],_x[i*3+2]);

        std::cout << "FINISH STEPPING " << "\t" << niter << "\t" << max_iters << std::endl;
        set_output("shape",shape); 
    }
};

ZENDEFNODE(SolveFEMFast,{
    {"integrator","shape","elmView","laplaceOp","skin",{"int","maxNRIters","10"},{"int","maxBTLs","10"},{"float","ArmijoCoeff","0.01"},
        {"float","CurvatureCoeff","0.9"},{"float","BTL_shrinkingRate","0.5"},
        {"float","epsilon","1e-8"},{"float","rel_epsilon","1e-5"},{"int","window_size","5"}
    },
    {"shape"},
    {},
    {"FEM"},
});


struct SolveFEM : zeno::INode {
    virtual void apply() override {
        // std::cout << "BEGIN SOLVER " << std::endl;
        Eigen::SparseLU<SpMat> _LUSolver;
        Eigen::SimplicialLDLT<SpMat> _LDLTSolver;

        auto integrator = get_input<FEMIntegrator>("integrator");
        auto shape = get_input<PrimitiveObject>("shape");
        auto elmView = get_input<PrimitiveObject>("elmView");
        std::shared_ptr<PrimitiveObject> interpShape = has_input("skin") ? get_input<PrimitiveObject>("skin") : nullptr;

        // int max_iters = get_param<int>("maxNRIters");
        // int max_linesearch = get_param<int>("maxBTLs");
        // float c1 = get_param<float>("ArmijoCoeff");
        // float c2 = get_param<float>("CurvatureCoeff");
        // float beta = get_param<float>("BTL_shrinkingRate");
        // float epsilon = get_param<float>("epsilon");

        auto max_iters = get_input2<int>("maxNRIters");
        auto max_linesearch = get_input2<int>("maxBTLs");
        auto c1 = get_input2<float>("ArmijoCoeff");
        auto c2 = get_input2<float>("CurvatureCoeff");
        auto beta = get_input2<float>("BTL_shrinkingRate");
        auto epsilon = get_input2<float>("epsilon");

        std::vector<Vec2d> wolfeBuffer;
        wolfeBuffer.resize(max_linesearch);

        int search_idx = 0;

        VecXd r,HBuffer,dp;
        r.resize(shape->size() * 3);
        dp.resize(shape->size() * 3);
        HBuffer.resize(integrator->_connMatrix.nonZeros());

        auto& cpos = shape->attr<zeno::vec3f>("curPos");
        auto& ppos = shape->attr<zeno::vec3f>("prePos");
        auto& pppos = shape->attr<zeno::vec3f>("preprePos");

        for(size_t i = 0;i < shape->size();++i){
            pppos[i] = ppos[i];
            ppos[i] = cpos[i];
        }


        size_t iter_idx = 0;

        FEM_Scaler r0 = 0;
        FEM_Scaler e_start = 0;

        FEM_Scaler stop_error = 0;

        // std::cout << "BEGIN LOOP" << std::endl;

        do{
            FEM_Scaler e0,e1,eg0;

            e0 = integrator->EvalObjDerivHessian(shape,elmView,interpShape,r,HBuffer,true);
            // std::cout << "FINISH EVAL A X B" << std::endl;
            // break;
            if(iter_idx == 0)
                r0 = r.norm();

            if(std::isnan(e0) || std::isnan(r.norm()) || std::isnan(HBuffer.norm())){
                const auto& pos = cpos;
                const auto& examShape = shape->attr<zeno::vec3f>("examShape");
                const auto& examW = shape->attr<float>("examW");
                for(size_t i = 0;i < shape->size();++i){
                    if(std::isnan(zeno::length(pos[i]))){
                        std::cout << "NAN POS : " << i << "\t" << pos[i][0] << "\t" << pos[i][1] << "\t" << pos[i][2] << std::endl;
                    }
                    if(std::isnan(zeno::length(examShape[i])) || std::isnan(examW[i])){
                        std::cout << "EXAMSHAPE : " << i << "\t" << examShape[i][0] << "\t" << examShape[i][1] << "\t" << examShape[i][2] << std::endl;
                        std::cout << "EXAMW : " << i << "\t" << examW[i] << std::endl;
                    }
                }
                std::cerr << "NAN VALUE DETECTED : " << e0 << "\t" << r.norm() << "\t" << HBuffer.norm() << std::endl;
                // std::cout << "R:" << std::endl << r.transpose() << std::endl;
                for(size_t i = 0;i < shape->size();++i){
                    if(std::isnan(r.segment(i*3,3).norm()))
                        std::cout << "<" << i << "> : \t" << r.segment(i*3,3).transpose() << std::endl;
                }
                throw std::runtime_error("NAN VALUE DETECTED");
            }

            FEM_Scaler stopError = 0;
            if(iter_idx == 0){
                std::vector<std::vector<Vec3d>> interpPs;
                std::vector<std::vector<Vec3d>> interpWs;
                integrator->AssignElmInterpShape(shape->quads.size(),interpShape,interpPs,interpWs);
                for(size_t i = 0;i < shape->quads.size();++i){
                    Vec12d tet_shape;
                    const auto& tet = shape->quads[i];
                    integrator->RetrieveElmCurrentShape(i,tet_shape,shape);

                    Mat3x3d F;
                    BaseIntegrator::ComputeDeformationGradient(integrator->_elmMinv[i],tet_shape,F);



                    TetAttributes attrs;
                    integrator->AssignElmAttribs(i,shape,elmView,attrs);
                    attrs.interpPenaltyCoeff = elmView->has_attr("embed_PC") ? elmView->attr<float>("embed_PC")[i] : 0;
                    attrs.interpPs = interpPs[i];
                    attrs.interpWs = interpWs[i]; 


                    FEM_Scaler psi;
                    Vec9d dpsi;
                    Mat9x9d ddpsi;
                    integrator->muscle->_forceModel->ComputePsiDerivHessian(attrs,F,psi,dpsi,ddpsi,false);

                    stop_error += integrator->_elmCharacteristicNorm[i] * ddpsi.norm();
                }

                // std::cout << "EVAL CNORM" << std::endl;

                stop_error *= epsilon;
                stop_error *= sqrt(shape->quads.size());               
            }

            if(r.norm() < stop_error){
                // std::cout << "BREAK WITH " << r.norm() << "\t" << iter_idx << std::endl;
                break;
            }
            r *= -1;

            clock_t begin_solve = clock();
            _LDLTSolver.compute(MatHelper::MapHMatrix(shape->size(),integrator->_connMatrix,HBuffer.data()));
            dp = _LDLTSolver.solve(r);
            clock_t end_solve = clock();

            // std::cout << "INTERNAL SIZE : " << r.norm() << "\t" << dp.norm() << HBuffer.norm() << std::endl;

            eg0 = -dp.dot(r);
            if(eg0 > 0){
                std::cout << "eg0 = " << eg0 << std::endl;
                throw std::runtime_error("non-negative descent direction");
            }

            // std::cout << "DO LINE SEARCH" << std::endl;
            bool do_line_search = true;
            size_t search_idx = 0;
            if(!do_line_search)
                UpdateCurrentShape(shape,dp,1.0);
            else{
                search_idx = 0;

                FEM_Scaler alpha = 2.0f;
                FEM_Scaler beta = 0.5f;
                FEM_Scaler c1 = 0.001f;

                double armijo_condition;
                do{
                    if(search_idx != 0)
                        UpdateCurrentShape(shape,dp,-alpha);
                    alpha *= beta;
                    UpdateCurrentShape(shape,dp,alpha);
                    e1 = integrator->EvalObj(shape,elmView,interpShape);
                    ++search_idx;
                    wolfeBuffer[search_idx-1](0) = (e1 - e0)/alpha;
                    wolfeBuffer[search_idx-1](1) = eg0;

                    armijo_condition = double(e1) - double(e0) - double(c1)*double(alpha)*double(eg0);
                }while(/*(e1 > e0 + c1*alpha*eg0)*/ armijo_condition > 0.0f /* || (fabs(eg1) > c2*fabs(eg0))*/ && (search_idx < max_linesearch));

                if(search_idx == max_linesearch){
                    std::cout << "LINESEARCH EXCEED" << std::endl;
                    for(size_t i = 0;i < max_linesearch;++i)
                        std::cout << "idx:" << i << "\t" << wolfeBuffer[i].transpose() << std::endl;
                    break;
                }
            }
            // std::cout << "SOLVE TIME : " << (float)(end_solve - begin_solve)/CLOCKS_PER_SEC << "\t" << r0 << "\t" << r.norm() << "\t" << eg0 << "\t" << search_idx << "\t" << e_start << "\t" << e0 << "\t" << e1 << std::endl;

            ++iter_idx;
        }while(iter_idx < max_iters);

        if(iter_idx == max_iters){
            std::cout << "MAX NEWTON ITERS EXCEED" << std::endl;
        }
        std::cout << "FINISH STEPPING " << "\t" << iter_idx << "\t" << r.norm() << std::endl;
        set_output("shape",shape); 

    }

    static void UpdateCurrentShape(std::shared_ptr<PrimitiveObject> prim,const VecXd& dp,double alpha){
        auto& cpos = prim->attr<zeno::vec3f>("curPos");
        for(size_t i = 0;i < prim->size();++i)
            cpos[i] += alpha * zeno::vec3f(dp[i*3 + 0],dp[i*3 + 1],dp[i*3 + 2]);
    }
};

ZENDEFNODE(SolveFEM,{
    {"integrator","shape","elmView","skin",{"int","maxNRIters","10"},{"int","maxBTLs","10"},{"float","ArmijoCoeff","0.01"},
        {"float","CurvatureCoeff","0.9"},{"float","BTL_shrinkingRate","0.5"},
        {"float","epsilon","1e-8"}
    },
    {"shape"},
    {},
    {"FEM"},
});

};