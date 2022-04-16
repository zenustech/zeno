#include "declares.h"

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