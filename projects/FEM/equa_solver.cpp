#include "declares.h"

namespace zeno{

struct SolveFEM : zeno::INode {
    virtual void apply() override {
        Eigen::SparseLU<SpMat> _LUSolver;
        Eigen::SimplicialLDLT<SpMat> _LDLTSolver;

        auto integrator = get_input<FEMIntegrator>("integrator");
        auto shape = get_input<PrimitiveObject>("shape");

        int max_iters = get_param<int>("maxNRIters");
        int max_linesearch = get_param<int>("maxBTLs");
        float c1 = get_param<float>("ArmijoCoeff");
        float c2 = get_param<float>("CurvatureCoeff");
        float beta = get_param<float>("BTL_shrinkingRate");
        float epsilon = get_param<float>("epsilon");

        std::vector<Vec2d> wolfeBuffer;
        wolfeBuffer.resize(max_linesearch);

        int search_idx = 0;

        VecXd r,HBuffer,dp;
        r.resize(shape->size() * 3);
        dp.resize(shape->size() * 3);
        HBuffer.resize(integrator->_connMatrix.nonZeros());

        auto& cpos = shape->attr<zeno::vec3f>("curPos");

        size_t iter_idx = 0;

        FEM_Scaler r0 = 0;
        FEM_Scaler e_start = 0;

        FEM_Scaler stop_error = 0;


        do{
            FEM_Scaler e0,e1,eg0;

            e0 = integrator->EvalObjDerivHessian(shape,r,HBuffer,true);
            // break;
            if(iter_idx == 0)
                r0 = r.norm();

            if(std::isnan(e0) || std::isnan(r.norm()) || std::isnan(HBuffer.norm())){
                const auto& pos = shape->verts;
                const auto& examShape = shape->attr<zeno::vec3f>("examShape");
                const auto& examW = shape->attr<float>("examW");
                const auto& interpWSum = shape->attr<float>("interpWSum");
                for(size_t i = 0;i < shape->size();++i){
                    if(std::isnan(zeno::length(pos[i]))){
                        std::cout << "NAN POS : " << i << "\t" << pos[i][0] << "\t" << pos[i][1] << "\t" << pos[i][2] << std::endl;
                    }
                    if(std::isnan(zeno::length(examShape[i])) || std::isnan(examW[i])){
                        std::cout << "EXAMSHAPE : " << i << "\t" << examShape[i][0] << "\t" << examShape[i][1] << "\t" << examShape[i][2] << std::endl;
                        std::cout << "EXAMW : " << i << "\t" << examW[i] << std::endl;
                        std::cout << "INTERPW : " << i << "\t" << interpWSum[i] << std::endl;
                    }
                }
                std::cerr << "NAN VALUE DETECTED : " << e0 << "\t" << r.norm() << "\t" << HBuffer.norm() << std::endl;
                throw std::runtime_error("NAN VALUE DETECTED");
            }

            FEM_Scaler stopError = 0;
            if(iter_idx == 0){
                for(size_t i = 0;i < shape->quads.size();++i){
                    Vec12d tet_shape;
                    const auto& tet = shape->quads[i];
                    integrator->RetrieveElmCurrentShape(i,tet_shape,shape);

                    Mat3x3d F;
                    BaseIntegrator::ComputeDeformationGradient(integrator->_elmMinv[i],tet_shape,F);

                    TetAttributes attrs;
                    integrator->AssignElmAttribs(i,shape,attrs);

                    FEM_Scaler psi;
                    Vec9d dpsi;
                    Mat9x9d ddpsi;
                    integrator->muscle->_forceModel->ComputePsiDerivHessian(attrs,F,psi,dpsi,ddpsi,false);

                    stop_error += integrator->_elmCharacteristicNorm[i] * ddpsi.norm();
                }

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
                    e1 = integrator->EvalObj(shape);
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

        set_output("shape",shape); 
        std::cout << "FINISH STEPPING " << "\t" << iter_idx << "\t" << r.norm() << std::endl;
    }

    static void UpdateCurrentShape(std::shared_ptr<PrimitiveObject> prim,const VecXd& dp,double alpha){
        auto& cpos = prim->attr<zeno::vec3f>("curPos");
        for(size_t i = 0;i < prim->size();++i)
            cpos[i] += alpha * zeno::vec3f(dp[i*3 + 0],dp[i*3 + 1],dp[i*3 + 2]);
    }
};

ZENDEFNODE(SolveFEM,{
    {"integrator","shape"},
    {"shape"},
    {{"int","maxNRIters","10"},{"int","maxBTLs","10"},{"float","ArmijoCoeff","0.01"},
        {"float","CurvatureCoeff","0.9"},{"float","BTL_shrinkingRate","0.5"},
        {"float","epsilon","1e-8"}
    },
    {"FEM"},
});

};