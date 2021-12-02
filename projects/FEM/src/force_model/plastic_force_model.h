#include <base_elastic_model.h>
#include <base_force_model.h>
#include <cmath>


// the Joshph Terra approach for gaurantee SPD is needed? or not
class PlasticForceModel : public BaseForceModel {
public:
    PlasticForceModel(std::shared_ptr<ElasticModel> _elastic_model) : BaseForceModel(),elastic_model(_elastic_model) {
        P = Mat3x3d::Identity() - Mat3x3d::Constant(1.0/3);
        PTP = P.transpose() * P;
    }
    ~PlasticForceModel() {}

    // We explicitly update the plastic material parameter before the Newton-Raphson algorithm take place. And simplify the implementation of the algorithm
    void UpdatePlasticParameters(size_t elm_id,TetAttributes& attrs,const Mat3x3d& F,FEM_Scaler& vm,FEM_Scaler& rm) const {

        // std::cout << "UPDATE P PARAMETERS" << std::endl;

        Mat3x3d U,V;Vec3d total_strain;
        DiffSVD::SVD_Decomposition(F,U,total_strain,V);

        // Mat3x3d R,S;
        // DiffSVD::Polar_Decomposition(F,R,S);

        Mat3x3d Up,Vp;Vec3d pstrain;
        DiffSVD::SVD_Decomposition(attrs.pmp.PS,Up,pstrain,Vp);


        // DO THE ALIGN
        Mat3x3d Ur = U,Vr = V;
        Vec3d sr = total_strain;
        for(size_t i = 0;i < 3;++i){
            Ur = MatHelper::ROLL(Ur);
            Vr = MatHelper::ROLL(Vr);
            sr = MatHelper::ROLL(sr);

            FEM_Scaler U_SHIFT = (Up.transpose() * Ur - Mat3x3d::Identity()).norm();
            FEM_Scaler V_SHIFT = (Vp.transpose() * Vr - Mat3x3d::Identity()).norm();

            std::cout << "SHIFT<" << i << "> : " << U_SHIFT << "\t" << V_SHIFT << std::endl;

            if(U_SHIFT < 1 && V_SHIFT < 1)
                break;

            if(i == 2){
                std::cout << "SHIFT : " << U_SHIFT << "\t" << V_SHIFT << std::endl;
                std::cout << "U : " << std::endl << U << std::endl << Up << std::endl << Ur << std::endl;
                std::cout << "V : " << std::endl << V << std::endl << Vp << std::endl << Vr << std::endl;
                std::cout << "DET : " << U.determinant() << "\t" << V.determinant() << std::endl;
                std::cout << "ROTATION? :" << std::endl << U * U.transpose()  << std::endl << V * V.transpose() << std::endl;
                throw std::runtime_error("SHIFT_ERROR");
            }
        }

        // Mat3x3d Rp,Sp;
        // DiffSVD::Polar_Decomposition(pmp.PS,Rp,Sp);

        // Mat3x3d dR = R * Rp.transpose() - Mat3x3d::Identity();
        // if(dR.norm() > 1e-2){
        //     std::cout << "THE ROTATION DIFF IS TOO BIG" << std::endl;
        //     std::cout << dR << std::endl;
        //     throw std::runtime_error("THE ROTATION DIFF IS TOO BIG");
        // }


        FEM_Scaler s01 = total_strain[0] - total_strain[1] - (attrs.pmp.init_strain[0] - attrs.pmp.init_strain[1]);
        FEM_Scaler s02 = total_strain[0] - total_strain[2] - (attrs.pmp.init_strain[0] - attrs.pmp.init_strain[2]);
        FEM_Scaler s12 = total_strain[1] - total_strain[2] - (attrs.pmp.init_strain[1] - attrs.pmp.init_strain[2]);

        rm = sqrt(0.5 * (s01*s01 + s02*s02 + s12*s12));

        if(rm > attrs.pmp.failed_strain){
            attrs.pmp.failed = true;
            attrs.pmp.the_strain_failed = total_strain;
            attrs.pmp.F_failed = U * attrs.pmp.the_strain_failed.asDiagonal() * V.transpose();

            // std::cout << "FAIL_ELM<" << elm_id << "> : " << total_strain.transpose() << std::endl;
            // throw std::runtime_error("FAIL");
            return;
        }else{
            attrs.pmp.failed = false;
        }

        Vec3d trial_strain = total_strain - attrs.pmp.plastic_strain;
        Vec3d trial_stress;
        elastic_model->ComputePrincipalStress(attrs,trial_strain,trial_stress);
        
        vm = EvalVM(trial_stress,attrs.pmp.kinematic_hardening_shift,attrs.pmp.init_stress,attrs.pmp.yield_stress);      

        // the effective stress in the interior of yielding surface, only pure elastic behavior, no update is needed
        if(vm < 0){
            // update the orientation of platic deformation
            attrs.pmp.PS = U * attrs.pmp.plastic_strain.asDiagonal() * V.transpose();
            return;
        }

        // if the effective strain go beyond the restoring strain, the object return back to pure elastic behavior
        FEM_Scaler restoring = 1.0;// the default kinematric hardening with plastic behavior
        if(rm > attrs.pmp.restoring_strain && rm < attrs.pmp.failed_strain){
            // std::cout << "<" << elm_id << ">rm = " << rm << "\t" << attrs.pmp.restoring_strain << std::endl;
            restoring = 0.0;
            // attrs.pmp.PS = U * attrs.pmp.plastic_strain.asDiagonal() * V.transpose();
            // return;
        }
        // std::cout << "VM<" << elm_id << "> : \t" << vm << "\t" << attrs.pmp.yield_stress << std::endl;


        
        // else{
        //     std::cout << "VM<" << elm_id << "> : \t" << vm << "\t" << attrs.pmp.yield_stress << std::endl;
        //     return;
        // }

        // std::cout << "UPDATE PLASTIC PARAM " << vm << std::endl;
        // std::cout << "ORI_PARAM : " << std::endl;
        // std::cout << attrs.pmp.plastic_strain.transpose() << std::endl;
        // std::cout << attrs.pmp.kinematic_hardening_shift.transpose() << std::endl;

        Vec3d p_flow;
        EvalAssociativePFlow(trial_stress,attrs.pmp.kinematic_hardening_shift,attrs.pmp.init_stress,attrs.pmp.yield_stress,p_flow);

        // std::cout << "PFLOW : " << p_flow.transpose() << std::endl;

        //solve the step using newthon-raphson algorithm
        Vec4d x;// set the initial solution for return mapping algorithm.....i.e [dstep, dpTrans]
        x << 0,attrs.pmp.kinematic_hardening_shift;


        bool debug = false;
        if(debug){
            // // std::cout << "DEBUG CODE : " << std::endl;
            Vec4d x_tmp = x;
            Vec4d res_cmp,res_tmp;
            Mat4x4d J_cmp,J_fd;
            EvalReturnMappingResJacobi(attrs,total_strain,p_flow,x,res_cmp,J_cmp,restoring);

            for(size_t i = 0;i < 4;++i){
                FEM_Scaler step = x[i] * 1e-6;
                step = fabs(step) < 1e-6 ? 1e-6 : step;
                x_tmp = x;
                x_tmp[i] += step;

                EvalReturnMappingRes(attrs,total_strain,p_flow,x_tmp,res_tmp,restoring);
                J_fd.col(i) = (res_tmp - res_cmp) / step;
            }


            FEM_Scaler J_error = (J_cmp - J_fd).norm() / J_cmp.norm();
            if(J_error > 1e-5){
                std::cout << "J_cmp : " << std::endl << J_cmp << std::endl;
                std::cout << "J_fd : " << std::endl << J_fd << std::endl;
                std::cout << "restoring : " << restoring << std::endl;

                throw std::runtime_error("J_CHECK");
            }
        }

        Vec4d res;
        Vec3d resp_stress = trial_stress;
        Mat4x4d Jacobi;
        size_t nm_iters = 0;
        size_t max_iters = 200;
        std::vector<FEM_Scaler> iter_buffer;
        iter_buffer.resize(max_iters);
        do{
            EvalReturnMappingResJacobi(attrs,total_strain,p_flow,x,res,Jacobi,restoring);
            iter_buffer[nm_iters] = res.norm();
            if(res.norm() < 1e-2) 
                break;

            Vec4d dx = -Jacobi.inverse() * res;
            x += dx;
            ++nm_iters;
        }while(nm_iters < max_iters);

        if(nm_iters == max_iters){
            std::cout << "ELM<" << elm_id << "> : " << "FAIL UPDATE PLASTIC" << std::endl;
            return;
        }

        attrs.pmp.plastic_strain += x[0] * p_flow * restoring; // dstrain_p = test_step * p_flow;
        attrs.pmp.kinematic_hardening_shift += x.segment(1,3);   


        // if(elm_id == 9180 || restoring == 0.0){
        //     Vec4d x_test = Vec4d::Zero();
        //     std::cout << "FINISH_RETURN_MAPPING_RES<" << elm_id << "> : " << res.transpose() << std::endl;
        //     EvalReturnMappingResJacobi(pmp,emp,total_strain,p_flow,x_test,res,Jacobi,restoring);
        //     std::cout << "CHECK_RETURN_MAPPING_RES<" << elm_id <<  "> : " << res.transpose() << std::endl;
        //     trial_strain = total_strain - attrs.pmp.plastic_strain;
        //     elastic_model->ComputePrincipalStress(emp,trial_strain,trial_stress);
        //     vm = EvalVM(trial_stress,attrs.pmp.kinematic_hardening_shift,attrs.pmp.init_stress,attrs.pmp.yield_stress);   
        //     std::cout << "VM : " << vm << std::endl;
        // }
        

        attrs.pmp.PS = U * attrs.pmp.plastic_strain.asDiagonal() * V.transpose();

        trial_strain = total_strain - attrs.pmp.plastic_strain;
        elastic_model->ComputePrincipalStress(attrs,trial_strain,trial_stress);
        vm = EvalVM(trial_stress,attrs.pmp.kinematic_hardening_shift,attrs.pmp.init_stress,attrs.pmp.yield_stress);    
        // std::cout << "UPDATE PS : " << std::endl << attrs.pmp.PS << std::endl;

        // std::cout << "UPDATE P PARAMETERS" << std::endl;

        // std::cout << "NEW PLASTIC PARAM" << std::endl;
        // std::cout << attrs.pmp.plastic_strain.transpose() << std::endl;
        // std::cout << attrs.pmp.kinematic_hardening_shift.transpose() << std::endl;        

    }

    // void UpdatePlasticParametersAlign(size_t elm_id,Plasti)


    // I don't think there is some fucking naive approach for defining the energy of plastic model, energy defination for damping model might be a good reference
    // We use semi-implicit formulation here, and assume the plastic parameters does not change in one step of time
    void ComputePsi(const TetAttributes& attrs,const Mat3x3d& F,FEM_Scaler& psi) const {
        Mat3x3d Feff = F - attrs.pmp.PS;
        if(attrs.pmp.failed)
            Feff = attrs.pmp.F_failed - attrs.pmp.PS;

        elastic_model->ComputePsi(attrs,Feff,psi);
    }

    void ComputePsiDeriv(const TetAttributes& attrs,
            const Mat3x3d& F,FEM_Scaler& psi,Vec9d& dpsi) const {
        if(attrs.pmp.failed){
            ComputePsi(attrs,F,psi);
            dpsi.setZero();
            return;
        }

        // Mat3x3d U,V;Vec3d total_strain;
        // DiffSVD::SVD_Decomposition(F,U,total_strain,V);  

        // Vec3d elastic_strain = total_strain - attrs.pmp.plastic_strain;
        // Mat3x3d Feff = U * elastic_strain.asDiagonal() * V.transpose();

        Mat3x3d Feff = F - attrs.pmp.PS;

        elastic_model->ComputePsiDeriv(attrs,Feff,psi,dpsi);
    }    
    // Compute the consistent Jacobi
    void ComputePsiDerivHessian(const TetAttributes& attrs,
            const Mat3x3d& F,FEM_Scaler& psi,Vec9d& dpsi,Mat9x9d& ddpsi,bool spd) const {
        if(attrs.pmp.failed){
            ComputePsi(attrs,F,psi);
            dpsi.setZero();
            ddpsi = Mat9x9d::Identity() * 1e-6;
            return;
        }

        // Mat3x3d U,V;Vec3d total_strain;
        // DiffSVD::SVD_Decomposition(F,U,total_strain,V);  

        // Vec3d elastic_strain = total_strain - attrs.pmp.plastic_strain;
        // Mat3x3d Feff = U * elastic_strain.asDiagonal() * V.transpose();

        Mat3x3d Feff = F - attrs.pmp.PS;

        elastic_model->ComputePsiDerivHessian(attrs,Feff,psi,dpsi,ddpsi,spd);   
    }

protected:
    // evaluate deviatoric stress in principal stress space
    inline Vec3d Dev(const Vec3d& s) const {return P * s;}
    // evaluate von mises
    FEM_Scaler EvalVM(const Vec3d& trial_stress,const Vec3d& pTrans,const Vec3d& init_stress,FEM_Scaler yield_stress) const {
        Vec3d eff_stress = trial_stress - pTrans - init_stress;
        Vec3d dev_stress = Dev(eff_stress);
        return sqrt(0.5 * dev_stress.squaredNorm()) - yield_stress;
    }

    // FEM_Scaler EvalVMSquared(const Vec3d& trial_stress,const Vec3d& pTrans,const Vec3d& init_stress,FEM_Scaler yield_stress) const {
    //     Vec3d eff_stress = trial_stress - pTrans - init_stress;
    //     Vec3d dev_stress = Dev(eff_stress);
    //     return 0.5 * dev_stress.squaredNorm()/yield_stress*yield_stress - 1.0;
    // }

    void EvalAssociativePFlow(const Vec3d& trial_stress,const Vec3d& pTrans,const Vec3d& init_stress,FEM_Scaler yield_stress,Vec3d& p_flow) const {
        // if(init_stress.norm() > 1e-8)
        //     throw std::runtime_error("INVALID INIT_STRESS");
        Vec3d eff_stress = trial_stress - pTrans - init_stress;
        Vec3d dev_stress = Dev(eff_stress);
        p_flow = 1/std::sqrt(0.5 * dev_stress.squaredNorm()) * PTP * eff_stress / 2;

        // std::cout << "PFLOW CMP : " << 1/std::sqrt(0.5 * dev_stress.squaredNorm()) << "\n" << PTP << std::endl << eff_stress.transpose() << "\t" << yield_stress << std::endl;
    }

    void EvalReturnMappingRes(const TetAttributes& attrs,const Vec3d& strain_total,const Vec3d& pflow,const Vec4d& x,Vec4d& res,FEM_Scaler restoring) const {
            assert(res.size() == 4);
            Vec3d strain_e = strain_total - attrs.pmp.plastic_strain - x[0] * pflow * restoring;
            Vec3d stress_trial;
            elastic_model->ComputePrincipalStress(attrs,strain_e,stress_trial);
            Vec3d alpha_trial = attrs.pmp.kinematic_hardening_shift + x.segment(1,3);
            res[0] =  EvalVM(stress_trial,alpha_trial,attrs.pmp.init_stress,attrs.pmp.yield_stress);
            // if(pmp.init_stress.norm() > 1e-8)
            //     throw std::runtime_error("INVALID INIT_STRESS");
            res.segment(1,3) = x.segment(1,3) - x[0] * attrs.pmp.kinematic_hardening_coeff * (stress_trial - alpha_trial - attrs.pmp.init_stress);
    }

    void EvalReturnMappingResJacobi(const TetAttributes& attrs,const Vec3d& strain_total,const Vec3d& pflow,const Vec4d& x,Vec4d& res,Mat4x4d& J,FEM_Scaler restoring) const {
            assert(res.size() == 4);
            Vec3d strain_e = strain_total - attrs.pmp.plastic_strain - x[0] * pflow * restoring;
            Vec3d stress_trial;
            Mat3x3d Hip;
            elastic_model->ComputePrincipalStressJacobi(attrs,strain_e,stress_trial,Hip);


            Vec3d alpha_trial = attrs.pmp.kinematic_hardening_shift + x.segment(1,3);
            res[0] =  EvalVM(stress_trial,alpha_trial,attrs.pmp.init_stress,attrs.pmp.yield_stress);
            // if(pmp.init_stress.norm() > 1e-8)
                // throw std::runtime_error("INVALID INIT_STRESS");
            res.segment(1,3) = x.segment(1,3) - x[0] * attrs.pmp.kinematic_hardening_coeff * (stress_trial - alpha_trial - attrs.pmp.init_stress);

            assert(J.rows() == 4 && J.cols() == 4);
    
            

            Vec3d stress_eff = stress_trial - alpha_trial - attrs.pmp.init_stress;

            FEM_Scaler tmp = sqrt(0.5 * stress_eff.transpose() * PTP * stress_eff);

            J.block(0,0,1,1) = -stress_eff.transpose() * (PTP * Hip) * pflow / (2 * tmp) * restoring;
            J.block(0,1,1,3) = -stress_eff.transpose() * PTP  / (2 * tmp);

            J.block(1,0,3,1) = -attrs.pmp.kinematic_hardening_coeff * stress_eff + x[0] * attrs.pmp.kinematic_hardening_coeff * Hip * pflow * restoring;
            J.block(1,1,3,3) = (1 + x[0] * attrs.pmp.kinematic_hardening_coeff) * Mat3x3d::Identity();
    }

    // Feff = F - U * (sp) * V'
    void EvaldFeffdF(const Mat3x3d& U,const Mat3x3d& V,const Vec3d& seff,const Vec3d& s,Mat9x9d& dFeffdF) const {
        // std::cout << "U : " << std::endl << U << std::endl << "V : " << V << std::endl;

        // Vec3d sp = s - seff;
        // for(size_t idx = 0;idx < 9;++idx){
        //     int k = idx / 3;
        //     int l = idx % 3;
        //     Mat3x3d dF = Mat3x3d::Zero();

        //     dF(k,l) = 1.0;
        //     Mat3x3d W = U.transpose() * dF * V;
        //     Mat3x3d dP = Mat3x3d::Zero();
        //     for(int i = 0;i < 3;++i)
        //         for(int j = 0;j < 3;++j){
        //             if(i == j)
        //                 continue;
        //             double wij = W(i,j);
        //             double wji = W(j,i);
        //             double si = s[i];
        //             double sj = s[j];
        //             double pi = sp[i];
        //             double pj = sp[j];
        //             if(fabs(si - sj) < 1e-6){
        //                 dP(i,j) = pj*wij - pi*wji;
        //                 dP(i,j) /= (sj + si);// TODOLIST handle degenerate case
        //             }else{
        //                 dP(i,j) = (sj*pj - si*pi)*wij + (si*pj - sj*pi)*wji;
        //                 dP(i,j) /= (sj+si)*(sj-si);
        //             }

        //             // dP(j,i) = dP(i,j);
        //         }
        //     dP = U * dP * V.transpose(); 
        //     dP = dP.transpose(); 

        //     dFeffdF.col(idx) = MatHelper::VEC(dP);
        // }

        // dFeffdF = Mat9x9d::Identity() - dFeffdF;


        Vec3d sp = s - seff;
        Mat3x3d dpds = Mat3x3d::Zero();
        for(size_t idx = 0;idx < 9;++idx){
            size_t k = idx % 3;size_t l = idx/3;
            Mat3x3d dF = Mat3x3d::Zero();
            dF(k,l) = 1.0;
            Vec3d ds = (U.transpose() * dF * V).diagonal();
            Vec3d dp = dpds * ds;
            Mat3x3d W = U.transpose() * dF * V;
            Mat3x3d dP = Mat3x3d::Zero();
            dP.diagonal() = dp;
            for(int i = 0;i < 3;++i)
                for(int j = 0;j < 3;++j){
                    if(i == j)
                        continue;
                    double wij = W(i,j);
                    double wji = W(j,i);
                    double si = s[i];
                    double sj = s[j];
                    double pi = sp[i];
                    double pj = sp[j];
                    if(fabs(s[i] - s[j]) < 1e-6){
                        double psij = dpds(i,j);
                        double psjj = dpds(j,j);
                        // dP(i,j) = (si*wij + sj*wji)*psjj + (wji*pj - wij*pi) - (wij*sj + wji*si)*psij;
                        dP(i,j) = (psjj*sj + pj - psij*si)*wij + (si*psjj - pi - sj*psij)*wji;
                        dP(i,j) /= (sj + si);// TODOLIST handle degenerate case
                    }else{
                        dP(i,j) = (sj*pj - si*pi)*wij + (si*pj - sj*pi)*wji;
                        dP(i,j) /= (sj+si)*(sj-si);
                    }
                }
            dP = U * dP * V.transpose();   

            dFeffdF.col(idx) =  MatHelper::VEC(dP);
        }

        dFeffdF = Mat9x9d::Identity() - dFeffdF;
    }

    std::shared_ptr<ElasticModel> elastic_model;
    Mat3x3d P;
    Mat3x3d PTP;
};