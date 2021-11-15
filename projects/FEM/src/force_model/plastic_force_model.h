#include <base_elastic_model.h>
#include <base_force_model.h>
#include <cmath>

struct PlasticMaterialParam{
    // ElastoMaterialParam emp;

    FEM_Scaler yield_stress;
    Vec3d init_stress;

    FEM_Scaler kinematic_hardening_coeff;
    Vec3d kinematic_hardening_shift;

    FEM_Scaler isotropic_hardening_coeff;
    
    Vec3d plastic_strain;

    FEM_Scaler restoring_strain;
    FEM_Scaler failed_strain;

    bool failed;
    Vec3d the_strain_failed;// the strain at the very first time the material break down
};


// the Joshph Terra approach for gaurantee SPD is needed? or not
class PlasticForceModel : public BaseForceModel {
public:
    PlasticForceModel(std::shared_ptr<ElasticModel> _elastic_model) : BaseForceModel(),elastic_model(_elastic_model) {
        P = Mat3x3d::Identity() - Mat3x3d::Constant(1.0/3);
        PTP = P.transpose() * P;
    }
    ~PlasticForceModel() {}

    // We explicitly update the plastic material parameter before the Newton-Raphson algorithm take place. And simplify the implementation of the algorithm
    void UpdatePlasticParameters(size_t elm_id,PlasticMaterialParam& pmp,const ElastoMaterialParam& emp,const Mat3x3d& F) const {
        if(pmp.failed)
            return;

        Mat3x3d U,V;Vec3d total_strain;
        DiffSVD::SVD_Decomposition(F,U,total_strain,V);  

        Vec3d trial_strain = total_strain - pmp.plastic_strain;
        Vec3d trial_stress;
        elastic_model->ComputePrincipalStress(emp,trial_strain,trial_stress);
        
        FEM_Scaler vm = EvalVM(trial_stress,pmp.kinematic_hardening_shift,pmp.init_stress,pmp.yield_stress);      

        // std::cout << "VM<" << elm_id << "> : \t" << vm << "\t" << pmp.yield_stress << std::endl;

        // the effective stress in the interior of yielding surface, only pure elastic behavior, no update is needed
        if(vm < 0)
            return;
        
        FEM_Scaler s01 = total_strain[0] - total_strain[1];
        FEM_Scaler s02 = total_strain[0] - total_strain[2];
        FEM_Scaler s12 = total_strain[1] - total_strain[2];

        FEM_Scaler rm = sqrt(0.5 * (s01*s01 + s02*s02 + s12*s12));


        if(rm > pmp.failed_strain){
            pmp.failed = true;
            pmp.the_strain_failed = total_strain;

            std::cout << "FAIL_ELM<" << elm_id << "> : " << total_strain.transpose() << std::endl;
            return;
        }

        // if the effective strain go beyond the restoring strain, the object return back to pure elastic behavior
        if(rm > pmp.restoring_strain && rm < pmp.failed_strain){
            std::cout << "<" << elm_id << ">rm = " << rm << "\t" << pmp.restoring_strain << std::endl;
            return;
        }



        // else{
        //     std::cout << "VM<" << elm_id << "> : \t" << vm << "\t" << pmp.yield_stress << std::endl;
        //     return;
        // }

        // std::cout << "UPDATE PLASTIC PARAM " << vm << std::endl;
        // std::cout << "ORI_PARAM : " << std::endl;
        // std::cout << pmp.plastic_strain.transpose() << std::endl;
        // std::cout << pmp.kinematic_hardening_shift.transpose() << std::endl;

        Vec3d p_flow;
        EvalAssociativePFlow(trial_stress,pmp.kinematic_hardening_shift,pmp.init_stress,pmp.yield_stress,p_flow);

        // std::cout << "PFLOW : " << p_flow.transpose() << std::endl;

        //solve the step using newthon-raphson algorithm
        Vec4d x;// set the initial solution for return mapping algorithm.....i.e [dstep, dpTrans]
        x << 0,pmp.kinematic_hardening_shift;


        bool debug = false;
        if(debug){
            std::cout << "DEBUG CODE : " << std::endl;
            Vec4d x_tmp = x;
            Vec4d res_cmp,res_tmp;
            Mat4x4d J_cmp,J_fd;
            EvalReturnMappingResJacobi(pmp,emp,total_strain,p_flow,x,res_cmp,J_cmp);

            for(size_t i = 0;i < 4;++i){
                FEM_Scaler step = x[i] * 1e-6;
                step = fabs(step) < 1e-6 ? 1e-6 : step;
                x_tmp = x;
                x_tmp[i] += step;

                EvalReturnMappingRes(pmp,emp,total_strain,p_flow,x_tmp,res_tmp);
                J_fd.col(i) = (res_tmp - res_cmp) / step;
            }

            std::cout << "J_cmp : " << std::endl << J_cmp << std::endl;
            std::cout << "J_fd : " << std::endl << J_fd << std::endl;

            // throw std::runtime_error("J_CHECK");
        }

        Vec4d res;
        Vec3d resp_stress = trial_stress;
        Mat4x4d Jacobi;
        size_t nm_iters = 0;
        size_t max_iters = 200;
        std::vector<FEM_Scaler> iter_buffer;
        iter_buffer.resize(max_iters);
        do{
            EvalReturnMappingResJacobi(pmp,emp,total_strain,p_flow,x,res,Jacobi);
            iter_buffer[nm_iters] = res.norm();
            if(res.norm() < 1e-2) 
                break;

            Vec4d dx = -Jacobi.inverse() * res;
            x += dx;
            ++nm_iters;
        }while(nm_iters < max_iters);

        if(nm_iters == max_iters){
            for(size_t i = 0;i < nm_iters;++i)
                std::cout << "IDX: " << i << "\t" << iter_buffer[i] << std::endl;
            throw std::runtime_error("RETURN_MAPPING_OVERFLOW");
        }

        pmp.plastic_strain += x[0] * p_flow; // dstrain_p = test_step * p_flow;
        pmp.kinematic_hardening_shift += x.segment(1,3);   


        // std::cout << "NEW PLASTIC PARAM" << std::endl;
        // std::cout << pmp.plastic_strain.transpose() << std::endl;
        // std::cout << pmp.kinematic_hardening_shift.transpose() << std::endl;        

    }

    // I don't think there is some fucking naive approach for defining the energy of plastic model, energy defination for damping model might be a good reference
    // We use semi-implicit formulation here, and assume the plastic parameters does not change in one step of time
    void ComputePsi(const PlasticMaterialParam& pmp,const ElastoMaterialParam& emp,const Mat3x3d& F,FEM_Scaler& psi) const {
        Mat3x3d U,V;Vec3d total_strain;
        DiffSVD::SVD_Decomposition(F,U,total_strain,V);  

        if(pmp.failed)
            total_strain = pmp.the_strain_failed;

        Vec3d effective_strain = total_strain - pmp.plastic_strain;
        Mat3x3d Feff = U * effective_strain.asDiagonal() * V.transpose();

        elastic_model->ComputePhi(emp,Feff,psi);
    }

    void ComputePsiDeriv(const PlasticMaterialParam& pmp,
            const ElastoMaterialParam& emp,
            const Mat3x3d& F,FEM_Scaler& psi,Vec9d& dpsi) const {
        if(pmp.failed){
            ComputePsi(pmp,emp,F,psi);
            dpsi.setZero();
            return;
        }

        Mat3x3d U,V;Vec3d total_strain;
        DiffSVD::SVD_Decomposition(F,U,total_strain,V);  

        Vec3d effective_strain = total_strain - pmp.plastic_strain;
        Mat3x3d Feff = U * effective_strain.asDiagonal() * V.transpose();

        elastic_model->ComputePhiDeriv(emp,Feff,psi,dpsi);
    }    
    // Compute the consistent Jacobi
    void ComputePsiDerivHessian(const PlasticMaterialParam& pmp,
            const ElastoMaterialParam& emp,
            const Mat3x3d& F,FEM_Scaler& psi,Vec9d& dpsi,Mat9x9d& ddpsi,bool spd) const {
        if(pmp.failed){
            ComputePsi(pmp,emp,F,psi);
            dpsi.setZero();
            ddpsi = Mat9x9d::Identity() * 1e-6;
            return;
        }

        Mat3x3d U,V;Vec3d total_strain;
        DiffSVD::SVD_Decomposition(F,U,total_strain,V);  

        Vec3d effective_strain = total_strain - pmp.plastic_strain;
        Mat3x3d Feff = U * effective_strain.asDiagonal() * V.transpose();

        elastic_model->ComputePhiDerivHessian(emp,Feff,psi,dpsi,ddpsi,spd);   

        // std::cout << "PARAM_CHECK : " << std::endl;
        // std::cout << "E : " << emp.E << std::endl;
        // std::cout << "nu : " << emp.nu << std::endl;
        // std::cout << "Act : " << std::endl << emp.Act << std::endl;
        // std::cout << "forient : " << std::endl << emp.forient.transpose() << std::endl;
        // std::cout << "pstrain : " << pmp.plastic_strain.transpose() << std::endl;

        // throw std::runtime_error("PARAM_CHECK");
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
        Vec3d eff_stress = trial_stress - pTrans - init_stress;
        Vec3d dev_stress = Dev(eff_stress);
        p_flow = 1/std::sqrt(0.5 * dev_stress.squaredNorm()) * PTP * eff_stress / 2;

        // std::cout << "PFLOW CMP : " << 1/std::sqrt(0.5 * dev_stress.squaredNorm()) << "\n" << PTP << std::endl << eff_stress.transpose() << "\t" << yield_stress << std::endl;
    }

    void EvalReturnMappingRes(const PlasticMaterialParam& pmp,const ElastoMaterialParam& emp,const Vec3d& strain_total,const Vec3d& pflow,const Vec4d& x,Vec4d& res) const {
            assert(res.size() == 4);
            Vec3d strain_e = strain_total - pmp.plastic_strain - x[0] * pflow;
            Vec3d stress_trial;
            elastic_model->ComputePrincipalStress(emp,strain_e,stress_trial);
            Vec3d alpha_trial = pmp.kinematic_hardening_shift + x.segment(1,3);
            res[0] =  EvalVM(stress_trial,alpha_trial,pmp.init_stress,pmp.yield_stress);
            res.segment(1,3) = x.segment(1,3) - x[0] * pmp.kinematic_hardening_coeff * (stress_trial - alpha_trial - pmp.init_stress);
    }

    void EvalReturnMappingResJacobi(const PlasticMaterialParam& pmp,const ElastoMaterialParam& emp,const Vec3d& strain_total,const Vec3d& pflow,const Vec4d& x,Vec4d& res,Mat4x4d& J) const {
            assert(res.size() == 4);
            Vec3d strain_e = strain_total - pmp.plastic_strain - x[0] * pflow;
            Vec3d stress_trial;
            elastic_model->ComputePrincipalStress(emp,strain_e,stress_trial);
            Vec3d alpha_trial = pmp.kinematic_hardening_shift + x.segment(1,3);
            res[0] =  EvalVM(stress_trial,alpha_trial,pmp.init_stress,pmp.yield_stress);
            res.segment(1,3) = x.segment(1,3) - x[0] * pmp.kinematic_hardening_coeff * (stress_trial - alpha_trial - pmp.init_stress);

            assert(J.rows() == 4 && J.cols() == 4);
            Mat3x3d Hip;
            elastic_model->ComputePrincipalStressJacobi(emp,strain_e,stress_trial,Hip);

            Vec3d stress_eff = stress_trial - alpha_trial - pmp.init_stress;

            FEM_Scaler tmp = sqrt(0.5 * stress_eff.transpose() * PTP * stress_eff);

            J.block(0,0,1,1) = -stress_eff.transpose() * (PTP * Hip) * pflow / (2 * tmp);
            J.block(0,1,1,3) = -stress_eff.transpose() * PTP  / (2 * tmp);

            J.block(1,0,3,1) = -pmp.kinematic_hardening_coeff * stress_eff + x[0] * pmp.kinematic_hardening_coeff * Hip * pflow;
            J.block(1,1,3,3) = (1 + x[0] * pmp.kinematic_hardening_coeff) * Mat3x3d::Identity();
    }

    std::shared_ptr<ElasticModel> elastic_model;
    Mat3x3d P;
    Mat3x3d PTP;
};