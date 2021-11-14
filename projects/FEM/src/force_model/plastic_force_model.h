#include <base_elastic_model.h>
#include <base_force_model.h>
#include <cmath>

struct PlasticMaterialParam{
    ElastoMaterialParam emp;

    FEM_Scaler yield_stress;
    Vec3d init_stress;

    FEM_Scaler kinematic_hardening_coeff;
    Vec3d kinematic_hardening_shift;

    FEM_Scaler isotropic_hardening_coeff;
    
    Vec3d plastic_strain;
};

class PlasticForceModel : public BaseForceModel {
public:
    PlasticForceModel(std::shared_ptr<ElasticModel> _elastic_model) : BaseForceModel(),elastic_model(_elastic_model) {
        P = Mat3x3d::Identity() - Mat3x3d::Constant(1.0/3);
    }
    ~PlasticForceModel() {}

    void ComputeStress(const PlasticMaterialParam& pmp,const Mat3x3d& F,Vec9d& stress,Vec3d& dsp,Vec3d& dalpha) const {
            Mat3x3d U,V;Vec3d total_strain;
            DiffSVD::SVD_Decomposition(F,U,total_strain,V);
            Vec3d trial_strain = total_strain - pmp.plastic_strain;
            Vec3d trial_stress;
            elastic_model->ComputePrincipalStress(pmp.emp,trial_strain,trial_stress);
            
            Vec3d p_flow;
            FEM_Scaler vm = EvalVM(trial_stress,pmp.kinematic_hardening_shift,pmp.init_stress,pmp.yield_stress);
            Vec3d resp_stress;
            if(vm <= 0){
                resp_stress = trial_stress;
                dsp.setZero();
                dalpha.setZero();
            }else{
                Vec3d p_flow;
                EvalAssociativePFlow(trial_stress,pmp.kinematic_hardening_shift,pmp.init_stress,pmp.yield_stress,p_flow);
                //solve the step using newthon-raphson algorithm
                Vec4d x;// set the initial solution for return mapping algorithm.....i.e [step, pTrans]
                x << 0,pmp.kinematic_hardening_shift;
                Vec4d res;
                Vec3d resp_stress = trial_stress;
                Mat4x4d Jacobi;
                size_t nm_iters = 0;
                size_t max_iters = 10;
                do{
                    EvalReturnMappingResJacobi(pmp,total_strain,p_flow,x,res,Jacobi);
                    if(res.norm() < 1e-6)
                        break;

                    Vec4d dx = -Jacobi.inverse() * res;
                    x += dx;
                }while(nm_iters < max_iters);

                dsp = x[0] * p_flow; // dstrain_p = test_step * p_flow;
                dalpha = x.segment(1,3) - pmp.kinematic_hardening_shift;

                elastic_model->ComputePrincipalStress(pmp.emp,trial_strain - dsp,trial_stress);

                FEM_Scaler vm_tmp = EvalVM(trial_stress,pmp.kinematic_hardening_shift + dalpha,pmp.init_stress,pmp.yield_stress);    
            }

            Mat3x3d P = U * resp_stress.asDiagonal() * V.transpose();
            stress = MatHelper::VEC(P);
    }    

    void ComputeResponseStressHessian() const {
        throw std::runtime_error("Hessian not implemented yet");
    }

protected:
    // evaluate deviatoric stress in principal stress space
    inline Vec3d Dev(const Vec3d& s) const {return P * s;}
    // evaluate von mises
    FEM_Scaler EvalVM(const Vec3d& trial_stress,const Vec3d& pTrans,const Vec3d& init_stress,FEM_Scaler yield_stress) const {
        Vec3d eff_stress = trial_stress - pTrans - init_stress;
        FEM_Scaler three_second = 3.0/2;
        return sqrt(three_second * eff_stress.transpose() * P * eff_stress) - yield_stress;
    }

    FEM_Scaler EvalVMSquared(const Vec3d& trial_stress,const Vec3d& pTrans,const Vec3d& init_stress,FEM_Scaler yield_stress) const {
        Vec3d eff_stress = trial_stress - pTrans - init_stress;
        FEM_Scaler three_second = 3.0/2;
        return three_second * eff_stress.transpose() * P * eff_stress - yield_stress*yield_stress;
    }

    void EvalAssociativePFlow(const Vec3d& trial_stress,const Vec3d& pTrans,const Vec3d& init_stress,FEM_Scaler yield_stress,Vec3d& p_flow) const {
        Vec3d eff_stress = trial_stress - pTrans - init_stress;
        FEM_Scaler ds_norm = eff_stress.transpose() * P * eff_stress;
        p_flow = 1/std::sqrt(2*ds_norm/3) * P * eff_stress;
    }

    std::shared_ptr<ElasticModel> elastic_model;

    FEM_Scaler compute_return_mapping(const Vec3d& pTrans,FEM_Scaler yield_stress,const Vec3d& init_stress,const Vec3d& trial_strain,const Vec3d& pflow) const {

    }

    // the residual of return mapping algorithm for solving appropriate 'step' and 'p
    void EvalReturnMappingRes(const PlasticMaterialParam& pmp,const Vec3d& strain_total,const Vec3d& pflow,const Vec4d& x,Vec4d& res) const {
        assert(res.size() == 4);
        Vec3d strain_e = strain_total - pmp.plastic_strain - x[0] * pflow;
        Vec3d stress_trial;
        elastic_model->ComputePrincipalStress(pmp.emp,strain_e,stress_trial);
        Vec3d alpha_trial = pmp.kinematic_hardening_shift + x.segment(1,3);
        res[0] =  EvalVMSquared(stress_trial,alpha_trial,pmp.init_stress,pmp.yield_stress);
        res.segment(1,3) = alpha_trial - x[0] * pmp.kinematic_hardening_coeff * (stress_trial - alpha_trial - pmp.init_stress);
    }

    void EvalReturnMappingResJacobi(const PlasticMaterialParam& pmp,const Vec3d& strain_total,const Vec3d& pflow,const Vec4d& x,Vec4d& res,Mat4x4d& J) const {
            assert(res.size() == 4);
            Vec3d strain_e = strain_total - pmp.plastic_strain - x[0] * pflow;
            Vec3d stress_trial;
            elastic_model->ComputePrincipalStress(pmp.emp,strain_e,stress_trial);
            Vec3d alpha_trial = pmp.kinematic_hardening_shift + x.segment(1,3);
            res[0] =  EvalVMSquared(stress_trial,alpha_trial,pmp.init_stress,pmp.yield_stress);
            res.segment(1,3) = alpha_trial - x[0] * pmp.kinematic_hardening_coeff * (stress_trial - alpha_trial - pmp.init_stress);

            assert(J.rows() == 4 && J.cols() == 4);
            Mat3x3d Hip;
            elastic_model->ComputePrincipalStressJacobi(pmp.emp,strain_e,stress_trial,Hip);

            Vec3d stress_eff = stress_trial - x.segment(1,3) - pmp.init_stress;
            J(0,0) = -3 * (stress_eff).transpose() * P * Hip * pflow;
            J.block(0,1,1,3) = 3 * stress_eff.transpose() * P;
            J.block(1,0,3,1) = -pmp.kinematic_hardening_coeff * stress_eff + x[0] * pmp.kinematic_hardening_coeff * Hip * pflow;
            J.block(1,1,3,3) = (1 - x[0] * pmp.kinematic_hardening_coeff) * Mat3x3d::Identity();
    }

    Mat3x3d P;
};