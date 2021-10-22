#pragma

#include "base_force_model.h"


class StableAnisotropicMuscle : public BaseForceModel {
public:
    StableAnisotropicMuscle() : BaseForceModel() {
        isoQs[0] <<   1, 0, 0,
                0, 0, 0,
                0, 0, 0;
        isoQs[1]  <<   0, 0, 0,
                0, 1, 0,
                0, 0, 0;
        isoQs[2]  <<   0, 0, 0,
                0, 0, 0,
                0, 0, 1;

        isoQs[3]  <<   0,-1, 0,
                1, 0, 0,
                0, 0, 0;
        isoQs[3]  /= sqrt(2);
    
        isoQs[4]  <<   0, 0, 0,
                0, 0, 1,
                0,-1, 0;
        isoQs[4]  /= sqrt(2);

        isoQs[5]  <<   0, 0, 1,
                0, 0, 0,
                -1,0, 0;
        isoQs[5]  /= sqrt(2);

        isoQs[6]  <<   0, 1, 0,
                1, 0, 0,
                0, 0, 0;
        isoQs[6]  /= sqrt(2);

        isoQs[7]  <<   0, 0, 0,
                0, 0, 1,
                0, 1, 0;
        isoQs[7]  /= sqrt(2);

        isoQs[8]  <<   0, 0, 1,
                0, 0, 0,
                1, 0, 0;
        isoQs[8]  /= sqrt(2);      

        for(size_t i = 0;i < 9;++i)
            anisoQs[i].setZero();
    }
    /**
     * @brief destructor method.
     */
    virtual ~StableAnisotropicMuscle(){}

    inline FEM_Scaler EvalAnisotropicInvarient(const Vec3d &a, const Mat3x3d& F) {
        Vec3d fa = F * a;
        FEM_Scaler Ia = fa.squaredNorm();
        return Ia;
    }

    inline FEM_Scaler EvalAnisotropicInvarientDeriv(const Vec3d& a,const Mat3x3d& F,Vec9d& g) {
        Vec3d fa = F * a;
        FEM_Scaler Ia = fa.squaredNorm();
        g = 2 * MatHelper::VEC(F * MatHelper::DYADIC(a,a));

        return Ia;
    }

    void ComputeIsoEigenSystem(FEM_Scaler YoungModulus,FEM_Scaler PossonRatio,
            const Mat3x3d& F,
            Vec9d& eigen_vals,
            Vec9d eigen_vecs[9]) const = 0;  

    // only anisotropic model need reimplementing these method
    void ComputeAnisoEigenSystem(FEM_Scaler YoungModulus,FEM_Scaler PossonRatio,
            const Mat3x3d& F,
            const Vec3d& a,
            Vec9d& eigen_vals,
            Vec9d eigen_vecs[9]) const override {

    }  


private:
    Mat3x3d isoQs[9];// 3 stretchings + 3 twisted + 3 reflecting
    Mat3x3d anisoQs[9];// three fiber direction x 3 eigen matrices for each
};