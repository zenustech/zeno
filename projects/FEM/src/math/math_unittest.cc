#include "differentiable_SVD.h"
#include "dfft.h"
#include <cstdlib>
#include <strstream>

// TESTING ON DifferentialSVD Module
#include "gtest/gtest.h"


const unsigned int SEED = 3141592653;

namespace SVD_UNITTEST_NSP{

class SVDTest : public testing::Test {
protected:
    void SetUp() override{
        srand(SEED);

        for (size_t i = 0; i < 10; ++i)
            Fs.push_back(Mat3x3d::Random());

        Fs.push_back(Vec3d(1, 2, 3).asDiagonal());
        Fs.push_back(Vec3d(1, 2, -3).asDiagonal());

        Fs.push_back(Vec3d(1, 1, 2).asDiagonal());
        Fs.push_back(Vec3d(1, 1, 1 + 1e-6).asDiagonal());
        Fs.push_back(Vec3d(1, 1, -1 + 1e-6).asDiagonal());

        Fs.push_back(Vec3d(1, 1, 1).asDiagonal());

        Fs.push_back(Mat3x3d::Zero());

        Mat3x3d A_tmp = Mat3x3d::Random(3, 3);
        A_tmp = A_tmp * A_tmp.transpose();
        Fs_neg.push_back(A_tmp * Vec3d(1, 1, -2).asDiagonal());
        Fs_neg.push_back(Vec3d(1, 2, -5).asDiagonal());
    }

    // virtual void TearDown() will be called after each test is run.
    // You should define it if there is cleanup work to do.  Otherwise,
    // you don't have to provide it.
    //
    // virtual void TearDown() {
    // }

    std::vector<Mat3x3d> Fs;
    std::vector<Mat3x3d> Fs_neg;

    void SYM_ED_ACCURAVY(const Mat3x3d& sym_F) const;
    void SVD_ACCURAVY(const Mat3x3d& F) const;
    void SVD_NEGATIVE_SINGULAR(const Mat3x3d& F) const;
};

TEST_F(SVDTest,SYM_ED_ACCURACY){
    for (size_t k = 0; k < Fs.size(); ++k) {
        Mat3x3d F = Fs[k];
        Mat3x3d sym_F = 0.5 * (F + F.transpose());
        SYM_ED_ACCURAVY(sym_F);
    }
}

TEST_F(SVDTest, SVD_ACCURACY) {
    for (size_t k = 0; k < Fs.size(); ++k) {
        Mat3x3d F = Fs[k];
        SVD_ACCURAVY(F);
    }
}

TEST_F(SVDTest, SVD_NEGATIVE_SINGULAR) {
    for (size_t k = 0; k < Fs_neg.size(); ++k) {
        Mat3x3d F = Fs_neg[k];
        SVD_NEGATIVE_SINGULAR(F);
    }
}


void SVDTest::SYM_ED_ACCURAVY(const Mat3x3d& sym_F) const {
    Mat3x3d eigen_vecs;
    Vec3d eigen_vals;
    DiffSVD::SYM_Eigen_Decomposition(sym_F, eigen_vals, eigen_vecs);

    // testing the whether the rand-one update of eigen vectors form the original matrix
    Mat3x3d sym_F_rec = Mat3x3d::Zero();
    for (size_t i = 0; i < 3; ++i)
        sym_F_rec += eigen_vals[i] * eigen_vecs.col(i) * eigen_vecs.col(i).transpose();

    FEM_Scaler sym_F_error = (sym_F_rec - sym_F).norm();
    EXPECT_NEAR(sym_F_error, 0, 1e-6) << "SYM_F_REC:" << sym_F_error << "\t" << 1e-6;

    // testing whether Ax = l * x hold
    Mat3x3d eigen_vecs_product = sym_F * eigen_vecs;
    for (size_t i = 0; i < 3; ++i) {
        std::stringstream msg;

        FEM_Scaler eigen_vec_error = (eigen_vecs.col(i) * eigen_vals[i] - eigen_vecs_product.col(i)).norm();
        msg.clear();
        msg << "SYM_EIGEN_ERROR : " << i << "\t" << eigen_vec_error << "\t" << eigen_vals[i] << std::endl;
        for (size_t j = 0; j < 3; ++j) {
            msg << eigen_vecs(j, i) << "\t" << eigen_vecs_product(j, i) << std::endl;
        }
        EXPECT_NEAR(eigen_vec_error, 0, 1e-6) << msg.str();
    }
}

void SVDTest::SVD_ACCURAVY(const Mat3x3d& F) const {
    Mat3x3d U, V;
    Vec3d s;
    DiffSVD::SVD_Decomposition(F, U, s, V);
    Mat3x3d UsVT = U * s.asDiagonal() * V.transpose();

    FEM_Scaler svd_error = (UsVT - F).norm();
    EXPECT_NEAR(svd_error, 0, 1e-6);
}

void SVDTest::SVD_NEGATIVE_SINGULAR(const Mat3x3d& F) const {
    Mat3x3d U, V;
    Vec3d s;
    DiffSVD::SVD_Decomposition(F, U, s, V);

    EXPECT_GE(abs(s[0]), abs(s[1])) << "abs((s[0]) < abs(s[1]) : " << s[0] << "\t" << s[1];
    EXPECT_GE(abs(s[1]), abs(s[2])) << "abs(s[1]) < abs(s[2]) : " << s[1] << "\t" << s[2];
    EXPECT_GE(s[0], 0) << "s[0] < 0 : " << s[0] << std::endl;
    EXPECT_GE(s[1], 0) << "s[1] < 0 : " << s[1] << std::endl;
    EXPECT_LE(s[2], 0) << "s[2] > 0 : " << s[2] << std::endl;
}

};

namespace DFT_UNITTEST_NSP{

class DFTTest : public testing::Test {
protected:
    void SetUp() override {
        Nh = 2;
        N = 5 * Nh + 1;
        dim = 2;

        traj.resize(N, VecXd::Zero(dim));
        spec = VecXd::Random((2 * Nh + 1) * dim);
        omega = 1;
    }
protected:
    FEM_Scaler omega;
    size_t Nh;
    size_t N;
    size_t dim;
    std::vector<VecXd> traj;
    VecXd spec;
};

TEST_F(DFTTest,TEST_MAPPING) {
    MatXd Spec2Sig(N, Nh * 2 + 1);
    // auto spec_map = Eigen::Map<MatXd>(spec.data(),2 * Nh + 1, dim);

    for (size_t i = 0; i < N; ++i) {
        FEM_Scaler t = 2 * DFT::PI * i / N / omega;
        for (size_t j = 0; j < Nh * 2 + 1; ++j)
            Spec2Sig(i, j) = DFT::EvalQ(j,t,omega);
    }

    MatXd Sig2Spec(Nh * 2 + 1, N);
    for (size_t i = 0; i < Nh * 2 + 1; ++i)
        for (size_t j = 0; j < N; ++j)
            Sig2Spec(i, j) = DFT::EvalQinv(i, j, N);


    MatXd should_be_identity = Sig2Spec * Spec2Sig;
    MatXd Identity_ref = MatXd::Identity(Nh * 2 + 1, Nh * 2 + 1);
    FEM_Scaler map_error = (should_be_identity - Identity_ref).norm();
    EXPECT_NEAR(map_error, 0, 1e-6) << "\nREF:\n" <<  Identity_ref << "\nCMP:\n" << should_be_identity << std::endl;
}


TEST_F(DFTTest, TEST_EVALDFT_IDFT) {
    VecXd restart_frame(dim);
    for (size_t i = 0; i < N + 1; ++i) {
        FEM_Scaler t = 2 * DFT::PI * i / N / omega;
        DFT::EvalIDFT(i == N ? restart_frame : traj[i], spec, t, omega);
    }

    FEM_Scaler cycle_error = (restart_frame - traj[0]).norm();
    EXPECT_NEAR(cycle_error, 0, 1e-6) << "cycle_error : " << cycle_error;

    VecXd spec_rec((2 * Nh + 1) * dim);
    DFT::EvalDFT(traj, spec_rec, Nh * 2 + 1);

    FEM_Scaler spec_error = (spec - spec_rec).norm();
    EXPECT_NEAR(spec_error, 0, 1e-6) << "\nSPEC:\n" << spec.transpose() << "\nSPEC_REC:\n" << spec_rec.transpose() << std::endl;
}

TEST_F(DFTTest,TEST_TIMENODE) {
    for(size_t i = 0;i < N;++i)
        EXPECT_NEAR(DFT::GetTimeNode(i,N,1),2*DFT::PI * i / N,1e-6);
}

};
