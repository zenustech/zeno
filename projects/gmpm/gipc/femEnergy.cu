#include "femEnergy.cuh"
#include "math.h"
#include <stdio.h>
__device__ __host__
void __calculateDms3D_double(const double3* vertexes, const uint4& index, __GEIGEN__::Matrix3x3d& M) {

    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;

    double o3x = vertexes[index.w].x - vertexes[index.x].x;
    double o3y = vertexes[index.w].y - vertexes[index.x].y;
    double o3z = vertexes[index.w].z - vertexes[index.x].z;

    M.m[0][0] = o1x; M.m[0][1] = o2x; M.m[0][2] = o3x;
    M.m[1][0] = o1y; M.m[1][1] = o2y; M.m[1][2] = o3y;
    M.m[2][0] = o1z; M.m[2][1] = o2z; M.m[2][2] = o3z;
}

__device__
__GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D_double(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, double lengthRate, double volumRate) {

    double I3 = Sigma.m[0][0] * Sigma.m[1][1] * Sigma.m[2][2];
    double I2 = Sigma.m[0][0] * Sigma.m[0][0] + Sigma.m[1][1] * Sigma.m[1][1] + Sigma.m[2][2] * Sigma.m[2][2];

    double u = lengthRate, r = volumRate;
    __GEIGEN__::Matrix3x3d pI3pF;

    pI3pF.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
    pI3pF.m[0][1] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
    pI3pF.m[0][2] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];

    pI3pF.m[1][0] = F.m[2][1] * F.m[0][2] - F.m[2][2] * F.m[0][1];
    pI3pF.m[1][1] = F.m[2][2] * F.m[0][0] - F.m[2][0] * F.m[0][2];
    pI3pF.m[1][2] = F.m[2][0] * F.m[0][1] - F.m[2][1] * F.m[0][0];

    pI3pF.m[2][0] = F.m[0][1] * F.m[1][2] - F.m[1][1] * F.m[0][2];
    pI3pF.m[2][1] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
    pI3pF.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];


    //printf("volRate and LenRate:  %f    %f\n", volumRate, lengthRate);

    __GEIGEN__::Matrix3x3d PEPF, tempA, tempB;
    tempA = __GEIGEN__::__S_Mat_multiply(F, u * (1 - 1 / (I2 + 1)));
    tempB = __GEIGEN__::__S_Mat_multiply(pI3pF, (r * (I3 - 1 - u * 3 / (r * 4))));
    __GEIGEN__::__Mat_add(tempA, tempB, PEPF);
    return PEPF;
}

__device__
__GEIGEN__::Matrix3x3d computePEPF_ARAP_double(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate)
{
    __GEIGEN__::Matrix3x3d R, S;

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, Sigma), __GEIGEN__::__Transpose3x3(V));//V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d g = __GEIGEN__::__Mat3x3_minus(F, R);
    return __GEIGEN__::__S_Mat_multiply(g, lengthRate);//lengthRate * g;
}

__device__
__GEIGEN__::Matrix9x9d project_ARAP_H_3D(const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate)
{
    __GEIGEN__::Matrix3x3d R, S;

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, Sigma), __GEIGEN__::__Transpose3x3(V));//V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d T0, T1, T2;

    __GEIGEN__::__set_Mat_val(T0, 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(T1, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(T2, 0, 0, 1, 0, 0, 0, -1, 0, 0);

    double ml = 1 / sqrt(2.0);

    __GEIGEN__::Matrix3x3d VTransp = __GEIGEN__::__Transpose3x3(V);

    T0 = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(U, T0), VTransp), ml);
    T1 = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(U, T1), VTransp), ml);
    T2 = __GEIGEN__::__S_Mat_multiply(__GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(U, T2), VTransp), ml);

    __GEIGEN__::Vector9 t0, t1, t2;
    t0 = __GEIGEN__::__Mat3x3_to_vec9_double(T0);
    t1 = __GEIGEN__::__Mat3x3_to_vec9_double(T1);
    t2 = __GEIGEN__::__Mat3x3_to_vec9_double(T2);

    double sx = Sigma.m[0][0];
    double sy = Sigma.m[1][1];
    double sz = Sigma.m[2][2];
    double lambda0 = 2 / (sx + sy);
    double lambda1 = 2 / (sz + sy);
    double lambda2 = 2 / (sx + sz);

    if (sx + sy < 2)lambda0 = 1;
    if (sz + sy < 2)lambda1 = 1;
    if (sx + sz < 2)lambda2 = 1;

    __GEIGEN__::Matrix9x9d SH, M9_temp;
    __GEIGEN__::__identify_Mat9x9(SH);
    __GEIGEN__::Vector9 V9_temp;


    M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(t0, t0);
    M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda0);
    SH = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(t1, t1);
    M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda1);
    SH = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(t2, t2);
    M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, -lambda2);
    SH = __GEIGEN__::__Mat9x9_add(SH, M9_temp);

    return __GEIGEN__::__S_Mat9x9_multiply(SH, lengthRate);;
}

__device__ 
__GEIGEN__::Matrix9x9d __project_StabbleNHK_H_3D(const __GEIGEN__::Matrix3x3d& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate, const double& volumRate) {

    double I3 = sigma.m[0][0] * sigma.m[1][1] * sigma.m[2][2];
    double I2 = sigma.m[0][0] * sigma.m[0][0] + sigma.m[1][1] * sigma.m[1][1] + sigma.m[2][2] * sigma.m[2][2];
    double g2 = sigma.m[0][0] * sigma.m[1][1] * sigma.m[0][0] * sigma.m[1][1] +
        sigma.m[0][0] * sigma.m[2][2] * sigma.m[0][0] * sigma.m[2][2] +
        sigma.m[2][2] * sigma.m[1][1] * sigma.m[2][2] * sigma.m[1][1];

    double u = lengthRate, r = volumRate;

    double n = 2 * u / ((I2 + 1) * (I2 + 1) * (r * (I3 - 1) - 3 * u / 4));
    double p = r / (r * (I3 - 1) - 3 * u / 4);
    double c2 = -g2 * p - I2 * n;
    double c1 = -(1 + 2 * I3 * p) * I2 - 6 * I3 * n + (g2 * I2 - 9 * I3 * I3) * p * n;
    double c0 = -(2 + 3 * I3 * p) * I3 + (I2 * I2 - 4 * g2) * n + 2 * I3 * p * n * (I2 * I2 - 3 * g2);

    double roots[3] = { 0 };
    int num_solution = 0;
    __GEIGEN__::__NewtonSolverForCubicEquation(1, c2, c1, c0, roots, num_solution, 1e-6);
    
    if (num_solution < 3) {
        //printf("================%d\n", num_solution);
        //printf("================\n====================\n=================\n===================\n================\n====================\n==================\n=================\n");
    }

    __GEIGEN__::Matrix3x3d D[3], M_temp[3];
    double q[3];
    __GEIGEN__::Matrix3x3d Q[9];
    double lamda[9];
    double Ut = u * (1 - 1 / (I2 + 1));
    double alpha = 1 + 3 * u / r / 4;

    //sigma.m[0][0] * sigma.m[1][1] * sigma.m[2][2]
    for (int i = 0; i < num_solution; i++) {
        double alpha0 = roots[i] * (sigma.m[1][1] + sigma.m[0][0] * sigma.m[2][2] * n + I3 * sigma.m[1][1] * p) +
            sigma.m[0][0] * sigma.m[2][2] + sigma.m[1][1] * (sigma.m[0][0] * sigma.m[0][0] - sigma.m[1][1] * sigma.m[1][1] + sigma.m[2][2] * sigma.m[2][2]) * n +
            I3 * sigma.m[0][0] * sigma.m[2][2] * p +
            sigma.m[0][0] * (sigma.m[0][0] * sigma.m[0][0] - sigma.m[1][1] * sigma.m[1][1]) * sigma.m[2][2] *
            (sigma.m[1][1] * sigma.m[1][1] - sigma.m[2][2] * sigma.m[2][2]) * p * n;

        double alpha1 = roots[i] * (sigma.m[0][0] + sigma.m[1][1] * sigma.m[2][2] * n + I3 * sigma.m[0][0] * p) +
            sigma.m[1][1] * sigma.m[2][2] - sigma.m[0][0] * (sigma.m[0][0] * sigma.m[0][0] - sigma.m[1][1] * sigma.m[1][1] - sigma.m[2][2] * sigma.m[2][2]) * n +
            I3 * sigma.m[1][1] * sigma.m[2][2] * p -
            sigma.m[1][1] * (sigma.m[0][0] * sigma.m[0][0] - sigma.m[1][1] * sigma.m[1][1]) * sigma.m[2][2] *
            (sigma.m[0][0] * sigma.m[0][0] - sigma.m[2][2] * sigma.m[2][2]) * p * n;

        double alpha2 = roots[i] * roots[i] - roots[i] * (sigma.m[0][0] * sigma.m[0][0] + sigma.m[1][1] * sigma.m[1][1]) * (n + sigma.m[2][2] * sigma.m[2][2] * p) -
            sigma.m[2][2] * sigma.m[2][2] - 2 * I3 * n - 2 * I3 * sigma.m[2][2] * sigma.m[2][2] * p +
            ((sigma.m[0][0] * sigma.m[0][0] - sigma.m[1][1] * sigma.m[1][1]) * sigma.m[2][2]) * ((sigma.m[0][0] * sigma.m[0][0] - sigma.m[1][1] * sigma.m[1][1]) * sigma.m[2][2]) * p * n;

        q[i] = 1 / sqrt(alpha0 * alpha0 + alpha1 * alpha1 + alpha2 * alpha2);
        __GEIGEN__::__set_Mat_val(D[i], alpha0, 0, 0, 0, alpha1, 0, 0, 0, alpha2);
        __GEIGEN__::__M_Mat_multiply(U, D[i], M_temp[0]);
        M_temp[1] = __Transpose3x3(V);
        __GEIGEN__::__M_Mat_multiply(M_temp[0], M_temp[1], M_temp[2]);
        Q[i] = __GEIGEN__::__S_Mat_multiply(M_temp[2], q[i]);
        //Q[i] = q[i] * U * D[i] * V.transpose();
        lamda[i] = r * (I3 - alpha) * roots[i] + Ut;
    }

    lamda[3] = Ut + sigma.m[2][2] * r * (I3 - alpha);
    lamda[4] = Ut + sigma.m[0][0] * r * (I3 - alpha);
    lamda[5] = Ut + sigma.m[1][1] * r * (I3 - alpha);

    lamda[6] = Ut - sigma.m[2][2] * r * (I3 - alpha);
    lamda[7] = Ut - sigma.m[0][0] * r * (I3 - alpha);
    lamda[8] = Ut - sigma.m[1][1] * r * (I3 - alpha);

    __GEIGEN__::__set_Mat_val(Q[3], 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[4], 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(Q[5], 0, 0, 1, 0, 0, 0, -1, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[6], 0, 1, 0, 1, 0, 0, 0, 0, 0);
    __GEIGEN__::__set_Mat_val(Q[7], 0, 0, 0, 0, 0, 1, 0, 1, 0);
    __GEIGEN__::__set_Mat_val(Q[8], 0, 0, 1, 0, 0, 0, 1, 0, 0);

    double ml = 1 / sqrt(2.0);

    M_temp[1] = __GEIGEN__::__Transpose3x3(V);
    for (int i = 3;i < 9;i++) {
        //Q[i] = ml* U* Q[i] * V.transpose();
        __GEIGEN__::__M_Mat_multiply(U, Q[i], M_temp[0]);

        __GEIGEN__::__M_Mat_multiply(M_temp[0], M_temp[1], M_temp[2]);
        Q[i] = __GEIGEN__::__S_Mat_multiply(M_temp[2], ml);
    }

    __GEIGEN__::Matrix9x9d H, M9_temp[2];
    __GEIGEN__::__init_Mat9x9(H, 0);
    __GEIGEN__::Vector9 V9_temp;
    for (int i = 0; i < 9; i++) {
        if (i < 3) {
            if (i >= num_solution) continue;
        }
        if (lamda[i] > 0) {
            V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q[i]);
            M9_temp[0] = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
            M9_temp[1] = __GEIGEN__::__S_Mat9x9_multiply(M9_temp[0], lamda[i]);
            H = __GEIGEN__::__Mat9x9_add(H, M9_temp[1]);
        }
    }

    return H;
}



__device__ 
__GEIGEN__::Matrix9x9d __project_ANIOSI5_H_3D(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double3& fiber_direction, const double& scale, const double& contract_length) {
    double3 direction = __GEIGEN__::__normalized(fiber_direction);
    __GEIGEN__::Matrix3x3d S, M_temp[3], Vtranspose;


    //S = V * sigma * V.transpose();
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_temp[0]);
    Vtranspose = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_temp[0], Vtranspose, S);
    //__S_Mat_multiply(M_temp[2], ml, Q[i]);

    double3 v_temp = __GEIGEN__::__M_v_multiply(S, direction);
    double I4 = __GEIGEN__::__v_vec_dot(direction, v_temp);//direction.transpose() * S * direction;
    double I5 = __GEIGEN__::__v_vec_dot(v_temp, v_temp);//direction.transpose() * S.transpose() * S * direction;

    __GEIGEN__::Matrix9x9d H;
    __GEIGEN__::__init_Mat9x9(H, 0);
    if (abs(I5) < 1e-15) return H;

    double s = 0;
    if (I4 < 0) {
        s = -1;
    }
    else if (I4 > 0) {
        s = 1;
    }

    double lamda0 = scale;
    double lamda1 = scale * (1 - s * contract_length / sqrt(I5));
    double lamda2 = lamda1;
    //double lamda2 = lamda1;
    __GEIGEN__::Matrix3x3d Q0, Q1, Q2, A;
    A = __GEIGEN__::__v_vec_toMat(direction, direction);

    __GEIGEN__::__M_Mat_multiply(F, A, M_temp[0]);
    Q0 = __GEIGEN__::__S_Mat_multiply(M_temp[0], (1 / sqrt(I5)));
    //Q0 = (1 / sqrt(I5)) * F * A;

    __GEIGEN__::Matrix3x3d Tx, Ty, Tz;

    __GEIGEN__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __GEIGEN__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __GEIGEN__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    //__Transpose3x3(V, M_temp[0]);
    double3 directionM = __GEIGEN__::__M_v_multiply(Vtranspose, direction);

    double ratio = 1.f / sqrt(2.f);
    Tx = __GEIGEN__::__S_Mat_multiply(Tx, ratio);
    Ty = __GEIGEN__::__S_Mat_multiply(Ty, ratio);
    Tz = __GEIGEN__::__S_Mat_multiply(Tz, ratio);

    //Q1 = U * Tx * sigma * V.transpose() * A;
    __GEIGEN__::__M_Mat_multiply(U, Tx, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], A, Q1);

    //Q2 = (sigma(1, 1) * directionM[1]) * U * Tz * sigma * V.transpose() * A - (sigma(2, 2) * directionM[2]) * U * Ty * sigma * V.transpose() * A;
    __GEIGEN__::__M_Mat_multiply(U, Tz, M_temp[0]);
    __GEIGEN__::__M_Mat_multiply(M_temp[0], sigma, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], Vtranspose, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], A, M_temp[0]);
    M_temp[0] = __S_Mat_multiply(M_temp[0], (sigma.m[1][1] * directionM.y));
    __GEIGEN__::__M_Mat_multiply(U, Ty, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __GEIGEN__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __GEIGEN__::__M_Mat_multiply(M_temp[1], A, M_temp[2]);
    M_temp[2] = __GEIGEN__::__S_Mat_multiply(M_temp[2], -(sigma.m[2][2] * directionM.z));
    __GEIGEN__::__Mat_add(M_temp[0], M_temp[2], Q2);

    //H = lamda0 * vec_double(Q0) * vec_double(Q0).transpose();
    __GEIGEN__::Vector9 V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q0);
    __GEIGEN__::Matrix9x9d M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
    H = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda0);
    //H = __Mat9x9_add(H, M9_temp[1]);
    if (lamda1 > 0) {
        //H += lamda1 * vec_double(Q1) * vec_double(Q1).transpose();
        //H += lamda2 * vec_double(Q2) * vec_double(Q2).transpose();
        V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q1);
        M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda1);
        H = __GEIGEN__::__Mat9x9_add(H, M9_temp);

        V9_temp = __GEIGEN__::__Mat3x3_to_vec9_double(Q2);
        M9_temp = __GEIGEN__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __GEIGEN__::__S_Mat9x9_multiply(M9_temp, lamda2);
        H = __GEIGEN__::__Mat9x9_add(H, M9_temp);
    }

    return H;
}

__device__ 
__GEIGEN__::Matrix3x3d __computePEPF_Aniostropic3D_double(const __GEIGEN__::Matrix3x3d& F, double3 fiber_direction, const double& scale, const double& contract_length) {

    double3 direction = __GEIGEN__::__normalized(fiber_direction);
    __GEIGEN__::Matrix3x3d U, V, S, sigma, M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_Temp0, M_Temp1, S);
    double3 V_Temp0, V_Temp1;
    V_Temp0 = __GEIGEN__::__v_M_multiply(direction, S);
    double I4, I5;
    I4 = __GEIGEN__::__v_vec_dot(V_Temp0, direction);
    V_Temp1 = __GEIGEN__::__M_v_multiply(S, direction);
    I5 = __GEIGEN__::__v_vec_dot(V_Temp1, V_Temp1);

    if (I4 == 0) {
        // system("pause");
    }

    double s = 0;
    if (I4 < 0) {
        s = -1;
    }
    else if (I4 > 0) {
        s = 1;
    }

    __GEIGEN__::Matrix3x3d PEPF;
    double s_temp0 = scale * (1 - s * contract_length / sqrt(I5));
    M_Temp0 = __GEIGEN__::__v_vec_toMat(direction, direction);
    __GEIGEN__::__M_Mat_multiply(F, M_Temp0, M_Temp1);
    PEPF = __GEIGEN__::__S_Mat_multiply(M_Temp1, s_temp0);
    return PEPF;
}

__device__
double __cal_StabbleNHK_energy_3D(const double3* vertexes, const uint4& tetrahedra, const __GEIGEN__::Matrix3x3d& DmInverse, const double& volume, const double& lenRate, const double& volRate) {
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0], F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0], F.m[2][1], F.m[2][2]);
    __GEIGEN__::Matrix3x3d U, V, sigma, S;
    __GEIGEN__::Matrix3x3d M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", V.m[0][0], V.m[0][1], V.m[0][2], V.m[1][0], V.m[1][1], V.m[1][2], V.m[2][0], V.m[2][1], V.m[2][2]);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", U.m[0][0], U.m[0][1], U.m[0][2], U.m[1][0], U.m[1][1], U.m[1][2], U.m[2][0], U.m[2][1], U.m[2][2]);
    __GEIGEN__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __GEIGEN__::__Transpose3x3(V);
    __GEIGEN__::__M_Mat_multiply(M_Temp0, M_Temp1, S);

    __GEIGEN__::__M_Mat_multiply(S, S, M_Temp0);
    double I2 = __GEIGEN__::__Mat_Trace(M_Temp0);
    double I3;
    __GEIGEN__::__Determiant(S, I3);
    //printf("%f     %f\n\n\n", I2, I3);
    return (0.5 * lenRate * (I2 - 3) + 0.5 * volRate * (I3 - 1 - 3 * lenRate / 4 / volRate) * (I3 - 1 - 3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(I2 + 1) /*- (0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(4.0))*/) * volume;
    //printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__
double __cal_ARAP_energy_3D(const double3* vertexes, const uint4& tetrahedra, const __GEIGEN__::Matrix3x3d& DmInverse, const double& volume, const double& lenRate) {
    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedra, Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverse, F);
    //printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0], F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0], F.m[2][1], F.m[2][2]);
    __GEIGEN__::Matrix3x3d U, V, sigma, S, R;
    __GEIGEN__::Matrix3x3d M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);

    S = __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(V, sigma), __GEIGEN__::__Transpose3x3(V));//V * sigma * V.transpose();
    R = __GEIGEN__::__M_Mat_multiply(U, __GEIGEN__::__Transpose3x3(V));
    __GEIGEN__::Matrix3x3d g = __GEIGEN__::__Mat3x3_minus(F, R);
    double energy = 0;
    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            energy += g.m[i][j] * g.m[i][j];
        }
    }
    return energy * volume * lenRate * 0.5;
    //printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__
__GEIGEN__::Matrix9x12d __computePFDsPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm) {
    __GEIGEN__::Matrix9x12d matOut;
    __GEIGEN__::__init_Mat9x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1 = -(m + p + s);
    double t2 = -(n + q + t);
    double t3 = -(o + r + u);
    matOut.m[0][0] = t1;  matOut.m[0][3] = m;  matOut.m[0][6] = p;  matOut.m[0][9] = s;
    matOut.m[1][1] = t1;  matOut.m[1][4] = m;  matOut.m[1][7] = p;  matOut.m[1][10] = s;
    matOut.m[2][2] = t1;  matOut.m[2][5] = m;  matOut.m[2][8] = p;  matOut.m[2][11] = s;
    matOut.m[3][0] = t2;  matOut.m[3][3] = n;  matOut.m[3][6] = q;  matOut.m[3][9] = t;
    matOut.m[4][1] = t2;  matOut.m[4][4] = n;  matOut.m[4][7] = q;  matOut.m[4][10] = t;
    matOut.m[5][2] = t2;  matOut.m[5][5] = n;  matOut.m[5][8] = q;  matOut.m[5][11] = t;
    matOut.m[6][0] = t3;  matOut.m[6][3] = o;  matOut.m[6][6] = r;  matOut.m[6][9] = u;
    matOut.m[7][1] = t3;  matOut.m[7][4] = o;  matOut.m[7][7] = r;  matOut.m[7][10] = u;
    matOut.m[8][2] = t3;  matOut.m[8][5] = o;  matOut.m[8][8] = r;  matOut.m[8][11] = u;

    return matOut;
}

__device__
__GEIGEN__::Matrix6x12d __computePFDsPX3D_6x12_double(const __GEIGEN__::Matrix2x2d& InverseDm) {
    __GEIGEN__::Matrix6x12d matOut;
    __GEIGEN__::__init_Mat6x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1];

    matOut.m[0][0] = -m; matOut.m[3][0] = -n;
    matOut.m[1][1] = -m; matOut.m[4][1] = -n;
    matOut.m[2][2] = -m; matOut.m[5][2] = -n;

    matOut.m[0][3] = -p; matOut.m[3][3] = -q;
    matOut.m[1][4] = -p; matOut.m[4][4] = -q;
    matOut.m[2][5] = -p; matOut.m[5][5] = -q;

    matOut.m[0][6] = p; matOut.m[3][6] = q;
    matOut.m[1][7] = p; matOut.m[4][7] = q;
    matOut.m[2][8] = p; matOut.m[5][8] = q;

    matOut.m[0][9] = m; matOut.m[3][9] = n;
    matOut.m[1][10] = m; matOut.m[4][10] = n;
    matOut.m[2][11] = m; matOut.m[5][11] = n;

    return matOut;
}

__device__
__GEIGEN__::Matrix6x9d __computePFDsPX3D_6x9_double(const __GEIGEN__::Matrix2x2d& InverseDm) {
    __GEIGEN__::Matrix6x9d matOut;
    __GEIGEN__::__init_Mat6x9_val(matOut, 0);
    double d0 = InverseDm.m[0][0], d2 = InverseDm.m[0][1];
    double d1 = InverseDm.m[1][0], d3 = InverseDm.m[1][1];

    double s0 = d0 + d1;
    double s1 = d2 + d3;

    matOut.m[0][0] = -s0;
    matOut.m[3][0] = -s1;

    // dF / dy0
    matOut.m[1][1] = -s0;
    matOut.m[4][1] = -s1;

    // dF / dz0
    matOut.m[2][2] = -s0;
    matOut.m[5][2] = -s1;

    // dF / dx1
    matOut.m[0][3] = d0;
    matOut.m[3][3] = d2;

    // dF / dy1
    matOut.m[1][4] = d0;
    matOut.m[4][4] = d2;

    // dF / dz1
    matOut.m[2][5] = d0;
    matOut.m[5][5] = d2;

    // dF / dx2
    matOut.m[0][6] = d1;
    matOut.m[3][6] = d3;

    // dF / dy2
    matOut.m[1][7] = d1;
    matOut.m[4][7] = d3;

    // dF / dz2
    matOut.m[2][8] = d1;
    matOut.m[5][8] = d3;

    return matOut;
}

__device__
__GEIGEN__::Matrix3x6d __computePFDsPX3D_3x6_double(const double& InverseDm) {
    __GEIGEN__::Matrix3x6d matOut;
    __GEIGEN__::__init_Mat3x6_val(matOut, 0);

    matOut.m[0][0] = -InverseDm;
    matOut.m[1][1] = -InverseDm;
    matOut.m[2][2] = -InverseDm;

    matOut.m[0][3] = InverseDm;
    matOut.m[1][4] = InverseDm;
    matOut.m[2][5] = InverseDm;

    return matOut;
}

__device__
__GEIGEN__::Matrix9x12d __computePFDmPX3D_double(const __GEIGEN__::Matrix12x9d& PDmPx, const __GEIGEN__::Matrix3x3d& Ds, const __GEIGEN__::Matrix3x3d& DmInv) {
    __GEIGEN__::Matrix9x12d DsPDminvPx;
    __GEIGEN__::__init_Mat9x12_val(DsPDminvPx, 0);

    for (int i = 0; i < 12; i++) {
        __GEIGEN__::Matrix3x3d PDmPxi = __GEIGEN__::__vec9_to_Mat3x3_double(PDmPx.m[i]);
        __GEIGEN__::Matrix3x3d DsPDminvPxi;
        __GEIGEN__::__M_Mat_multiply(Ds, __GEIGEN__::__M_Mat_multiply(__GEIGEN__::__M_Mat_multiply(DmInv, PDmPxi), DmInv), DsPDminvPxi);

        __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(DsPDminvPxi);

        for (int j = 0;j < 9;j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix6x12d __computePFDmPX3D_6x12_double(const __GEIGEN__::Matrix12x4d& PDmPx, const __GEIGEN__::Matrix3x2d& Ds, const __GEIGEN__::Matrix2x2d& DmInv) {
    __GEIGEN__::Matrix6x12d DsPDminvPx;
    __GEIGEN__::__init_Mat6x12_val(DsPDminvPx, 0);

    for (int i = 0; i < 12; i++) {
        __GEIGEN__::Matrix2x2d PDmPxi = __GEIGEN__::__vec4_to_Mat2x2_double(PDmPx.m[i]);

        __GEIGEN__::Matrix3x2d DsPDminvPxi = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, __GEIGEN__::__M2x2_Mat2x2_multiply(__GEIGEN__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(DsPDminvPxi);
        for (int j = 0;j < 6;j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }

    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix3x6d __computePFDmPX3D_3x6_double(const __GEIGEN__::Vector6& PDmPx, const double3& Ds, const double& DmInv) {
    __GEIGEN__::Matrix3x6d DsPDminvPx;
    __GEIGEN__::__init_Mat3x6_val(DsPDminvPx, 0);

    for (int i = 0; i < 6; i++) {
        double PDmPxi = PDmPx.v[i];

        double3 DsPDminvPxi = __GEIGEN__::__s_vec_multiply(Ds, ((DmInv * PDmPxi) * DmInv));
        DsPDminvPx.m[0][i] = -DsPDminvPxi.x;
        DsPDminvPx.m[1][i] = -DsPDminvPxi.y;
        DsPDminvPx.m[2][i] = -DsPDminvPxi.z;
    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix6x9d __computePFDmPX3D_6x9_double(const __GEIGEN__::Matrix9x4d& PDmPx, const __GEIGEN__::Matrix3x2d& Ds, const __GEIGEN__::Matrix2x2d& DmInv) {
    __GEIGEN__::Matrix6x9d DsPDminvPx;
    __GEIGEN__::__init_Mat6x9_val(DsPDminvPx, 0);

    for (int i = 0; i < 9; i++) {
        __GEIGEN__::Matrix2x2d PDmPxi = __GEIGEN__::__vec4_to_Mat2x2_double(PDmPx.m[i]);

        __GEIGEN__::Matrix3x2d DsPDminvPxi = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, __GEIGEN__::__M2x2_Mat2x2_multiply(__GEIGEN__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(DsPDminvPxi);
        for (int j = 0;j < 6;j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }

    }

    return DsPDminvPx;
}

__device__
__GEIGEN__::Matrix9x12d __computePFPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm) {
    __GEIGEN__::Matrix9x12d matOut;
    __GEIGEN__::__init_Mat9x12_val(matOut, 0);
    double m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    double p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    double s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    double t1 = -(m + p + s);
    double t2 = -(n + q + t);
    double t3 = -(o + r + u);
    matOut.m[0][0] = t1;  matOut.m[0][3] = m;  matOut.m[0][6] = p;  matOut.m[0][9] = s;
    matOut.m[1][1] = t1;  matOut.m[1][4] = m;  matOut.m[1][7] = p;  matOut.m[1][10] = s;
    matOut.m[2][2] = t1;  matOut.m[2][5] = m;  matOut.m[2][8] = p;  matOut.m[2][11] = s;
    matOut.m[3][0] = t2;  matOut.m[3][3] = n;  matOut.m[3][6] = q;  matOut.m[3][9] = t;
    matOut.m[4][1] = t2;  matOut.m[4][4] = n;  matOut.m[4][7] = q;  matOut.m[4][10] = t;
    matOut.m[5][2] = t2;  matOut.m[5][5] = n;  matOut.m[5][8] = q;  matOut.m[5][11] = t;
    matOut.m[6][0] = t3;  matOut.m[6][3] = o;  matOut.m[6][6] = r;  matOut.m[6][9] = u;
    matOut.m[7][1] = t3;  matOut.m[7][4] = o;  matOut.m[7][7] = r;  matOut.m[7][10] = u;
    matOut.m[8][2] = t3;  matOut.m[8][5] = o;  matOut.m[8][8] = r;  matOut.m[8][11] = u;
    return matOut;
}

__global__
void _calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    __GEIGEN__::Matrix12x12d* Hessians, uint32_t offset, const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double IPC_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tetrahedraNum) return;

    __GEIGEN__::Matrix9x12d PFPX = __computePFPX3D_double(DmInverses[idx]);

    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedras[idx], Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    __GEIGEN__::Matrix3x3d U, V, Sigma;
    SVD(F, U, V, Sigma);

#ifdef USE_SNK
    __GEIGEN__::Matrix3x3d Iso_PEPF = __computePEPF_StableNHK3D_double(F, Sigma, U, V, lenRate, volRate);
#else
    __GEIGEN__::Matrix3x3d Iso_PEPF = computePEPF_ARAP_double(F, Sigma, U, V, lenRate);
#endif


    __GEIGEN__::Matrix3x3d PEPF = Iso_PEPF;

    __GEIGEN__::Vector9 pepf = __GEIGEN__::__Mat3x3_to_vec9_double(PEPF);




    __GEIGEN__::Matrix12x9d PFPXTranspose = __GEIGEN__::__Transpose9x12(PFPX);
    __GEIGEN__::Vector12 f = __GEIGEN__::__s_vec12_multiply(__GEIGEN__::__M12x9_v9_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * volume[idx]);
    //printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0], f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9], f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        atomicAdd(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        atomicAdd(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        atomicAdd(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        atomicAdd(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        atomicAdd(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }

#ifdef USE_SNK
    __GEIGEN__::Matrix9x9d Hq = __project_StabbleNHK_H_3D(Sigma, U, V, lenRate, volRate);
#else
    __GEIGEN__::Matrix9x9d Hq = project_ARAP_H_3D(Sigma, U, V, lenRate);
#endif
    

    __GEIGEN__::Matrix12x9d M12x9_temp;// = __Transpose9x12(PFPX);
    M12x9_temp = __GEIGEN__::__M12x9_M9x9_Multiply(PFPXTranspose, Hq);
    __GEIGEN__::Matrix12x12d H = __GEIGEN__::__M12x9_M9x12_Multiply(M12x9_temp, PFPX);
    H = __GEIGEN__::__s_M12x12_Multiply(H, volume[idx] * IPC_dt * IPC_dt);
    Hessians[idx + offset] = H;
}

__global__
void _calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tetrahedraNum) return;

    __GEIGEN__::Matrix9x12d PFPX = __computePFPX3D_double(DmInverses[idx]);

    __GEIGEN__::Matrix3x3d Ds;
    __calculateDms3D_double(vertexes, tetrahedras[idx], Ds);
    __GEIGEN__::Matrix3x3d F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    __GEIGEN__::Matrix3x3d U, V, Sigma;
    SVD(F, U, V, Sigma);
    //printf("%f %f\n\n\n", lenRate, volRate);
#ifdef USE_SNK
    __GEIGEN__::Matrix3x3d Iso_PEPF = __computePEPF_StableNHK3D_double(F, Sigma, U, V, lenRate, volRate);
#else
    __GEIGEN__::Matrix3x3d Iso_PEPF = computePEPF_ARAP_double(F, Sigma, U, V, lenRate);
#endif



    __GEIGEN__::Matrix3x3d PEPF = Iso_PEPF;

    __GEIGEN__::Vector9 pepf = __GEIGEN__::__Mat3x3_to_vec9_double(PEPF);

    __GEIGEN__::Matrix12x9d PFPXTranspose = __GEIGEN__::__Transpose9x12(PFPX);

    __GEIGEN__::Vector12 f = __GEIGEN__::__M12x9_v9_multiply(PFPXTranspose, pepf);

    for (int i = 0; i < 12; i++) {
        f.v[i] = volume[idx] * f.v[i];
    }

    {
        atomicAdd(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        atomicAdd(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        atomicAdd(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        atomicAdd(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        atomicAdd(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        atomicAdd(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }
}

double calculateVolum(const double3* vertexes, const uint4& index) {
    int id0 = 0;
    int id1 = 1;
    int id2 = 2;
    int id3 = 3;
    double o1x = vertexes[index.y].x - vertexes[index.x].x;
    double o1y = vertexes[index.y].y - vertexes[index.x].y;
    double o1z = vertexes[index.y].z - vertexes[index.x].z;
    double3 OA = make_double3(o1x, o1y, o1z);

    double o2x = vertexes[index.z].x - vertexes[index.x].x;
    double o2y = vertexes[index.z].y - vertexes[index.x].y;
    double o2z = vertexes[index.z].z - vertexes[index.x].z;
    double3 OB = make_double3(o2x, o2y, o2z);

    double o3x = vertexes[index.w].x - vertexes[index.x].x;
    double o3y = vertexes[index.w].y - vertexes[index.x].y;
    double o3z = vertexes[index.w].z - vertexes[index.x].z;
    double3 OC = make_double3(o3x, o3y, o3z);

    double3 heightDir = __GEIGEN__::__v_vec_cross(OA, OB);//OA.cross(OB);
    double bottomArea = __GEIGEN__::__norm(heightDir);//heightDir.norm();
    heightDir = __GEIGEN__::__normalized(heightDir);

    double volum = bottomArea * __GEIGEN__::__v_vec_dot(heightDir, OC) / 6;
    return volum > 0 ? volum : -volum;
}