#ifndef WENO_H
#define WENO_H

#include "Types.h"
#include <Eigen/Eigen>

namespace ZenEulerGas {
namespace Math {
namespace RPSolver {
/**
   WENO(weighted essentially non-oscillatory) reconstructor
*/
template <class T, class DerivedV, class DerivedE>
DerivedV WENO2(const DerivedV& v1, const DerivedV& v2, const DerivedV& v3,
    const DerivedE& epsilon)
{
    DerivedV s1 = (v2 - v1) * (v2 - v1);
    DerivedV s2 = (v3 - v2) * (v3 - v2);
    DerivedV a1 = (T)1 / (T)3 / ((epsilon + s1) * (epsilon + s1));
    DerivedV a2 = (T)2 / (T)3 / ((epsilon + s2) * (epsilon + s2));
    DerivedV one_over_sum_a = (T)1 / (a1 + a2);
    DerivedV w1 = a1 * (one_over_sum_a);
    DerivedV w2 = (T)1 - w1;
    return (T)(.5) * (w1 * (-v1 + (T)3 * v2) + w2 * (v2 + v3));
}

template <class T, class DerivedV, class DerivedE>
DerivedV WENO3(const DerivedV& v1, const DerivedV& v2, const DerivedV& v3,
    const DerivedV& v4, const DerivedV& v5,
    const DerivedE& epsilon)
{
    DerivedV s1 = (T)13 / (T)12 * (v1 - (T)2 * v2 + v3) * (v1 - (T)2 * v2 + v3) + (T)(.25) * (v1 - (T)4 * v2 + (T)3 * v3) * (v1 - (T)4 * v2 + (T)3 * v3);
    DerivedV s2 = (T)13 / (T)12 * (v2 - (T)2 * v3 + v4) * (v2 - (T)2 * v3 + v4) + (T)(.25) * (v2 - v4) * (v2 - v4);
    DerivedV s3 = (T)13 / (T)12 * (v3 - (T)2 * v4 + v5) * (v3 - (T)2 * v4 + v5) + (T)(.25) * ((T)3 * v3 - (T)4 * v4 + v5) * ((T)3 * v3 - (T)4 * v4 + v5);
    DerivedV a1 = (T)(.1) / ((epsilon + s1) * (epsilon + s1));
    DerivedV a2 = (T)(.6) / ((epsilon + s2) * (epsilon + s2));
    DerivedV a3 = (T)(.3) / ((epsilon + s3) * (epsilon + s3));
    DerivedV one_over_sum_a = (T)1 / (a1 + a2 + a3);
    DerivedV w1 = a1 * (one_over_sum_a);
    DerivedV w2 = a2 * (one_over_sum_a);
    DerivedV w3 = (T)1 - w1 - w2;
    return (T)1 / (T)6 * (w1 * ((T)2 * v1 - (T)7 * v2 + (T)11 * v3) + w2 * (-v2 + (T)5 * v3 + (T)2 * v4) + w3 * ((T)2 * v3 + (T)5 * v4 - v5));
}

/**
   1st-LLF(local Lax–Friedrichs) scheme
*/
template <class T, class DerivedV>
DerivedV First_LLF(const std::array<T, 2>& U,
    const std::array<DerivedV, 2>& Q)
{
    const T& u1 = U[0];
    const T& u2 = U[1];

    const DerivedV& q1 = Q[0];
    const DerivedV& q2 = Q[1];

    T alpha = std::max(std::abs(u2), std::abs(u2));

    return 0.5 * ((u1 + alpha) * q1 + (u2 - alpha) * q2);
};

/**
   WENO-LLF(local Lax–Friedrichs) scheme
*/
template <class T, class DerivedV, class DerivedE>
DerivedV WENO2_LLF(const std::array<T, 4>& U, const std::array<DerivedV, 4>& Q,
    const DerivedE& eps)
{
    const T& u1 = U[0];
    const T& u2 = U[1];
    const T& u3 = U[2];
    const T& u4 = U[3];

    const DerivedV& q1 = Q[0];
    const DerivedV& q2 = Q[1];
    const DerivedV& q3 = Q[2];
    const DerivedV& q4 = Q[3];

    T alpha = std::max(std::abs(u2), std::abs(u3));

    return 0.5 * (WENO2<T, DerivedV, DerivedE>((u1 + alpha) * q1, (u2 + alpha) * q2, (u3 + alpha) * q3, eps) + WENO2<T, DerivedV, DerivedE>((u4 - alpha) * q4, (u3 - alpha) * q3, (u2 - alpha) * q2, eps));
};

template <class T, class DerivedV, class DerivedE>
DerivedV WENO3_LLF(const std::array<T, 5>& U, const std::array<DerivedV, 5>& Q,
    const DerivedE& eps)
{
    const T& u1 = U[0];
    const T& u2 = U[1];
    const T& u3 = U[2];
    const T& u4 = U[3];
    const T& u5 = U[4];

    const DerivedV& q1 = Q[0];
    const DerivedV& q2 = Q[1];
    const DerivedV& q3 = Q[2];
    const DerivedV& q4 = Q[3];
    const DerivedV& q5 = Q[4];

    T alpha = std::max(std::abs(u2), std::abs(u3));

    return 0.5 * (WENO3<T, DerivedV, DerivedE>((u1 + alpha) * q1, (u2 + alpha) * q2, (u3 + alpha) * q3, (u4 + alpha) * q4, eps) + WENO3<T, DerivedV, DerivedE>((u5 - alpha) * q5, (u4 - alpha) * q4, (u3 - alpha) * q3, (u2 - alpha) * q2, eps));
};

}

}
} // namespace ZenEulerGas::Math::RPSolver

#endif