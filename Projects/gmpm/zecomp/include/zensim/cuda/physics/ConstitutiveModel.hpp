#pragma once
#include "zensim/cuda/math/matrix/svd.cuh"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/MatrixUtils.h"
#include "zensim/physics/ConstitutiveModel.hpp"

namespace zs {

  template <typename T = float>
  __forceinline__ __device__ void compute_stress_fixedcorotated(T volume, T mu, T lambda,
                                                                const vec<T, 9> &F, vec<T, 9> &PF) {
    T U[9], S[3], V[9];
    math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4],
              U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2],
              V[5], V[8]);
    T J = S[0] * S[1] * S[2];
    T scaled_mu = 2.f * mu;
    T scaled_lambda = lambda * (J - 1.f);
    vec<T, 3> P_hat;
    P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
    P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
    P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);

    vec<T, 9> P;
    P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6];
    P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6];
    P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
    P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7];
    P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7];
    P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
    P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8];
    P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8];
    P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

    /// PF'
    PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
    PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
    PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
    PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
    PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
    PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
    PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
    PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
    PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
  }

  template <typename T = float>
  __forceinline__ __device__ void compute_stress_vonmisesfixedcorotated(T volume, T mu, T lambda,
                                                                        T yield_stress,
                                                                        vec<T, 9> &F,
                                                                        vec<T, 9> &PF) {
    using TV = vec<T, 3>;
    TV S, Sclamp;
    T U[9], V[9];
    math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4],
              U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2],
              V[5], V[8]);
    T J;

    for (int d = 0; d < 3; d++) Sclamp(d) = (T)1e-4 > S(d) ? (T)1e-4 : S(d);
    J = Sclamp.prod();
    TV tau_trial;
    for (int d = 0; d < 3; d++)
      tau_trial(d) = 2 * mu * (Sclamp(d) - 1) * Sclamp(d) + lambda * (J - 1) * J;
    T trace_tau = tau_trial.sum();
    TV s_trial = tau_trial - (trace_tau / (T)3);
    T s_norm = sqrtf(s_trial.l2NormSqr());
    T scaled_tauy = sqrtf((T)2 / ((T)6 - 3)) * yield_stress;
    if (s_norm - scaled_tauy > 0) {
      T alpha = scaled_tauy / s_norm;
      TV tau_new = alpha * s_trial + (trace_tau / (T)3);
      J = 1.f;
      for (int d = 0; d < 3; d++) {
        T b2m4ac = mu * mu - 2 * mu * (lambda * (J - 1) * J - tau_new(d));
        // ZIRAN_ASSERT(b2m4ac >= 0, "Wrong projection ", b2m4ac);
        if (b2m4ac < 0) printf("Wrong projection!\n");
        T sqrtb2m4ac = sqrtf(b2m4ac);
        T x1 = (mu + sqrtb2m4ac) / (2 * mu);
        S(d) = x1;
      }
      matmul_mat_diag_matT_3D(F.data(), U, S.data(), V);
    }

    ///
    J = S.prod();
    T scaled_mu = 2.f * mu;
    T scaled_lambda = lambda * (J - 1.f);
    vec<T, 3> P_hat;
    P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
    P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
    P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);

    vec<T, 9> P;
    P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6];
    P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6];
    P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
    P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7];
    P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7];
    P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
    P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8];
    P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8];
    P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

    /// PF'
    PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
    PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
    PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
    PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
    PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
    PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
    PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
    PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
    PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
  }

  template <typename T = float>
  __forceinline__ __device__ void compute_stress_nacc(T volume, T mu, T lambda, T bm, T xi, T beta,
                                                      T Msqr, bool hardeningOn, T &logJp,
                                                      vec<T, 9> &F, vec<T, 9> &PF) {
    T U[9], S[3], V[9];
    math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4],
              U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2],
              V[5], V[8]);
    T p0 = bm * (T(0.00001) + sinh(xi * (-logJp > 0 ? -logJp : 0)));
    T p_min = -beta * p0;

    T Je_trial = S[0] * S[1] * S[2];

    ///< 0). calculate YS
    T B_hat_trial[3] = {S[0] * S[0], S[1] * S[1], S[2] * S[2]};
    T trace_B_hat_trial_divdim = (B_hat_trial[0] + B_hat_trial[1] + B_hat_trial[2]) / 3.f;
    T J_power_neg_2_d_mulmu = mu * powf(Je_trial, -2.f / 3.f);  ///< J^(-2/dim) * mu
    T s_hat_trial[3] = {J_power_neg_2_d_mulmu * (B_hat_trial[0] - trace_B_hat_trial_divdim),
                        J_power_neg_2_d_mulmu * (B_hat_trial[1] - trace_B_hat_trial_divdim),
                        J_power_neg_2_d_mulmu * (B_hat_trial[2] - trace_B_hat_trial_divdim)};
    T psi_kappa_partial_J = bm * 0.5f * (Je_trial - 1.f / Je_trial);
    T p_trial = -psi_kappa_partial_J * Je_trial;

    T y_s_half_coeff = 3.f / 2.f * (1 + 2.f * beta);  ///< a
    T y_p_half = (Msqr * (p_trial - p_min) * (p_trial - p0));
    T s_hat_trial_sqrnorm = s_hat_trial[0] * s_hat_trial[0] + s_hat_trial[1] * s_hat_trial[1]
                            + s_hat_trial[2] * s_hat_trial[2];
    T y = (y_s_half_coeff * s_hat_trial_sqrnorm) + y_p_half;

    //< 1). update strain and hardening alpha(in logJp)

    ///< case 1, project to max tip of YS
    if (p_trial > p0) {
      T Je_new = sqrtf(-2.f * p0 / bm + 1.f);
      S[0] = S[1] = S[2] = powf(Je_new, 1.f / 3.f);
      T New_F[9];
      matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
      for (int i = 0; i < 9; i++) F[i] = New_F[i];
      if (hardeningOn) logJp += logf(Je_trial / Je_new);
    }  ///< case 1 -- end

    /// case 2, project to min tip of YS
    else if (p_trial < p_min) {
      T Je_new = sqrtf(-2.f * p_min / bm + 1.f);
      S[0] = S[1] = S[2] = powf(Je_new, 1.f / 3.f);
      T New_F[9];
      matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
      for (int i = 0; i < 9; i++) F[i] = New_F[i];
      if (hardeningOn) logJp += logf(Je_trial / Je_new);
    }  ///< case 2 -- end

    /// case 3, keep or project to YS by hardening
    else {
      ///< outside YS
      if (y >= 1e-4) {
        ////< yield surface projection
        T B_s_coeff = powf(Je_trial, 2.f / 3.f) / mu * sqrtf(-y_p_half / y_s_half_coeff)
                      / sqrtf(s_hat_trial_sqrnorm);
#pragma unroll 3
        for (int i = 0; i < 3; i++)
          S[i] = sqrtf(s_hat_trial[i] * B_s_coeff + trace_B_hat_trial_divdim);
        T New_F[9];
        matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
        for (int i = 0; i < 9; i++) F[i] = New_F[i];

        ////< hardening
        if (hardeningOn && p0 > 1e-4 && p_trial < p0 - 1e-4 && p_trial > 1e-4 + p_min) {
          T p_center = ((T)1 - beta) * p0 / 2;
#if 1  /// solve in 19 Josh Fracture paper
          T q_trial = sqrtf(3.f / 2.f * s_hat_trial_sqrnorm);
          T direction[2] = {p_center - p_trial, -q_trial};
          T direction_norm = sqrtf(direction[0] * direction[0] + direction[1] * direction[1]);
          direction[0] /= direction_norm;
          direction[1] /= direction_norm;

          T C = Msqr * (p_center - p_min) * (p_center - p0);
          T B = Msqr * direction[0] * (2 * p_center - p0 - p_min);
          T A = Msqr * direction[0] * direction[0] + (1 + 2 * beta) * direction[1] * direction[1];

          T l1 = (-B + sqrtf(B * B - 4 * A * C)) / (2 * A);
          T l2 = (-B - sqrtf(B * B - 4 * A * C)) / (2 * A);

          T p1 = p_center + l1 * direction[0];
          T p2 = p_center + l2 * direction[0];
#else  /// solve in ziran - Compare_With_Physbam
          T aa = Msqr * powf(p_trial - p_center, 2) / (y_s_half_coeff * s_hat_trial_sqrnorm);
          T dd = 1 + aa;
          T ff = aa * beta * p0 - aa * p0 - 2 * p_center;
          T gg = (p_center * p_center) - aa * beta * (p0 * p0);
          T zz = sqrtf(fabsf(ff * ff - 4 * dd * gg));
          T p1 = (-ff + zz) / (2 * dd);
          T p2 = (-ff - zz) / (2 * dd);
#endif

          T p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;
          T tmp_Je_sqr = (-2 * p_fake / bm + 1);
          T Je_new_fake = sqrtf(tmp_Je_sqr > 0 ? tmp_Je_sqr : -tmp_Je_sqr);
          if (Je_new_fake > 1e-4) logJp += logf(Je_trial / Je_new_fake);
        }
      }  ///< outside YS -- end
    }    ///< case 3 --end

    //< 2). elasticity
    ///< known: F(renewed), U, V, S(renewed)
    ///< unknown: J, dev(FF^T)
    T J = S[0] * S[1] * S[2];
    T b_dev[9], b[9];
    matrixMatrixTranposeMultiplication3d(F.data(), b);
    matrixDeviatoric3d(b, b_dev);

    ///< |f| = P * F^T * Volume
    T dev_b_coeff = mu * powf(J, -2.f / 3.f);
    T i_coeff = bm * .5f * (J * J - 1.f);
    PF[0] = (dev_b_coeff * b_dev[0] + i_coeff) * volume;
    PF[1] = (dev_b_coeff * b_dev[1]) * volume;
    PF[2] = (dev_b_coeff * b_dev[2]) * volume;
    PF[3] = (dev_b_coeff * b_dev[3]) * volume;
    PF[4] = (dev_b_coeff * b_dev[4] + i_coeff) * volume;
    PF[5] = (dev_b_coeff * b_dev[5]) * volume;
    PF[6] = (dev_b_coeff * b_dev[6]) * volume;
    PF[7] = (dev_b_coeff * b_dev[7]) * volume;
    PF[8] = (dev_b_coeff * b_dev[8] + i_coeff) * volume;
  }

  template <typename T = float>
  __forceinline__ __device__ void compute_stress_sand(T volume, T mu, T lambda, T cohesion, T beta,
                                                      T yieldSurface, bool volCorrection, T &logJp,
                                                      vec<T, 9> &F, vec<T, 9> &PF) {
    T U[9], S[3], V[9];
    math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4],
              U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2],
              V[5], V[8]);
    T scaled_mu = T(2) * mu;

    T epsilon[3], New_S[3];  ///< helper
    T New_F[9];

#pragma unroll 3
    for (int i = 0; i < 3; i++) {
      T abs_S = S[i] > 0 ? S[i] : -S[i];
      abs_S = abs_S > 1e-4 ? abs_S : 1e-4;
      epsilon[i] = logf(abs_S) - cohesion;
    }
    T sum_epsilon = epsilon[0] + epsilon[1] + epsilon[2];
    T trace_epsilon = sum_epsilon + logJp;

    T epsilon_hat[3];
#pragma unroll 3
    for (int i = 0; i < 3; i++) epsilon_hat[i] = epsilon[i] - (trace_epsilon / (T)3);

    T epsilon_hat_norm = sqrtf(epsilon_hat[0] * epsilon_hat[0] + epsilon_hat[1] * epsilon_hat[1]
                               + epsilon_hat[2] * epsilon_hat[2]);

    /* Calculate Plasticiy */
    if (trace_epsilon >= (T)0) {  ///< case II: project to the cone tip
      New_S[0] = New_S[1] = New_S[2] = expf(cohesion);
      matmul_mat_diag_matT_3D(New_F, U, New_S, V);  // new F_e
                                                    /* Update F */
#pragma unroll 9
      for (int i = 0; i < 9; i++) F[i] = New_F[i];
      if (volCorrection) {
        logJp = beta * sum_epsilon + logJp;
      }
    } else if (mu != 0) {
      logJp = 0;
      T delta_gamma = epsilon_hat_norm
                      + ((T)3 * lambda + scaled_mu) / scaled_mu * trace_epsilon * yieldSurface;
      T H[3];
      if (delta_gamma <= 0) {  ///< case I: inside the yield surface cone
#pragma unroll 3
        for (int i = 0; i < 3; i++) H[i] = epsilon[i] + cohesion;
      } else {  ///< case III: project to the cone surface
#pragma unroll 3
        for (int i = 0; i < 3; i++)
          H[i] = epsilon[i] - (delta_gamma / epsilon_hat_norm) * epsilon_hat[i] + cohesion;
      }
#pragma unroll 3
      for (int i = 0; i < 3; i++) New_S[i] = expf(H[i]);
      matmul_mat_diag_matT_3D(New_F, U, New_S, V);  // new F_e
                                                    /* Update F */
#pragma unroll 9
      for (int i = 0; i < 9; i++) F[i] = New_F[i];
    }

    /* Elasticity -- Calculate Coefficient */
    T New_S_log[3] = {logf(New_S[0]), logf(New_S[1]), logf(New_S[2])};
    T P_hat[3];

    // T S_inverse[3] = {1.f/S[0], 1.f/S[1], 1.f/S[2]};  // TO CHECK
    // T S_inverse[3] = {1.f / New_S[0], 1.f / New_S[1], 1.f / New_S[2]}; // TO
    // CHECK
    T trace_log_S = New_S_log[0] + New_S_log[1] + New_S_log[2];
#pragma unroll 3
    for (int i = 0; i < 3; i++)
      P_hat[i] = (scaled_mu * New_S_log[i] + lambda * trace_log_S) / New_S[i];

    T P[9];
    matmul_mat_diag_matT_3D(P, U, P_hat, V);
    ///< |f| = P * F^T * Volume
    PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
    PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
    PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
    PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
    PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
    PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
    PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
    PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
    PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
  }

}  // namespace zs