#ifndef KRYLOV_SOLVERS_H
#define KRYLOV_SOLVERS_H

#include <linear_operator.h>

enum KrylovSolverStatus{
    KRYLOV_CONVERGED,
    KRYLOV_EXCEEDED_MAX_ITERATIONS,
    KRYLOV_BREAKDOWN
};

//============================================================================
// Only guaranteed for symmetric positive definite systems.
// Singular systems may be solved, but round-off error may cause problems,
// and if the system is inconsistent convergence will not be automatically detected.
struct CG_Solver
{
    double tolerance_factor;
    unsigned int max_iterations;
    double residual_norm; // we use the infinity norm
    unsigned int iteration;
    KrylovSolverStatus status;
    std::vector<double> r, z, s;
    
    CG_Solver(void)
    : tolerance_factor(1e-9), max_iterations(100), residual_norm(0), iteration(0), status(KRYLOV_CONVERGED), r(0), z(0), s(0)
    {}
    
    // Attempt to solve A*result=rhs for result.
    // Sets residual_norm, iteration, and status (and also returns status)
    // If optional preconditioner given, use it.
    // If "use_given_initial_guess" is false (default), take initial guess to be all zeros;
    //  if true, instead use whatever is in result at the moment. 
    KrylovSolverStatus solve(const LinearOperator &A, const double *rhs, double *result,
                             const LinearOperator *preconditioner=0, bool use_given_initial_guess=false);
};

//============================================================================
// MINRES using the Conjugate Residual (CR) algorithm.
// May work on symmetric indefinite problems, but is vulnerable to breakdown
// except in the positive definite case.
struct MINRES_CR_Solver
{
    double tolerance_factor;
    unsigned int max_iterations;
    double residual_norm; // we use the infinity norm
    unsigned int iteration;
    KrylovSolverStatus status;
    std::vector<double> r, z, q, s, t;
    
    MINRES_CR_Solver(void)
    : tolerance_factor(1e-9), max_iterations(100), residual_norm(0), iteration(0), status(KRYLOV_CONVERGED), r(0), z(0), q(0), s(0), t(0)
    {}
    
    // Attempt to solve A*result=rhs for result.
    // Sets residual_norm, iteration, and status (and also returns status)
    // If optional preconditioner given, use it.
    // If "use_given_initial_guess" is false (default), take initial guess to be all zeros;
    //  if true, instead use whatever is in result at the moment. 
    KrylovSolverStatus solve(const LinearOperator &A, const double *rhs, double *result,
                             const LinearOperator *preconditioner=0, bool use_given_initial_guess=false);
};

//============================================================================
// CGNR (Conjugate Gradient applied to the Normal Equations)
// Should be able to solve min ||b-Ax|| for A with full column rank,
// including the case Ax=b for general non-singular A. Rank-deficient problems
// may still work.
struct CGNR_Solver
{
    double tolerance_factor;
    unsigned int max_iterations;
    double residual_norm; // we use the infinity norm of the residual in the normal equations, A^T*(b-A*x)
    // Please note --- this is not the same as the residual of the original, b-A*x
    unsigned int iteration;
    KrylovSolverStatus status;
    std::vector<double> r, z, s, u;
    
    CGNR_Solver(void)
    : tolerance_factor(1e-9), max_iterations(100), residual_norm(0), iteration(0), status(KRYLOV_CONVERGED), r(0), z(0), s(0), u(0)
    {}
    
    // Attempt to solve min ||rhs-A*result|| (or A*result=rhs for nonsingular A) for result.
    // Sets residual_norm, iteration, and status (and also returns status)
    // If optional preconditioner given, use it; it should precondition A, so that A*M is close to orthogonal.
    // If "use_given_initial_guess" is false (default), take initial guess to be all zeros;
    //  if true, instead use whatever is in result at the moment. 
    KrylovSolverStatus solve(const LinearOperator &A, const double *rhs, double *result,
                             const LinearOperator *preconditioner=0, bool use_given_initial_guess=false);
};

#endif
