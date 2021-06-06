#include <krylov_solvers.h>
#include <blas_wrapper.h>
#include <cassert>

//============================================================================
KrylovSolverStatus CG_Solver::
solve(const LinearOperator &A, const double *rhs, double *result,
      const LinearOperator *preconditioner, bool use_given_initial_guess)
{
    const int n=A.m;
    assert(A.n==n);
    assert(preconditioner==0 || (preconditioner->m==n && preconditioner->n==n));
    if((int)s.size()!=n){
        r.resize(n);
        z.resize(n);
        s.resize(n);
    }
    // convergence tolerance
    double tol=tolerance_factor*BLAS::abs_max(n, rhs);
    // initial guess
    if(use_given_initial_guess){
        A.apply_and_subtract(result, rhs, &r[0]);
    }else{
        BLAS::set_zero(n, result);
        BLAS::copy(n, rhs, &r[0]);
    }
    // check instant convergence
    iteration=0;
    residual_norm=BLAS::abs_max(r);
    if(residual_norm==0) return status=KRYLOV_CONVERGED;
    // set up CG
    double rho;
    if(preconditioner) preconditioner->apply(r, z); else BLAS::copy(r, z);
    rho=BLAS::dot(r, z);
    if(rho<=0 || rho!=rho) return status=KRYLOV_BREAKDOWN;
    BLAS::copy(z, s);
    // and iterate
    for(iteration=1; iteration<max_iterations; ++iteration){
        double alpha;
        A.apply(s, z); // reusing z=A*s
        double sz=BLAS::dot(s, z);
        if(sz<=0 || sz!=sz) return status=KRYLOV_BREAKDOWN;
        alpha=rho/sz;
        BLAS::add_scaled(n, alpha, &s[0], result);
        BLAS::add_scaled(-alpha, z, r);
        residual_norm=BLAS::abs_max(r);
        if(residual_norm<=tol) return status=KRYLOV_CONVERGED;
        if(preconditioner) preconditioner->apply(r, z); else BLAS::copy(r, z);
        double rho_new=BLAS::dot(r, z);
        if(rho_new<=0 || rho_new!=rho_new) return status=KRYLOV_BREAKDOWN;
        double beta=rho_new/rho;
        BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
        rho=rho_new;
    }
    return status=KRYLOV_EXCEEDED_MAX_ITERATIONS;
}

//============================================================================
KrylovSolverStatus MINRES_CR_Solver::
solve(const LinearOperator &A, const double *rhs, double *result,
      const LinearOperator *preconditioner, bool use_given_initial_guess)
{
    const int n=A.m;
    assert(A.n==n);
    assert(preconditioner==0 || (preconditioner->m==n && preconditioner->n==n));
    if((int)s.size()!=n){
        r.resize(n);
        z.resize(n);
        q.resize(n);
        s.resize(n);
        t.resize(n);
    }
    // convergence tolerance
    double tol=tolerance_factor*BLAS::abs_max(n, rhs);
    // initial guess
    if(use_given_initial_guess){
        A.apply_and_subtract(result, rhs, &r[0]);
    }else{
        BLAS::set_zero(n, result);
        BLAS::copy(n, rhs, &r[0]);
    }
    // check instant convergence
    iteration=0;
    residual_norm=BLAS::abs_max(r);
    if(residual_norm==0) return status=KRYLOV_CONVERGED;
    // set up CR
    double rho;
    if(preconditioner) preconditioner->apply(r, z); else BLAS::copy(r, s);
    A.apply(s, t);
    rho=BLAS::dot(r, t);
    if(rho==0 || rho!=rho) return status=KRYLOV_BREAKDOWN;
    // and iterate
    for(iteration=1; iteration<max_iterations; ++iteration){
        double alpha;
        double tt=BLAS::dot(t, t);
        if(tt==0 || tt!=tt) return status=KRYLOV_BREAKDOWN;
        alpha=rho/tt;
        BLAS::add_scaled(n, alpha, &s[0], result);
        BLAS::add_scaled(-alpha, t, r);
        residual_norm=BLAS::abs_max(r);
        if(residual_norm<=tol) return KRYLOV_CONVERGED;
        if(preconditioner) preconditioner->apply(r, z);
        else               BLAS::copy(r, z);
        A.apply(z, q);
        double rho_new=BLAS::dot(r, q);
        if(rho_new==0 || rho_new!=rho_new) return KRYLOV_BREAKDOWN;
        double beta=rho_new/rho;
        BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
        BLAS::add_scaled(beta, t, q); t.swap(q); // t=beta*t+q
        rho=rho_new;
    }
    return KRYLOV_EXCEEDED_MAX_ITERATIONS;
}

//============================================================================
KrylovSolverStatus CGNR_Solver::
solve(const LinearOperator &A, const double *rhs, double *result,
      const LinearOperator *preconditioner, bool use_given_initial_guess)
{
    const int m=A.m, n=A.n;
    assert(preconditioner==0 || (preconditioner->m==n && preconditioner->n==n));
    if((int)s.size()!=n){
        r.resize(n);
        z.resize(n);
        s.resize(n);
        u.resize(m);
    }
    // convergence tolerance
    A.apply_transpose(rhs, &r[0]); // form A^T*rhs in r
    double tol=tolerance_factor*BLAS::abs_max(r);
    // initial guess
    if(use_given_initial_guess){
        A.apply_and_subtract(result, rhs, &u[0]);
        A.apply_transpose(u, r);
    }else{
        BLAS::set_zero(n, result);
    }
    // check instant convergence
    iteration=0;
    residual_norm=BLAS::abs_max(r);
    if(residual_norm==0) return status=KRYLOV_CONVERGED;
    // set up CG
    double rho;
    if(preconditioner) preconditioner->apply(r, z); else BLAS::copy(r, z);
    rho=BLAS::dot(r, z);
    if(rho<=0 || rho!=rho) return status=KRYLOV_BREAKDOWN;
    BLAS::copy(z, s);
    // and iterate
    for(iteration=1; iteration<max_iterations; ++iteration){
        double alpha;
        A.apply(s, u);
        A.apply_transpose(u, z);
        double sz=BLAS::dot(u, u);
        if(sz<=0 || sz!=sz) return status=KRYLOV_BREAKDOWN;
        alpha=rho/sz;
        BLAS::add_scaled(n, alpha, &s[0], result);
        BLAS::add_scaled(-alpha, z, r);
        residual_norm=BLAS::abs_max(r);
        if(residual_norm<=tol) return status=KRYLOV_CONVERGED;
        if(preconditioner) preconditioner->apply(r, z); else BLAS::copy(r, z);
        double rho_new=BLAS::dot(r, z);
        if(rho_new<=0 || rho_new!=rho_new) return status=KRYLOV_BREAKDOWN;
        double beta=rho_new/rho;
        BLAS::add_scaled(beta, s, z); s.swap(z); // s=beta*s+z
        rho=rho_new;
        
        //      if ( iteration % 5000 == 0 )
        //      {
        //         std::cout << "CGNR_Solver --- residual_norm: " << residual_norm << std::endl;
        //      }
        
    }
    return status=KRYLOV_EXCEEDED_MAX_ITERATIONS;
}

