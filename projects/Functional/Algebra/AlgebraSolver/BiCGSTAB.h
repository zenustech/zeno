/**
 * @file BiCGSTAB.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-04
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */


// Algrithm described on page 21 of thesis
//
// @article{2016Comparison,
//   title={Comparison of Some Preconditioners for the Incompressible Navier-Stokes Equations},
//   author={ He, X.  and  Vuik, C. },
//   journal={Numerical Mathematics Theory Methods & Applications},
//   volume={9},
//   number={02},
//   pages={239-261},
//   year={2016},
// }

// NOTE : Handle the breakdowns
//        (t_j, t_j) = 0, (t_j, s_j) = 0, \sqrt{(t_{j+1}, t_{j+1})} = 0, \beta = 0

// TODO : Preconditioner, it might be difficult for matrix-free solver.

#ifndef _BICGSTAB_H_
#define _BICGSTAB_H_

#include "LinearSolver.h"
#include <iostream>
#include <cmath>

template<typename VectorType>
class BiCGSTAB : public LinearSolver<VectorType>
{
private:
    double rho, alpha, omega, beta;
    std::shared_ptr<VectorType> r, v, s, t, p, r0, temp;
public:


    // Allocate memory.
    virtual void Initialize(size_t n) final {

        this->_problem_size = n;

        r = std::make_shared<VectorType>();
        r0 = std::make_shared<VectorType>();
        v = std::make_shared<VectorType>();
        s = std::make_shared<VectorType>();
        t = std::make_shared<VectorType>();
        p = std::make_shared<VectorType>();
        temp = std::make_shared<VectorType>();

        r->resize(n);
        r0->resize(n);
        v->resize(n);
        s->resize(n);
        t->resize(n);
        p->resize(n);
        temp->resize(n);

    }

    // NOTE : the definition of relative_error : r = b - Av
    // The definition of Ae = −r, e is difficult to solve.
    // we can give a bound of error : $\|e\|\le\|A^{-1}\|\|e\|$
    // relative error norms relate to the conditioning of A:
    // $\frac{1}{\kappa(A)} \frac{\|\mathbf{r}\|}{\|\mathbf{b}\|} \leq \frac{\|\mathbf{e}\|}{\|\mathbf{u}\|} \leq \kappa(A) \frac{\|\mathbf{r}\|}{\|\mathbf{b}\|}$
    // Which means it can be controled by $ \frac{\|r\|}{\|b\|} $.
    //
    // NOTE : We use 2-norm of relative_error as return instead of relative error.
    virtual std::pair<bool, std::pair<double, int>> Solve(
        std::shared_ptr<LinearProblem<VectorType>> linear_problem, 
        std::shared_ptr<VectorType> x0, 
        std::shared_ptr<const VectorType> b) final {
        
        CHECK_F(r->size() == x0->size() && x0->size() == b->size(), "Wrong size!");
        
        // r0 = b - Ax
        linear_problem->form(x0,temp);   // temp = Ax
        r0->axpy(-1.0,*temp,*b);         // r0 = -1.0*temp + b;
        *r = *r0;                          // r = r0
        *p = *r0;                          // p = r0

        int iter = 0;
        double norm_b = std::sqrt(b->inner(*b));
        if(norm_b == 0) {
            *x0 = 0.0;
            return std::make_pair(true, std::make_pair(0.0, iter));
        }
        double relative_error = std::sqrt(r->inner(*r))/norm_b;

        double delta, gamma;

        // start the iterations
        while (relative_error > LinearSolver<VectorType>::get_tolerance() && iter < LinearSolver<VectorType>::get_max_iteration())
        {
            LOG_F(INFO, "BiCGSTAB solver(start) : iteration %d.", iter);    

            linear_problem->form(p,v);              // v = Ap
            rho = r->inner(*r0);                      // rho = (r,r0)
            alpha = rho/v->inner(*r0);                // alpha = rho/(v,r0)

            if (!std::isnormal(alpha)) return std::make_pair(true, std::make_pair(relative_error,iter));

            s->axpy(-alpha,*v,*r);                     // s = r - apha * v;
            linear_problem->form(s,t);              // t = As
            
            gamma = t->inner(*t);
            delta = t->inner(*s);
            omega = delta / gamma;                  // omega = (t,s)/(t,t)
            
            if (!std::isnormal(omega)) return std::make_pair(true, std::make_pair(relative_error,iter));
                                                    // x = x + alpha*p + omega*s
            temp->axpy(omega,*s,*x0);                  // temp = x + omega*s
            x0->axpy(alpha,*p,*temp);                  // x = temp + alpha*p
                                                    // r = s - omega*t;
            *temp = *r;                               // r_j
            r->axpy(-omega, *t, *s);                   // r_{j+1}

            relative_error = std::sqrt(r->inner(*r))/norm_b;
            LOG_F(INFO, "BiCGSTAB solver(end) : iteration %d, relative_error norm : %.12lf.", iter, relative_error);    
            if (relative_error < LinearSolver<VectorType>::get_tolerance()) return std::make_pair(true, std::make_pair(relative_error,iter+1));

            beta = r->inner(*r0)/temp->inner(*r0)*alpha/omega;  // (r,r0)/(temp,r0)*alpha/omega

            if (!std::isnormal(beta)) return std::make_pair(true, std::make_pair(relative_error,iter+1));

            temp->axpy(-omega, *v, *p);                 // p = r + beta*(p-omega*v);
            p->axpy(beta, *temp, *r);

            iter++;
        }
        return std::make_pair(false, std::make_pair(relative_error,iter));

    }

    BiCGSTAB(size_t n){
        Initialize(n);
    }

    ~BiCGSTAB(){

    }
};

#endif