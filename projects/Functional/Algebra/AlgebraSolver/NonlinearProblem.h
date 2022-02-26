/**
 * @file NonlinearProblem.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-09
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */

#ifndef _NONLINEAR_PROBLEM_
#define _NONLINEAR_PROBLEM_

#include "LinearSolver.h"
#include "LinearProblem.h"
#include "StdVector.h"

template<typename VectorType>
class NonlinearProblem : public LinearProblem<VectorType>
{
private:
    
    VectorType xk;
    VectorType rhs;
    VectorType delta_x;
    VectorType residual;
    
    VectorType r1;
    VectorType r2;
    VectorType xk_plus_epsilon_x;

    // used to calculate Jx = (h(x+epsilon*x0)-h(x))/epsilon
    double epsilon = 1e-6;

public:

    VectorType& get_xk()        { return xk; } 
    VectorType& get_rhs()       { return rhs; } 
    VectorType& get_delta_x()   { return delta_x; } 
    VectorType& get_residual()  { return residual; } 
 
    void resize(size_t __size){

        xk.resize(__size);
        rhs.resize(__size);
        delta_x.resize(__size);
        residual.resize(__size);
        r1.resize(__size);
        r2.resize(__size);
        xk_plus_epsilon_x.resize(__size);

    }
    
    void set_epsilon(double a) { epsilon = a; }
    double get_epsilon() const { return epsilon; }

    NonlinearProblem(){
        
    }

    virtual void form(const VectorType& x, VectorType& r) final {
        
        // Timer timer("function residual in class MyProblem");
        LOG_F(WARNING, "Nonlinear form.\n\n");
        
        CHECK_F(x.size() == r.size(), "Wrong size.");
        CHECK_F(xk.size() == r.size(), "Wrong size.");



        // z = a*x + y
        // z.axpy(a,x,y);
        // x_k + \epsilon x = xk + epsilon * x;
        xk_plus_epsilon_x.axpy(epsilon, x, xk);

        // calculate residuals
        Residual(xk, r1);
        Residual(xk_plus_epsilon_x, r2);

        // (h(x0+e*x)-h(x0))/e
        // r = (r2 - r1)/epsilon
        // divided into two steps :
        // r = r2 - r1
        // r = r*(1/epsilon) = r + (-1+1/epsilon)*r

        // z = a*x + y
        // z.axpy(a,x,y);
        r.axpy(-1,r1,r2);

        // x = a*x + y
        // x.axpy(a,y);
        r.axpy(1.0/epsilon-1,r);
    }


    virtual void Residual(const VectorType& x, VectorType& r) = 0;

    void Residual(VectorType& r){
        Residual(xk, r);
    }
    
    virtual ~NonlinearProblem(){

    }
};


#endif