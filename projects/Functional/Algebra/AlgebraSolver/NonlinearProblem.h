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

    std::shared_ptr<VectorType> xk;
    std::shared_ptr<VectorType> rhs;
    std::shared_ptr<VectorType> delta_x;
    std::shared_ptr<VectorType> residual;
    
    std::shared_ptr<VectorType> r1;
    std::shared_ptr<VectorType> r2;
    std::shared_ptr<VectorType> xk_plus_epsilon_x;

    // used to calculate Jx = (h(x+epsilon*x0)-h(x))/epsilon
    double epsilon = 1e-6;

public:

    std::shared_ptr<VectorType> get_xk()        { return xk; } 
    std::shared_ptr<VectorType> get_rhs()       { return rhs; } 
    std::shared_ptr<VectorType> get_delta_x()   { return delta_x; } 
    std::shared_ptr<VectorType> get_residual()  { return residual; } 
 
    void resize(size_t __size){
        xk = std::make_shared<VectorType>();
        rhs = std::make_shared<VectorType>();
        delta_x = std::make_shared<VectorType>();
        residual = std::make_shared<VectorType>();
        r1 = std::make_shared<VectorType>();
        r2 = std::make_shared<VectorType>();
        xk_plus_epsilon_x = std::make_shared<VectorType>();

        xk->resize(__size);
        rhs->resize(__size);
        delta_x->resize(__size);
        residual->resize(__size);
        r1->resize(__size);
        r2->resize(__size);
        xk_plus_epsilon_x->resize(__size);

    }
    
    void set_epsilon(double a) { epsilon = a; }
    double get_epsilon() const { return epsilon; }

    NonlinearProblem(){
        
    }

    virtual void form(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> r) final {

        // Timer timer("function residual in class MyProblem");
        // LOG_F(WARNING, "Nonlinear form.\n\n");

        CHECK_F(x->size() == r->size() && xk->size() == r->size(), "Wrong size.");
        
        xk_plus_epsilon_x->axpy(epsilon, *x, *xk);

        Residual(xk, r1);
        Residual(xk_plus_epsilon_x, r2);

        r->axpy(-1,*r1,*r2);
        r->axpy(1.0/epsilon-1,*r);
    }

    virtual void Residual(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> r) = 0;

    void Residual(std::shared_ptr<VectorType> r) {
        Residual(xk, r);
    }
    
    virtual ~NonlinearProblem(){

    }
};


#endif