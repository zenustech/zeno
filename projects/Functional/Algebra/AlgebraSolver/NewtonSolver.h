/**
 * @file NewtonSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-09
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */

#ifndef _NEWTON_SOLVER_
#define _NEWTON_SOLVER_

#include <map>
#include <vector>
#include <memory>
#include <cmath>
#include "NonlinearProblem.h"
#include "NewtonSolver.h"
#include "NonlinearSolver.h"
template<typename VectorType>
class NewtonSolver : public NonlinearSolver<VectorType>
{
private:

    // using NonlinearSolver<VectorType>::data_;

public:


    NewtonSolver(std::shared_ptr<LinearSolver<VectorType>> _linear_solver) 
        : NonlinearSolver<VectorType>::NonlinearSolver(_linear_solver)
    {
    }

    virtual std::string method() const final {
        return "Default newton solver";
    };

    // TODO : param b is useless here.
    virtual std::pair<bool, std::pair<double, int>> Solve(
        std::shared_ptr<NonlinearProblem<VectorType>> nonlinear_problem, 
        std::shared_ptr<VectorType> x0, 
        std::shared_ptr<const VectorType> b) final {
        
        auto linear_solver = this->get_linear_solver();
        CHECK_F(linear_solver->problem_size() == x0->size() && b->size() == x0->size(), "Wrong size!");

        // NOTE : It will allocate memory for nonlinear_problem. For efficiency, it will not reallocate memory when VectorType = GpuVector if vectors exist.
        nonlinear_problem->resize(x0->size());

        auto xk = nonlinear_problem->get_xk();
        auto rhs = nonlinear_problem->get_rhs();
        auto delta_x = nonlinear_problem->get_delta_x();
        auto residual = nonlinear_problem->get_residual();

        double norm_b = std::sqrt(b->inner(*b));
        double norm_r;
        int    iter = 0;
        
        // NOTE : Set initial value for newton iteration.
        *xk = *x0;
        
        do{
            iter++;         
            // NOTE : rhs = h(xk), here we solve A(-dx)=rhs, and finnally xk = xk - (-dx);
            nonlinear_problem->Residual(xk, rhs);

            // NOTE : initial guess for the linear solver.
            *delta_x = *rhs;

            // solve delta_x, which is (-dx) actually.
            auto linear_result = linear_solver->Solve(nonlinear_problem, delta_x, rhs);
            if (linear_result.first) LOG_F(WARNING, "Linear solver succeed.");
            else                     LOG_F(WARNING, "Linear solver failed.");
            LOG_F(WARNING, "residual : %lf, iter : %d", linear_result.second.first, linear_result.second.second);
            // LOG_F(WARNING, "xk: %.8lf, %.8lf", xk[0], xk[1]);
            
            // update xk = xk - (-dx);
            xk->axpy(-1.0,*delta_x, *xk);
            *x0 = *xk;

            // calculate the residual.
            nonlinear_problem->Residual(residual);
            norm_r = std::sqrt(residual->inner(*residual));
            LOG_F(WARNING, "norm_r : %lf", norm_r);
        
            // TODO : assuming norm_e = norm_r/norm_b
            if (norm_r < this->max_nonlinear_tolerance) return std::make_pair(true, std::make_pair(norm_r,iter));

        }while(iter < this->max_nonlinear_iteration);
        
        return std::make_pair(false, std::make_pair(norm_r,iter));
    };
    
    virtual ~NewtonSolver(){
        
    }
};

#endif