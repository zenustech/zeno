/**
 * @file NonlinearSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-05 create
 * @date 2021-12-09 define NonlinearSolver
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */


// 
// Despite the perception that it is fragile or scary, Newton’s method works well on many nonlinear systems 
// if one has a good initial iterate x0 and one adds some important protections. A line search, which 
// sometimes moves a shorter distance than computed, is such a “protection”.
//                                                              -- PETSc for Partial Differenctial Equations
//
// Here is a list of nonlinear solver copied from dolfin/nls/PETScSNESSolver.h
// 
// { 
//     {"default",      {"default SNES method",                                  ""}},
//     {"newtonls",     {"Line search method",                                   SNESNEWTONLS}},
//     {"newtontr",     {"Trust region method",                                  SNESNEWTONTR}},
//     {"ngmres",       {"Nonlinear generalised minimum residual method",        SNESNGMRES}},
//     {"nrichardson",  {"Richardson nonlinear method (Picard iteration)",       SNESNRICHARDSON}},
//     {"vinewtonrsls", {"Reduced space active set solver method (for bounds)",  SNESVINEWTONRSLS}},
//     {"vinewtonssls", {"Reduced space active set solver method (for bounds)",  SNESVINEWTONSSLS}},
//     {"qn",           {"Limited memory quasi-Newton",                          SNESQN}},
//     {"ncg",          {"Nonlinear conjugate gradient method",                  SNESNCG}},
//     {"fas",          {"Full Approximation Scheme nonlinear multigrid method", SNESFAS}},
//     {"nasm",         {"Nonlinear Additive Schwartz",                          SNESNASM}},
//     {"anderson",     {"Anderson mixing method",                               SNESANDERSON}},
//     {"aspin",        {"Additive-Schwarz Preconditioned Inexact Newton",       SNESASPIN}},
//     {"ms",           {"Multistage smoothers",                                 SNESMS}}
// };
//

#ifndef _NONLINEAR_SOLVER_
#define _NONLINEAR_SOLVER_

#include <map>
#include <vector>
#include <memory>
#include "NonlinearProblem.h"
#include "LinearProblem.h"

template<typename VectorType>
class NonlinearSolver
{

// TODO : private members.
protected:
    // Set the problem size.
    size_t size;

    // control the convergence.
    int max_nonlinear_iteration = 1000;
    double max_nonlinear_tolerance = 1e-4;
    std::shared_ptr<LinearSolver<VectorType>> linear_solver;

public:

    /**
     * @brief Get the size object
     * @return size_t 
     */
    size_t get_size() const { return size; }

    /**
     * @brief Get the linear solver object
     * @return std::shared_ptr<LinearSolver> 
     */
    std::shared_ptr<LinearSolver<VectorType>> get_linear_solver() const { return linear_solver; }


    NonlinearSolver(std::shared_ptr<LinearSolver<VectorType>> _linear_solver) : linear_solver(_linear_solver)
    {
    }

    virtual std::string method() const = 0;

    virtual std::pair<bool, std::pair<double, int>> Solve(
        std::shared_ptr<NonlinearProblem<VectorType>> nonlinear_problem, 
        std::shared_ptr<VectorType> x0, 
        std::shared_ptr<const VectorType> b) = 0;
    
    // 类声明外部的说明符无效
    virtual ~NonlinearSolver(){

    }
};

#endif