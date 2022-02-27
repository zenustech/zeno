/**
 * @file LinearSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-04
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */

#ifndef _LINEARSOLVER_H_
#define _LINEARSOLVER_H_

#include"LinearProblem.h"
#include<loguru/loguru.hpp>
#include <memory>

// TODO : Take it as an abstract class.
template<typename VectorType>
class LinearSolver
{
protected:
    /* data */
    size_t _problem_size;

public:
    size_t problem_size() const { return _problem_size; }

    // TODO : How to use initialize inside constructor.
    LinearSolver()
    {

    }

    virtual ~LinearSolver()
    {
    }

    void SetMaxIteration(int iter) { max_linear_iteration = iter;}

    int get_max_iteration() const { return max_linear_iteration;}

    void set_tolerance(double tol) { max_linear_tolerance = tol;}

    double get_tolerance() const { return max_linear_tolerance;}

    /**
     * @brief Allocate memories for all variables used in the iteration.
     * @param n 
     */
    virtual void Initialize(size_t n) = 0;

    // 
    // TODO: is it better to use smart pointer for linear_problem?
    //       FEniCS uses const reference in such situation.
    //       Xinxin also uses reference.
    /**
     * @brief Solve the linear problem with initial guess "x0" and right hand side "b"
     * @param linear_problem 
     * @param x0
     * @param b
     * @return std::pair<bool, std::pair<double, int>> 
     */
    virtual std::pair<bool, std::pair<double, int>> Solve(
        std::shared_ptr<LinearProblem<VectorType>> linear_problem, 
        std::shared_ptr<VectorType> x0, 
        std::shared_ptr<const VectorType> b) = 0;

private:
    int max_linear_iteration = 1000;
    double max_linear_tolerance = 1e-5;
};



#endif