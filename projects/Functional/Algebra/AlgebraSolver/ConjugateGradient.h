/**
 * @file ConjugateGradient.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-04
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */

#ifndef _CONJUGATEGRADIENT_H_
#define _CONJUGATEGRADIENT_H_

#include <vector>
#include "LinearSolver.h"

class ConjugateGradient : public LinearSolver
{
private:
    /* data */
    std::vector<double> m, z, s, r, temp;

public:
    ConjugateGradient(/* args */){

    }

    ~ConjugateGradient(){
        
    }

    // Initial approximation.
    virtual void Initialize(size_t n) final {

        // TODO :

    }

    // Solve.
    virtual std::pair<bool, std::pair<double, int>> Solve(
        LinearProblem &forceOperator, 
        std::vector<double> &forceDofs, 
        const std::vector<double> &rhs) final {
        
        // TODO :
        return std::make_pair(false, std::make_pair(0.0,0));
    }
};

// TODO : Nonlinear solver.

#endif