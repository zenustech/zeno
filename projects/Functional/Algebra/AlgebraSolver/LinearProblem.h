/**
 * @file LinearProblem.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-04
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */

#ifndef _LINEARPROBLEM_H_
#define _LINEARPROBLEM_H_
#include <Timer.h>
#include <memory>

template<typename VectorType>
class LinearProblem
{
private:
    /* data */
public:
    LinearProblem(/* args */){

    }

    // TODO : Virtual function? 
    virtual ~LinearProblem(){

    }

    // Function called by matrix free linear solver.
    // must be supplied by the user.
    virtual void form(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> Ax) = 0;

};

#endif