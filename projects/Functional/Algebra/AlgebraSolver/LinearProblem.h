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
    virtual void form(const VectorType& x, VectorType& r) = 0;

};

#endif