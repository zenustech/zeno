//
// Created by admin on 2022/5/14.
//
#pragma once
#include "Ast.h"
#include <memory>
/*Statement you can think of it as one High IR based ast,
 * and for the time
 * */

namespace zfx {
    struct Statement {
        int id;//
        int dim;//dimensionality;
        explicit Statement(int id , int dim) : id(id), dim(dim) {

        }

        std::string print() {

        }


    };



}


