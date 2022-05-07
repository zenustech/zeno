//
// Created by admin on 2022/5/6.
//

#pragma once

#include <iostream>
#include <memory>
#include <string>
namespace zfx {

    class AstNode;
    class Variable;
    class Binary;
    class Unary;

    class AstVisitor {
        virtual ~visitor() {

        }
    };

    class AstTypeArray {

    };

    class AstNode {
      public:

        virtual void dump() {}

    };
    /*
    struct  AST {
        using Iter = typename std::vector<std::string>::iterator;

        Iter iter;
        std::string Token;


        void dump() {}

    };
     */

}

