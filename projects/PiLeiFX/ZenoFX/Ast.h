//
// Created by admin on 2022/5/6.
//

#pragma once

#include "Lexical.h"
#include "Location.h"
#include <iostream>
#include <memory>
#include <string>
#include <any>
namespace zfx {
    enum class Ast_Node_Type {
        Ast_Type_Statement,
        Ast_Type_Declaration,
        Ast_Type_Identifier,
        Ast_Type_Integer_Literal,
        Ast_Type_Binary_Op,
        Ast_Type_Unary_Op
    };
    class AstNode;
    class Variable;
    class Binary;
    class Unary;

    class AstVisitor {
        virtual ~visitor() {


        }
    };


    class AstNode {
      public:
        Ast_Node_Type type ;
        Position beginPos;
        Position endPos;
        bool isErrorNode {false};

        virtual std::any accept(AstVisitor& visitor, std::string additional = "");

    };

    class Statement : public AstNode {
      public:
        Ast_Node_Type type = Ast_Node_Type::Ast_Type_Statement;
        Position beginPos;
        Position endPos;

    };

    class Ast_Identifier {

    };

    class Ast_Binary_Op {
        Op op;

    };

    class Ast_Unary_Op {

    };

    class IntegerLiteral {

    };
}

