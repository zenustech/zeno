//
// Created by admin on 2022/5/7.
//

#pragma once
#include "Ast.h"
#include "parser.h"
#include <string>
#include <sstream>
namespace zfx {
    class IRNode {

    };

    class BinaryNode {

    };

    class UnaryNode {

    };

    class Use {

    };

    enum class Value_Type {

    };

    class Value {
      public:
        Value_Type type; //
        Use* use = nullptr;

        bool visited = false;
        bool live = true;

        Value(Value_Type type) : type(type) {}


    };

    class Constant : public Value {

    };

    class Global : public Value {

    };

    class Instruction_Binary : public Value {

    };

    class Instruction_Unary : public Value {

    };

    class Instruction_Branch : public Value {

    };

    class Function_Call : Value {

    };

    struct IRVisitor : public AstVisitor {

    };
}
