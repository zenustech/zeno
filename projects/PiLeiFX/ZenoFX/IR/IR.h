//
// Created by admin on 2022/5/7.
//
#pragma once
#include<memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
/*
 * zfx中statement就相当于一条一条的Instruction
 * */

namespace zfx {
    class Stmt ;

    class IRNode {
      public:

        virtual IRNode* get_parent() const = 0;

        virtual IRNode* get_ir_root();
        virtual ~IRNode() = default;
    };
    class Statement {
      protected:
        std::vector<>
    };

    class IRVisitor {

    };
}
