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

        //获取一个编译选项,表示怎么处理IR
        //
        std::unique_ptr<IRNode> clone();//克隆IRNode节点
    };
    class Statement {
      protected:
        std::vector<>
    };

    class IRVisitor {

    };
}
