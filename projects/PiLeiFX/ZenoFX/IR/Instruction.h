//
// Created by admin on 2022/6/7.
//

#pragma once
#include "./User.h"
#include "./BasicBlock.h"
#include "./Module.h"
#include <cstdint>
#include <utility>

namespace zfx {
    class BasicBlock;
    class Module;
    class Instruction : public User {
      public:
        Instruction(const Instruction &) = delete;
        Instruction& operator= (const Instruction &) = delete;

        const BasicBlock* getParent() const {

        }

        BasicBlock* getParent() {

        }

        const Module* getModule() const {

        }

        Module* getModule() {

        }


        /*
         * 留几个bool函数来判断指令到底是属于啥类型的
         * */

        inline bool isUnaryOp() const {

        }

        inline bool isBinaryOp() const {

        }

        inline bool isTenaryOp() const {

        }

        void insertBefore(Instruction *InsertPos);

        void insertAfter(Instruction *InsertPos);

        void moveBefore(Instruction *MovBefore);


      private:
        BasicBlock* Parent;

    };

    //接下来定义几个指令继承自Instruction

}
