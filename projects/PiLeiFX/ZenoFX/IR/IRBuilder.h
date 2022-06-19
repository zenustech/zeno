//
// Created by admin on 2022/6/7.
//

#pragma once

#include "./BasicBlock.h"
#include "./Instruction.h"

namespace zfx {
    class IRBuilder {
      public:
        IRBuilder() = default;
        IRBuilder(BasicBlock *bb) : bb(bb) {}

        void reset() ;
        //把所有的指令清空

        inline BasicBlock *getInsertBlock() {
            return bb;
        }

        void setInsertPoint(BasicBlock *bb) const {
            this->bb = bb;
        }

        //接下来是创建指令
      private:
        BasicBlock *bb;
    };
}
