//
// Created by admin on 2022/5/31.
//
#include "../Pass.h"
#include "../IR/BasicBlock.h"
#include <unordered_map>
#include <unordered_set>
//用来死代码删除
namespace zfx {
    class ActiveVars : public Pass {
      public:
        explicit ActiveVars(Module *m) : Pass(M) {}
        void run() override;
        std::unordered_set<Value *> getLiveOut(BasicBlock *bb);
        std::unordered_set<Value *> getLiveIn(BasicBlock *bb);
        bool isLiveOut() {

        }

      private:

    };
}
