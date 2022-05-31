//
// Created by admin on 2022/5/27.
//

#include "Module.h"
#include <list>
#include <set>
#include <string>
#include <vector>


namespace zfx {
    class BasicBlock {
      public:
        static BasicBlock *create(Module *m, const std::string& name) {

            return new BasicBlock(m, name);
        }

        Module* getModule() ;

        virtual std::string print() override;

        //api for cfg
        std::list<BasicBlock *> &get_pre_basic_blocks() {

        }

        std::list<BasicBlock *>
      private:
        explicit BasicBlock(Module *m, const std::string &name);
        std::list<BasicBlock *> pre_bbs;
        std::list<BasicBlock *> succ_bbs;

    };
}
