//
// Created by admin on 2022/5/25.
//

#pragma once

#include "IR/Module.h"

namespace  zfx {
    class Pass {
      public:
        Pass(Module *m) {}
       virtual ~Pass() ;
       virtual void run() = 0;
     private:
       //this is zfx top-level data structure
       Module *m;
    };

    class PassManger {
      public:
      PassManger(Module *m) : m(m) {}
      template<typename PassType>
      void add_pass(bool print_ir = false) {
          passes.push_back(std::pair<Pass*, bool>(new Pass(m), print_ir));
      }

      void run() {
          for (auto Pass : passes) {
              Pass.first->run();
              if (Pass.second) {
               //print ir;
              }
          }
      }

    private:
      std::vector<std::pair<Pass*, bool>> passes;
      Module *m;
    };
}
