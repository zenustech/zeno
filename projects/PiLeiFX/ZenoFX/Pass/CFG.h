//
// Created by admin on 2022/5/26.
//
#pragma once

#include "../Pass.h"
#include <memory>
#include <optional>
#include <unordered_set>

namespace zfx {
   class CFG : public Pass {
       //build CFG

     public:
       explicit CFG(Module *m) : Pass(m) {}

       void run() final;

       BasicBlock * getEntryBB(){
           return entry;
       }

       //get those unreachable block
       std::set<BasicBlock *> getUnreachableBB {
           return unreachable;
       };

       //Get the predecessors of a basic block (all blocks that can reach the basic block in one step)
       std::set<BasicBlock *>  getPrevBB(BasicBlock *b) {

       }

       std::set<BasicBlock *> getSuccBB(BasicBlock *b) {

       }

       std::set<BasicBlock *> getTerminators() {
           //
       }
     private:

       void cleanAll() {
           //clean CFG;
           entry = nullptr;
           successor_map.clear();
           precessor_map.clear();
           term_set.clear();
       }

       BasicBlock *entry{nullptr};
       std::map<BasicBlock *, std::set<BasicBlock *>> successor_map;
       std::map<BasicBlock *, std::set<BasicBlock *>> precessor_map;
       std::set<BasicBlock *> unreachable;//some unreachable BasicBlock used as dead code elimination
       std::set<BasicBlock *> term_set;
   };
}
