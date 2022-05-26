//
// Created by admin on 2022/5/26.
//
#pragma once

#include <optional>
#include <unordered_set>
#include <memory>

namespace zfx {
    class CFGNode {
      private::
        std::unordered_set<Block *> parent_blocks;
    public:
      Block *block;
      int begin_location, end_location;

      CFGNode *prev_in_the_same_block;
      CFGNode *next_in_the_same_block;

      //
      std::unordered_set<Stmt *> live_gen, live_kill, live_in, live_out;
      std::unordered_set<Stmt *> reach_gen, reach_kill, reach_in, reach_out;
      std::vector<CFGNode *> prev, next;
      CFGNode();

      CFGNode(Block *block, int begin_location,
              int end_location, CFGNode *prev_node_in_same_block);
      static void add_edge(CFGNode *from, CFGNode *to);

      bool empty() const;

      std::size_t size();

      void erase(int location);

      void insert(std::unique_ptr<Stmt> &&stmt, int location);

      void replace_with(int location, std::unique_ptr<Stmt> &&newStmt,
                        bool replace_usage = true) const;

      static bool contain_variable(const std::unordered_set<Stmt *> &stmt_set,
                                   Stmt *var);

      static bool may_contain_variable(const std::unordered_set<Stmt *> &stmt,
                                       Stmt *var);

      bool reach_kill_variable(Stmt *var) const;

      Stmt *get_store_forwarding_data(Stmt *stmt, int position) const;

      //Analyses and optimization inside CFGNode

      void reaching_definition_analysis(bool after_lower_access);

      void live_variable_anaylysis();
    };


    class ControlFlowGraph {
      private:
        void erase(int node_id);
      public:
        struct LiveVarAnalysisConfig {

        };

        std::vector<std::unique_ptr<CFGNode>> nodes;
        const int state_node = 0;
        int final_node{0};

        template<typename... Args>
        CFGNode *push_back(Args...args) {
            nodes.template emplace_back(std::make_unique<CFGNode>(std::forward<Args>(args)...));
            return nodes.back().get();
        }

        void reaching_definition_analysis(bool after_lower_access);

        void live_variable_analysis() {

        }


    };
}
