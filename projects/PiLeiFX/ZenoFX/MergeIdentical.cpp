//
// Created by admin on 2022/5/14.
//
#include "Ast.h"
#include <map>
#include <functional>

namespace zfx {
    struct MergeIdentical  {
        using visit_emit_types = std::tuple<>;

        void visit() {

        }


    };

    std::unique_ptr<> apply_merge_identical() {
        MergeIdentical visitor;

    }
}