//
// Created by admin on 2022/5/14.
//

#include "Ast.h
#include <map>
#include <functional>
/*
 * 这里我们用了一个最简单的bool 栈来判断if else 结构语句是否匹配成功
 *
 * */
namespace zfx {
    struct ControlCheck {
        using visit_emit_types = std::tuple<>;

        void visit() {

        }
    };

    void apply_control_check() {

    }
}