//
// Created by admin on 2022/5/8.
//

#pragma once

#include "IR/Module.h"
#include <any>
#include <iostream>
/*
 * This is a semantic analysis module
 * */
namespace zfx {
    void apply_control_check(Module *m);
    void apply_symbol_check(Module *m);
    void apply_type_check(Module *m);
    std::map<std::string, int> apply_detect_new_symbols(Module *m,
             std::map<int ,std::string>const &temps,
             std::vector<std::pair<std::string, int>& symbols>);
    std::unique_ptr<Module *m> apply_expand_function(Module *m);
    std::unique_ptr<Module *m> apply_lower_math(Module *m);
    std::unique_ptr<Module *m> apply_demote_math_funcs(Module *m);


    /*
     *
     *
     * */

    std::string apply_emit_assembly(Module* m);
}