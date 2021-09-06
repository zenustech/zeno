#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

void apply_control_check(IR *ir);
void apply_symbol_check(IR *ir);
void apply_type_check(IR *ir);
std::map<std::string, int> apply_detect_new_symbols(IR *ir,
        std::map<int, std::string> const &temps,
        std::vector<std::pair<std::string, int>> &symbols);
std::unique_ptr<IR> apply_expand_functions(IR *ir);
std::unique_ptr<IR> apply_lower_math(IR *ir);
std::unique_ptr<IR> apply_demote_math_funcs(IR *ir);
std::unique_ptr<IR> apply_lower_access(IR *ir);
std::unique_ptr<IR> apply_constant_fold(IR *ir);
std::map<int, int> apply_reassign_parameters(IR *ir);
std::map<int, float> apply_const_parametrize(IR *ir);
int apply_register_allocation(IR *ir, int nregs);
std::unique_ptr<IR> apply_save_math_registers(IR *ir,
        int nregs, int memsize);
std::unique_ptr<IR> apply_merge_identical(IR *ir);
std::unique_ptr<IR> apply_kill_unreachable(IR *ir);
std::map<int, int> apply_reassign_globals(IR *ir);
void apply_global_localize(IR *ir);
std::string apply_emit_assembly(IR *ir);

}
