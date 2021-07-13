#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

void apply_symbol_check(IR *ir);
void apply_type_check(IR *ir);
std::unique_ptr<IR> apply_expand_functions(IR *ir);
std::unique_ptr<IR> apply_lower_math(IR *ir);
std::unique_ptr<IR> apply_lower_access(IR *ir);
void apply_register_allocation(IR *ir);
std::unique_ptr<IR> apply_global_localize(IR *ir);
std::string apply_emit_assembly(IR *ir);

}
