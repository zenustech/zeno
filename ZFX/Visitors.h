#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

std::unique_ptr<IR> apply_clone(IR *ir);
std::unique_ptr<IR> apply_expand_functions(IR *ir);
void apply_type_check(IR *ir);
std::unique_ptr<IR> apply_lower_math(IR *ir);
std::unique_ptr<IR> apply_lower_access(IR *ir);
std::unique_ptr<IR> apply_kill_local_store(IR *ir);
std::string apply_emit_assembly(IR *ir);

}
