#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

std::unique_ptr<IR> apply_clone(IR *ir);
std::unique_ptr<IR> apply_lower_math(IR *ir);
std::unique_ptr<IR> apply_lower_access(IR *ir);
std::string apply_emit_assembly(IR *ir);
