#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

void apply_demo_visitor(IR *ir);
void apply_resolve_assign(IR *ir);
std::string apply_emit_assembly(IR *ir);
