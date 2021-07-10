#pragma once

#include "IRVisitor.h"
#include "Stmts.h"

std::unique_ptr<IR> apply_clone(IR *ir);
