#pragma once

#include "common.h"

struct IRVisitor {
#define _PER_STMT(StmtType) \
    virtual void visit(StmtType *stmt) { return generic_visit(stmt); }
#include "statements.inl"
#undef _PER_STMT
    virtual void generic_visit(Statement *) {}

    virtual ~IRVisitor() = 0;
};
