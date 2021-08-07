#pragma once

#include "common.h"

struct Stmt;
#define _PER_STMT(StmtType) \
    struct StmtType;
#include "statements.inl"
#undef _PER_STMT

struct IRVisitor {
#define _PER_STMT(StmtType) \
    virtual void visit(StmtType *stmt) { generic_visit((Stmt *)stmt); }
#include "statements.inl"
#undef _PER_STMT
    virtual void generic_visit(Stmt *) {}

    virtual ~IRVisitor() = 0;
};
