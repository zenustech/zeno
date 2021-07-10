#pragma once

#include "Statement.h"

struct UnaryOpStmt : Stmt<UnaryOpStmt> {
    std::string op;
    Statement *src;

    UnaryOpStmt
        ( int id_
        , std::string op_
        , Statement *src_
        )
        : Stmt(id_)
        , op(op_)
        , src(src_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = UnaryOp [%s] $%d"
            , id
            , op.c_str()
            , src->id
            );
    }
};

struct BinaryOpStmt : Stmt<BinaryOpStmt> {
    std::string op;
    Statement *lhs;
    Statement *rhs;

    BinaryOpStmt
        ( int id_
        , std::string op_
        , Statement *lhs_
        , Statement *rhs_
        )
        : Stmt(id_)
        , op(op_)
        , lhs(lhs_)
        , rhs(rhs_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = BinaryOp [%s] $%d $%d"
            , id
            , op.c_str()
            , lhs->id
            , rhs->id
            );
    }
};

struct AssignStmt : Stmt<AssignStmt> {
    Statement *dst;
    Statement *src;

    AssignStmt
        ( int id_
        , Statement *dst_
        , Statement *src_
        )
        : Stmt(id_)
        , dst(dst_)
        , src(src_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = Assign $%d $%d"
            , id
            , dst->id
            , src->id
            );
    }
};

struct SymbolStmt : Stmt<SymbolStmt> {
    std::string name;

    SymbolStmt
        ( int id_
        , std::string name_
        )
        : Stmt(id_)
        , name(name_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = Symbol [%s]"
            , id
            , name.c_str()
            );
    }
};

struct GlobalStoreStmt : Stmt<GlobalStoreStmt> {
    SymbolStmt *mem;
    Statement *val;

    GlobalStoreStmt
        ( int id_
        , SymbolStmt *mem_
        , Statement *val_
        )
        : Stmt(id_)
        , mem(mem_)
        , val(val_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = GlobalStore $%d $%d"
            , id
            , mem->id
            , val->id
            );
    }
};

struct GlobalLoadStmt : Stmt<GlobalLoadStmt> {
    SymbolStmt *mem;

    GlobalLoadStmt
        ( int id_
        , SymbolStmt *mem_
        )
        : Stmt(id_)
        , mem(mem_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = GlobalLoad $%d"
            , id
            , mem->id
            );
    }
};

struct LiterialStmt : Stmt<LiterialStmt> {
    std::string value;

    LiterialStmt
        ( int id_
        , std::string value_
        )
        : Stmt(id_)
        , value(value_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = Literial [%s]"
            , id
            , value.c_str()
            );
    }
};
