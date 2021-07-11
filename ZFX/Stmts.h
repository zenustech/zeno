#pragma once

#include "Statement.h"

namespace zfx {

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

    virtual StmtFields fields() override {
        return {
            src,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "UnaryOp [%s] $%d"
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

    virtual StmtFields fields() override {
        return {
            lhs,
            rhs,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "BinaryOp [%s] $%d $%d"
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

    virtual StmtFields fields() override {
        return {
            dst,
            src,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "Assign $%d $%d"
            , dst->id
            , src->id
            );
    }
};

struct SymbolStmt : Stmt<SymbolStmt> {
    std::vector<int> symids;

    SymbolStmt
        ( int id_
        , std::vector<int> symids_
        )
        : Stmt(id_)
        , symids(symids_)
    {}

    bool is_temporary() const {
        return symids.size() == 0 && dim != 0;
    }

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "Symbol [%s]"
            , format_join(", ", "%d", symids).c_str()
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

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "Literial [%s]"
            , value.c_str()
            );
    }
};

struct AsmAssignStmt : AsmStmt<AsmAssignStmt> {
    int dst;
    int src;

    AsmAssignStmt
        ( int id_
        , int dst_
        , int src_
        )
        : AsmStmt(id_)
        , dst(dst_)
        , src(src_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmAssign r%d r%d"
            , dst
            , src
            );
    }
};

struct AsmLoadConstStmt : AsmStmt<AsmLoadConstStmt> {
    int dst;
    std::string value;

    AsmLoadConstStmt
        ( int id_
        , int dst_
        , std::string value_
        )
        : AsmStmt(id_)
        , dst(dst_)
        , value(value_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmLoadConst r%d [%s]"
            , dst
            , value.c_str()
            );
    }
};

struct AsmBinaryOpStmt : AsmStmt<AsmBinaryOpStmt> {
    std::string op;
    int dst;
    int lhs;
    int rhs;

    AsmBinaryOpStmt
        ( int id_
        , std::string op_
        , int dst_
        , int lhs_
        , int rhs_
        )
        : AsmStmt(id_)
        , op(op_)
        , dst(dst_)
        , lhs(lhs_)
        , rhs(rhs_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmBinaryOp (%s) r%d r%d r%d"
            , op.c_str()
            , dst
            , lhs
            , rhs
            );
    }
};

struct AsmUnaryOpStmt : AsmStmt<AsmUnaryOpStmt> {
    std::string op;
    int dst;
    int src;

    AsmUnaryOpStmt
        ( int id_
        , std::string op_
        , int dst_
        , int src_
        )
        : AsmStmt(id_)
        , op(op_)
        , dst(dst_)
        , src(src_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmUnaryOp (%s) r%d r%d"
            , op.c_str()
            , dst
            , src
            );
    }
};

struct AsmLocalStoreStmt : AsmStmt<AsmLocalStoreStmt> {
    int mem;
    int val;

    AsmLocalStoreStmt
        ( int id_
        , int mem_
        , int val_
        )
        : AsmStmt(id_)
        , mem(mem_)
        , val(val_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmLocalStore r%d [%d]"
            , val
            , mem
            );
    }
};

struct AsmLocalLoadStmt : AsmStmt<AsmLocalLoadStmt> {
    int mem;
    int val;

    AsmLocalLoadStmt
        ( int id_
        , int mem_
        , int val_
        )
        : AsmStmt(id_)
        , mem(mem_)
        , val(val_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmLocalLoad r%d [%d]"
            , val
            , mem
            );
    }
};

struct AsmGlobalStoreStmt : AsmStmt<AsmGlobalStoreStmt> {
    int mem;
    int val;

    AsmGlobalStoreStmt
        ( int id_
        , int mem_
        , int val_
        )
        : AsmStmt(id_)
        , mem(mem_)
        , val(val_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmGlobalStore r%d [%d]"
            , val
            , mem
            );
    }
};

struct AsmGlobalLoadStmt : AsmStmt<AsmGlobalLoadStmt> {
    int mem;
    int val;

    AsmGlobalLoadStmt
        ( int id_
        , int mem_
        , int val_
        )
        : AsmStmt(id_)
        , mem(mem_)
        , val(val_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmGlobalLoad r%d [%d]"
            , val
            , mem
            );
    }
};

}
