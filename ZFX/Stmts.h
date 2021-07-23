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

struct FunctionCallStmt : Stmt<FunctionCallStmt> {
    std::string name;
    std::vector<Statement *> args;

    FunctionCallStmt
        ( int id_
        , std::string name_
        , std::vector<Statement *> const &args_
        )
        : Stmt(id_)
        , name(name_)
        , args(args_)
    {}

    virtual StmtFields fields() override {
        StmtFields fies;
        for (auto &arg: args) {
            fies.emplace_back(arg);
        }
        return fies;
    }

    virtual std::string to_string() const override {
        std::vector<int> argids;
        for (auto const &arg: args) {
            argids.push_back(arg->id);
        }
        return format(
            "FunctionCall [%s] (%s)"
            , name.c_str()
            , format_join(", ", "$%d", argids).c_str()
            );
    }
};

struct VectorSwizzleStmt : Stmt<VectorSwizzleStmt> {
    std::vector<int> swizzles;
    Statement *src;

    VectorSwizzleStmt
        ( int id_
        , std::vector<int> const &swizzles_
        , Statement *src_
        )
        : Stmt(id_)
        , swizzles(swizzles_)
        , src(src_)
    {}

    virtual StmtFields fields() override {
        return {
            src,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "VectorSwizzle [%s] $%d"
            , format_join(", ", "%d", swizzles).c_str()
            , src->id
            );
    }
};

struct VectorComposeStmt : Stmt<VectorComposeStmt> {
    int dimension;
    std::vector<Statement *> args;

    VectorComposeStmt
        ( int id_
        , int dimension_
        , std::vector<Statement *> const &args_
        )
        : Stmt(id_)
        , dimension(dimension_)
        , args(args_)
    {}

    virtual StmtFields fields() override {
        StmtFields fies;
        for (auto &arg: args) {
            fies.emplace_back(arg);
        }
        return fies;
    }

    virtual std::string to_string() const override {
        std::vector<int> argids;
        for (auto const &arg: args) {
            argids.push_back(arg->id);
        }
        return format(
            "VectorCompose %d (%s)"
            , dimension
            , format_join(", ", "$%d", argids).c_str()
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

struct ParamSymbolStmt : Stmt<ParamSymbolStmt> {
    std::vector<int> symids;

    ParamSymbolStmt
        ( int id_
        , std::vector<int> symids_
        )
        : Stmt(id_)
        , symids(symids_)
    {}

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "ParamSymbol [%s]"
            , format_join(", ", "%d", symids).c_str()
            );
    }
};

struct TempSymbolStmt : Stmt<TempSymbolStmt> {
    int tmpid;
    std::vector<int> symids;

    TempSymbolStmt
        ( int id_
        , int tmpid_
        , std::vector<int> symids_
        )
        : Stmt(id_)
        , tmpid(tmpid_)
        , symids(symids_)
    {}

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "TempSymbol %d [%s]"
            , tmpid
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

struct FrontendIfStmt : CtrlStmt<FrontendIfStmt> {
    Statement *cond;

    FrontendIfStmt
        ( int id_
        , Statement *cond_
        )
        : CtrlStmt(id_)
        , cond(cond_)
    {}

    virtual StmtFields fields() override {
        return {
            cond,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "FrontendIf $%d"
            , cond->id
            );
    }
};

struct FrontendElseIfStmt : CtrlStmt<FrontendElseIfStmt> {
    Statement *cond;

    FrontendElseIfStmt
        ( int id_
        , Statement *cond_
        )
        : CtrlStmt(id_)
        , cond(cond_)
    {}

    virtual StmtFields fields() override {
        return {
            cond,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "FrontendElseIf $%d"
            , cond->id
            );
    }
};

struct FrontendElseStmt : CtrlStmt<FrontendElseStmt> {
    FrontendElseStmt
        ( int id_
        )
        : CtrlStmt(id_)
    {}

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "FrontendElse"
            );
    }
};

struct FrontendEndIfStmt : CtrlStmt<FrontendEndIfStmt> {
    FrontendEndIfStmt
        ( int id_
        )
        : CtrlStmt(id_)
    {}

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "FrontendEndIf"
            );
    }
};

/*struct GotoStmt : CtrlStmt<GotoStmt> {
    GotoStmt
        ( int id_
        )
        : CtrlStmt(id_)
    {}

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "-> Goto"
            );
    }
};

struct GotoIfStmt : CtrlStmt<GotoIfStmt> {
    Statement *cond;

    GotoIfStmt
        ( int id_
        , Statement *cond_
        )
        : CtrlStmt(id_)
        , cond(cond_)
    {}

    virtual StmtFields fields() override {
        return {
            cond,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "<- GotoIf $%d"
            , cond->id
            );
    }
};

struct GofromStmt : CtrlStmt<GofromStmt> {
    Statement *from;

    GofromStmt
        ( int id_
        , Statement *from_
        )
        : CtrlStmt(id_)
        , from(from_)
    {}

    virtual StmtFields fields() override {
        return {
            from,
            };
    }

    virtual std::string to_string() const override {
        return format(
            "Gofrom <- $%d"
            , from->id
            );
    }
};*/

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

    virtual int affect_register() const override {
        return src == dst ? -1 : dst;
    }

    virtual std::vector<int> source_registers() const override {
        return {src};
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

    virtual int affect_register() const override {
        return dst;
    }

    virtual std::vector<int> source_registers() const override {
        return {};
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

    virtual int affect_register() const override {
        return dst;
    }

    virtual std::vector<int> source_registers() const override {
        return {lhs, rhs};
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

    virtual int affect_register() const override {
        return dst;
    }

    virtual std::vector<int> source_registers() const override {
        return {src};
    }
};

struct AsmFuncCallStmt : AsmStmt<AsmFuncCallStmt> {
    std::string name;
    int dst;
    std::vector<int> args;

    AsmFuncCallStmt
        ( int id_
        , std::string name_
        , int dst_
        , std::vector<int> const &args_
        )
        : AsmStmt(id_)
        , name(name_)
        , dst(dst_)
        , args(args_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmFuncCall [%s] r%d (%s)"
            , name.c_str()
            , dst
            , format_join(", ", "r%d", args).c_str()
            );
    }

    virtual int affect_register() const override {
        return dst;
    }

    virtual std::vector<int> source_registers() const override {
        return args;
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

    virtual int affect_register() const override {
        return -1;
    }

    virtual std::vector<int> source_registers() const override {
        return {val};
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

    virtual int affect_register() const override {
        return val;
    }

    virtual std::vector<int> source_registers() const override {
        return {};
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

    virtual int affect_register() const override {
        return -1;
    }

    virtual std::vector<int> source_registers() const override {
        return {val};
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

    virtual int affect_register() const override {
        return val;
    }

    virtual std::vector<int> source_registers() const override {
        return {};
    }
};

struct AsmParamLoadStmt : AsmStmt<AsmParamLoadStmt> {
    int mem;
    int val;

    AsmParamLoadStmt
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
            "AsmParamLoad r%d [%d]"
            , val
            , mem
            );
    }

    virtual int affect_register() const override {
        return val;
    }

    virtual std::vector<int> source_registers() const override {
        return {};
    }
};

struct AsmIfStmt : AsmStmt<AsmIfStmt> {
    int cond;

    AsmIfStmt
        ( int id_
        , int cond_
        )
        : AsmStmt(id_)
        , cond(cond_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmIf r%d"
            , cond
            );
    }

    virtual int affect_register() const override {
        return -1;
    }

    virtual std::vector<int> source_registers() const override {
        return {cond};
    }
};

struct AsmElseIfStmt : AsmStmt<AsmElseIfStmt> {
    int cond;

    AsmElseIfStmt
        ( int id_
        , int cond_
        )
        : AsmStmt(id_)
        , cond(cond_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmElseIf r%d"
            , cond
            );
    }

    virtual int affect_register() const override {
        return -1;
    }

    virtual std::vector<int> source_registers() const override {
        return {cond};
    }
};

struct AsmElseStmt : AsmStmt<AsmElseStmt> {
    AsmElseStmt
        ( int id_
        )
        : AsmStmt(id_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmElse"
            );
    }

    virtual int affect_register() const override {
        return -1;
    }

    virtual std::vector<int> source_registers() const override {
        return {};
    }
};

struct AsmEndIfStmt : AsmStmt<AsmEndIfStmt> {
    AsmEndIfStmt
        ( int id_
        )
        : AsmStmt(id_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmEndIf"
            );
    }

    virtual int affect_register() const override {
        return -1;
    }

    virtual std::vector<int> source_registers() const override {
        return {};
    }
};

/*struct AsmGotoIfStmt : AsmStmt<AsmGotoIfStmt> {
    int cond;

    AsmGotoIfStmt
        ( int id_
        , int cond_
        )
        : AsmStmt(id_)
        , cond(cond_)
    {}

    virtual std::string to_string() const override {
        return format(
            "<- AsmGotoIf r%d"
            , cond
            );
    }

    virtual int affect_register() const override {
        return -1;
    }
};

struct AsmJumpStmt : AsmStmt<AsmJumpStmt> {
    int target;

    AsmJumpStmt
        ( int id_
        , int target_
        )
        : AsmStmt(id_)
        , target(target_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmJump -> $%d"
            , target
            );
    }

    virtual int affect_register() const override {
        return -1;
    }
};

struct AsmJumpIfStmt : AsmStmt<AsmJumpIfStmt> {
    int cond;
    int target;

    AsmGotoIfStmt
        ( int id_
        , int cond_
        , int target_
        )
        : AsmStmt(id_)
        , cond(cond_)
        , target(target_)
    {}

    virtual std::string to_string() const override {
        return format(
            "AsmJumpIf r%d -> $%d"
            , cond
            , target
            );
    }

    virtual int affect_register() const override {
        return -1;
    }
};*/

}
