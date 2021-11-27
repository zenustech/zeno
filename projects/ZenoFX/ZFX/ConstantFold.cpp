#include "IRVisitor.h"
#include "Stmts.h"
#include <optional>
#include <cmath>
#include <map>

namespace zfx {

struct ConstantFold : Visitor<ConstantFold> {
    using visit_stmt_types = std::tuple
        < AsmLoadConstStmt
        , AsmBinaryOpStmt
        , AsmUnaryOpStmt
        , AsmFuncCallStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();
    std::map<int, float> values;

    bool lookup(int id, float &value) {
        if (auto it = values.find(id); it != values.end()) {
            value = it->second;
            return true;
        }
        return false;
    }

    void visit(AsmLoadConstStmt *stmt) {
        values[stmt->dst] = stmt->value;
        ir->push_clone_back(stmt);
    }

    void emplace_const(int dst, float value) {
        values[dst] = value;
        ir->emplace_back<AsmLoadConstStmt>(dst, value);
    }

    void visit(AsmFuncCallStmt *stmt) {
        union {
            uint32_t i;
            float f;
        } ret, args[2];
        size_t nargs = stmt->args.size();
        if (nargs >= 1 && !lookup(stmt->args[0], args[0].f))
            return visit((Statement *)stmt);
        if (nargs >= 2 && !lookup(stmt->args[1], args[1].f))
            return visit((Statement *)stmt);
        if (0) {
#define _PER_OP(x) } else if (nargs == 1 && stmt->name == #x) { ret.f = std::x(args[0].f);
        _PER_OP(sin)
        _PER_OP(cos)
        _PER_OP(tan)
        _PER_OP(asin)
        _PER_OP(acos)
        _PER_OP(atan)
        _PER_OP(exp)
        _PER_OP(log)
        _PER_OP(floor)
        _PER_OP(ceil)
#undef _PER_OP
#define _PER_OP(x) } else if (nargs == 2 && stmt->name == #x) { ret.f = std::x(args[0].f, args[1].f);
        _PER_OP(min)
        _PER_OP(max)
        _PER_OP(atan2)
        _PER_OP(pow)
        _PER_OP(fmod)
#undef _PER_OP
        } else { return visit((Statement *)stmt);
        }
        emplace_const(stmt->dst, ret.f);
    }

    void visit(AsmUnaryOpStmt *stmt) {
        union {
            uint32_t i;
            float f;
        } ret, src;
        if (!lookup(stmt->src, src.f))
            return visit((Statement *)stmt);
        if (0) {
#define _PER_OP(x) } else if (stmt->op == #x) { ret.f = x src.f;
        _PER_OP(+)
        _PER_OP(-)
#undef _PER_OP
        } else if (stmt->op == "!") { ret.i = ~src.i;
        } else { return visit((Statement *)stmt);
        }
        emplace_const(stmt->dst, ret.f);
    }

    void visit(AsmBinaryOpStmt *stmt) {
        union {
            uint32_t i;
            float f;
        } ret, lhs, rhs;
        if (!lookup(stmt->lhs, lhs.f))
            return visit((Statement *)stmt);
        if (!lookup(stmt->rhs, rhs.f))
            return visit((Statement *)stmt);
        if (0) {
#define _PER_OP(x) } else if (stmt->op == #x) { ret.f = lhs.f x rhs.f;
        _PER_OP(+)
        _PER_OP(-)
        _PER_OP(*)
        _PER_OP(/)
#undef _PER_OP
        } else if (stmt->op == "%") { ret.f = std::fmod(lhs.f, rhs.f);
#define _PER_OP(x) } else if (stmt->op == #x) { ret.i = lhs.f x rhs.f ? 0xffffffff : 0;
        _PER_OP(==)
        _PER_OP(!=)
        _PER_OP(>)
        _PER_OP(>=)
        _PER_OP(<)
        _PER_OP(<=)
#undef _PER_OP
#define _PER_OP(x) } else if (stmt->op == #x) { ret.i = lhs.i x rhs.i;
        _PER_OP(&)
        _PER_OP(^)
        _PER_OP(|)
#undef _PER_OP
        } else if (stmt->op == "&!") { ret.i = lhs.i & ~rhs.i;
        } else { return visit((Statement *)stmt);
        }
        emplace_const(stmt->dst, ret.f);
    }

    void visit(Statement *stmt) {
        if (stmt->is_control_stmt()) {
            values.clear();
        }
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_constant_fold(IR *ir) {
    ConstantFold visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
