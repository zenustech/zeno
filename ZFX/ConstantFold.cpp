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
        } else {
            return visit((Statement *)stmt);
        }
        values[stmt->dst] = ret.f;
        ir->emplace_back<AsmLoadConstStmt>(stmt->dst, ret.f);
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
