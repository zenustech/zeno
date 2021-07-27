#include "IRVisitor.h"
#include "Stmts.h"
#include <sstream>
#include "StmHelper.h"
#include <cmath>

namespace zfx {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct DemoteMathFuncs : Visitor<DemoteMathFuncs> {
    using visit_stmt_types = std::tuple
        < UnaryOpStmt
        , BinaryOpStmt
        , FunctionCallStmt
        , Statement
        >;

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    /*Stm emit_stm(std::string const &name, std::vector<Stm> const &args) {
        ERROR_IF(args.size() == 0);
        std::vector<Statement *> argptrs;
        for (auto const &arg: args) {
            argptrs.push_back(arg);
        }
        return {args[0].ir, emit_op(name, argptrs)};
    }*/

    Stm make_stm(Statement *stmt) {
        return {ir.get(), ir->push_clone_back(stmt)};
    }

    Stm stm_const(float x) {
        return {ir.get(), ir->emplace_back<LiterialStmt>(x)};
    }

    Stm stm_const(uint32_t x) {
        union {
            float f;
            uint32_t i;
        } u;
        u.i = x;
        return stm_const(u.f);
    }

    Statement *emit_op(std::string const &name, std::vector<Statement *> const &args) {
        if (0) {

        } else if (name == "!") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            auto mask = stm_const(0xffffffff);
            return stm("^", x, mask);

        } else if (name == "-" && args.size() == 1) {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            auto mask = stm_const(0x80000000);
            return stm("^", x, mask);

        } else if (name == "abs") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            auto mask = stm_const(0x80000000);
            return stm("&!", x, mask);

        /*} else if (name == "fsin") {
            ERROR_IF(args.size() != 1);
            auto x = make_stm(args[0]);
            auto z = x;
            auto z2 = z * z;
            auto r = stm_const(1.f);
            auto t = z2 * stm_const(1.f / 6);
            r = r - t;
            t = z2 * stm_const(1.f / 20) * t;
            r = r + t;
            t = z2 * stm_const(1.f / 48) * t;
            r = r - t;
            t = z2 * stm_const(1.f / 72) * t;
            r = r + t;
            r = r * z;
            return r;*/

        /* todo: also add fast exp
//    http://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
//    快速的指数运算，精度一般
inline __m128 _mm_fexp_ps(__m128 x)
{
    __m128i T = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(1512775)), _mm_set1_ps(1072632447)));
    __m128i TL = _mm_unpacklo_epi32(_mm_setzero_si128(), T);
    __m128i TH = _mm_unpackhi_epi32(_mm_setzero_si128(), T);
    return _mm_movelh_ps(_mm_cvtpd_ps(_mm_castsi128_pd(TL)), _mm_cvtpd_ps(_mm_castsi128_pd(TH)));
}*/

        } else {
            return nullptr;
        }
    }

    void visit(UnaryOpStmt *stmt) {
        auto new_stmt = emit_op(stmt->op, {stmt->src});
        if (!new_stmt) {
            return visit((Statement *)stmt);
        }
        ir->mark_replacement(stmt, new_stmt);
    }

    void visit(BinaryOpStmt *stmt) {
        auto new_stmt = emit_op(stmt->op, {stmt->lhs, stmt->rhs});
        if (!new_stmt) {
            return visit((Statement *)stmt);
        }
        ir->mark_replacement(stmt, new_stmt);
    }

    void visit(FunctionCallStmt *stmt) {
        auto new_stmt = emit_op(stmt->name, stmt->args);
        if (!new_stmt) {
            return visit((Statement *)stmt);
        }
        ir->mark_replacement(stmt, new_stmt);
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::unique_ptr<IR> apply_demote_math_funcs(IR *ir) {
    DemoteMathFuncs visitor;
    visitor.apply(ir);
    return std::move(visitor.ir);
}

}
