#pragma once

#include "IR.h"
#include "Stmts.h"

namespace zfx {

struct Stm {
    IR *ir = nullptr;
    Statement *stmt = nullptr;

    Stm() = default;

    Stm(IR *ir_, Statement *stmt_)
        : ir(ir_), stmt(stmt_) {}

    operator Statement *&() {
        return stmt;
    }

    operator Statement * const &() const {
        return stmt;
    }

    Statement *operator->() const {
        return stmt;
    }

    Stm operator[](std::vector<int> const &indices) const {
        auto const &src = *this;
        return {src.ir, src.ir->emplace_back<VectorSwizzleStmt>(indices, src.stmt)};
    }

    Stm operator[](int index) const {
        auto const &src = *this;
        return {src.ir, src.ir->emplace_back<VectorSwizzleStmt>(std::vector<int>{index}, src.stmt)};
    }
};

inline Stm stm(std::string const &op_name, Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>(op_name, lhs.stmt, rhs.stmt)};
}

inline Stm stm(std::string const &op_name, Stm const &src) {
    return {src.ir, src.ir->emplace_back<UnaryOpStmt>(op_name, src.stmt)};
}

inline Stm stm_func(std::string const &op_name, std::vector<Stm> const &args) {
    std::vector<Statement *> argptrs;
    for (auto const &p: args) argptrs.push_back(p);
    return {args[0].ir, args[0].ir->emplace_back<FunctionCallStmt>(op_name, argptrs)};
}

inline Stm operator+(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("+", lhs.stmt, rhs.stmt)};
}

inline Stm operator-(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("-", lhs.stmt, rhs.stmt)};
}

inline Stm operator*(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("*", lhs.stmt, rhs.stmt)};
}

inline Stm operator/(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("/", lhs.stmt, rhs.stmt)};
}

inline Stm operator%(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("%", lhs.stmt, rhs.stmt)};
}

inline Stm operator&(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("&", lhs.stmt, rhs.stmt)};
}

inline Stm stm_andnot(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("&~", lhs.stmt, rhs.stmt)};
}

inline Stm operator^(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("^", lhs.stmt, rhs.stmt)};
}

inline Stm operator|(Stm const &lhs, Stm const &rhs) {
    return {lhs.ir, lhs.ir->emplace_back<BinaryOpStmt>("|", lhs.stmt, rhs.stmt)};
}

inline Stm operator+(Stm const &src) {
    return {src.ir, src.ir->emplace_back<UnaryOpStmt>("+", src.stmt)};
}

inline Stm operator-(Stm const &src) {
    return {src.ir, src.ir->emplace_back<UnaryOpStmt>("-", src.stmt)};
}

inline Stm stm_sqrlength(Stm const &src) {
    Stm ret = src[0] * src[0];
    for (int i = 1; i < src->dim; i++) {
        ret = ret + src[i] * src[i];
    }
    return ret;
}

inline Stm stm_dot(Stm const &lhs, Stm const &rhs) {
    Stm ret = lhs[0] * rhs[0];
    for (int i = 1; i < lhs->dim; i++) {
        ret = ret + lhs[i] * rhs[i];
    }
    return ret;
}

inline Stm stm_cross(Stm const &lhs, Stm const &rhs) {
    error("cross product unimplemented for now, sorry");
}

}
