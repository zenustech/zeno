//
// Created by admin on 2022/5/17.
//

#include "Statement.h"

namespace zfx {
    struct UnaryOpStmt : Stmt<UnaryOpStmt> {
        std::string op;
        Statement *src;
        UnaryOpStmt(int id, const std::string op, Statement* src) : Stmt(id), op(op), src(src) {

        }

        virtual StmtFields fields() override {
            return {
                src,
            };//
        }

    };

    struct VectorSwizzleStmt : Stmt<VectorSwizzleStmt> {
//转换坐标

    };

    struct VectorComposeStmt : Stmt<VectorComposeStmt> {
        int dimension;
        std::vector<Statement*> args;

        VectorComposeStmt(int id, int dimension, std::vector<Statement>*)
    };
}
