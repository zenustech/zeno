//
// Created by admin on 2022/5/14.
//
#pragma once

#include "../Ast.h"
#include "Module.h"
#include <tuple>
#include <unordered_map>
/*TypeFinder - Walk various Statements, identifying various information including
 *
 * */

namespace zfx {
#define ERROR_IF(x) do { \
    if (x)  {         \
        error("%s", #x);  \
        }                 \
    } while(0)

    struct TypeFinder {
        using visit_emit_type = std::tuple<>;

        std::unordered_map<Statement, bool> VisitStatement ;
        /*
         * This is helper map to avoid repeated visit statement
         * */
        void visit(Statement* stmt) {
            if (VisitStatement.find(stmt) != VisitStatement.end()) {
                std::cout << "this is " << std::endl;
            }
            VisitStatement.insert(stmt, true);
        }

        //
        void visit(ParamSymbolStmt* stmt) {
            stmt->dim = stmt->;//设置维数
            visit((Statement*)stmt);
        }

        void visit(SymbolStmt* stmt) {
            stmt->dim = stmt->;
            visit((Statement*)stmt);
        }

        void visit(LiteralStmt* stmt) {
            stmt->dim = 1;
            visit((Statement*)stmt);
        }

        void visit(FunctionCallStmt* stmt) {
            stmt->dim = 0;//初始化维度为0
            auto const& name = stmt->name;

            //如果是vec那么就是
            if (name.substr(0, 3) == "vec" && name.size() == 4 && isdigit(name[3])) {
                int dim = name[3] - '0';//字符串转换为数字
                stmt->dim = dim;
            } else {
                if ()
                    //开始判断参数

            }


        }

    };

    void apply_type_check(Module* m) {
        TypeFinder visitor;
        visitor.apply(m);
    }
}

