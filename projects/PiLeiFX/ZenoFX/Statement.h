//
// Created by admin on 2022/5/14.
//
#pragma once
#include "Ast.h"
#include <memory>
#include <string>
/*Statement you can think of it as one High IR based ast,
 * and for the time
 * */

namespace zfx {

    strcut Statement;
    using StmtFields = std::vector<std::reference_wrapper<Statement* >>;
    using RegFields = std::vector<int>;

    struct Statement {
        int id;//
        int dim = 0;//dimensionality;
        explicit Statement(int id , int dim) : id(id), dim(dim) {

        }

        std::string print() {
            return ;
        }

        virtual std::string to_string() = 0;

        virtual std::unique_ptr<Statement> clone(int newid) const = 0;

        virtual StmtFields fields() = 0;

        virtual ~Statement() = default;

        virtual std::string serialize_identity() const {
            return to_string();
        }

        virtual bool is_control_stmt() const {
            return false;
        }


    };

    template<class T>
    struct Stmt : Statement {
        using Statement::Statement;

        virtual std::unique_ptr<Statement> clone(int newid) override {
            auto ret = std::make_unique<T>(static_cast<T const&>*this);
            ret->id = newid;
            return ret;
        }
    };

    template<class T>
    struct AsmStmt: Stmt<T> {
        using Stmt<T>::Stmt;

        virtual StmtFields fields() override {

        }
    };

    struct EmptyStmt : Stmt<EmptyStmt> {
        explicit EmptyStmt(int id_) : Stmt(id_) {}

        virtual StmtFields fields() override {
            return {};
        }
    };
}


