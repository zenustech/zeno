#pragma once

#include "common.h"

namespace zfx {

struct Statement;

using StmtFields = std::vector<std::reference_wrapper<Statement *>>;

struct Statement {
    int id;
    int dim = 0;

    explicit Statement
        ( int id_
        )
        : id(id_)
    {}

    std::string print() const {
        return format("$%d <%d> %s", id, dim, to_string().c_str());
    }

    virtual std::string to_string() const = 0;
    virtual std::unique_ptr<Statement> clone(int newid) const = 0;
    virtual int affect_register() const = 0;
    virtual StmtFields fields() = 0;
};

template <class T>
struct Stmt : Statement {
    using Statement::Statement;

    virtual std::unique_ptr<Statement> clone(int newid) const override {
        auto ret = std::make_unique<T>(static_cast<T const &>(*this));
        ret->id = newid;
        return ret;
    }

    virtual int affect_register() const override {
        return -1;
    }
};

template <class T>
struct AsmStmt : Stmt<T> {
    using Stmt<T>::Stmt;

    virtual StmtFields fields() override {
        return {};
    }
};

struct EmptyStmt : Stmt<EmptyStmt> {
    explicit EmptyStmt
        ( int id_
        )
        : Stmt(id_)
    {}

    virtual StmtFields fields() override {
        return {
            };
    }

    virtual std::string to_string() const override {
        return format(
            "Empty"
            );
    }
};

}
