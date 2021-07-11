#pragma once

#include "common.h"

namespace zfx {

struct Statement;

using StmtFields = std::vector<std::reference_wrapper<Statement *>>;

struct Statement {
    int id;
    int dim;

    explicit Statement
        ( int id_
        )
        : id(id_)
    {}

    std::string print() const {
        return format("$%d <%d> : %s", id, dim, to_string().c_str());
    }

    virtual std::string to_string() const = 0;
    virtual std::unique_ptr<Statement> clone(int newid) const = 0;
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
};

template <class T>
struct AsmStmt : Stmt<T> {
    using Stmt<T>::Stmt;

    virtual StmtFields fields() override {
        return {};
    }
};

}
