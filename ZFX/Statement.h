#pragma once

#include "common.h"

struct Statement;

using StmtFields = std::vector<std::reference_wrapper<Statement *>>;

struct Statement {
    int id;

    explicit Statement
        ( int id_
        )
        : id(id_)
    {}

    virtual std::string print() const = 0;
    virtual std::unique_ptr<Statement> clone(int newid) const = 0;
    virtual StmtFields fields() = 0;
};

template <class T>
struct Stmt : Statement {
    using Statement::Statement;

    virtual std::unique_ptr<Statement> clone(int newid) const {
        auto ret = std::make_unique<T>(static_cast<T const &>(*this));
        ret->id = newid;
        return ret;
    }
};
