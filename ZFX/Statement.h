#pragma once

#include "common.h"

struct Statement {
    int id;

    explicit Statement
        ( int id_
        )
        : id(id_)
    {}

    virtual std::string print() const {
        return format("$%d = Statement");
    }

    virtual std::unique_ptr<Statement> clone(int newid) const = 0;
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
