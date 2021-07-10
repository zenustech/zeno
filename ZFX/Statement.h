#pragma once

#include "common.h"

struct Statement {
    const int id;

    explicit Statement
        ( int id_
        )
        : id(id_)
    {}

    virtual std::string print() const {
        return format("$%d = Statement");
    }
};

template <class T>
struct Stmt : Statement {
    using Statement::Statement;
};
