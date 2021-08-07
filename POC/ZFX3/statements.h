#pragma once

#include "common.h"
#include "visitor.h"

struct Stmt {
    std::vector<int> inputs;
    std::vector<int> outputs;

    Stmt
        ( std::vector<int> const &inputs_
        , std::vector<int> const &outputs_
        )
        : inputs(inputs_)
        , outputs(outputs_)
    {
    }

    virtual ~Stmt() = 0;

    virtual std::string to_string() const = 0;
    virtual void accept(IRVisitor *visitor) = 0;
};

struct StmtConst : Stmt {
    float value;

    virtual void accept(IRVisitor *visitor) override {
        return visitor->visit(this);
    }

    virtual std::string to_string() const override {
        return fmt::format("{} = Const {}"
                , outputs.at(0)
                , value
                );
    }
};

struct StmtVariable : Stmt {
    std::string var_name;

    virtual void accept(IRVisitor *visitor) override {
        return visitor->visit(this);
    }

    virtual std::string to_string() const override {
        return fmt::format("{} = Variable {}"
                , outputs.at(0)
                , var_name
                );
    }
};

struct StmtOp : Stmt {
    std::string op_name;

    virtual void accept(IRVisitor *visitor) override {
        return visitor->visit(this);
    }

    virtual std::string to_string() const override {
        return fmt::format("{} = [{}] {}"
                , fmt::join(outputs, ", ")
                , op_name
                , fmt::join(inputs, ", ")
                );
    }
};
