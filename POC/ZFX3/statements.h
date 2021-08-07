#pragma once

#include "common.h"
#include "visitor.h"
#include "operators.h"

struct Stmt {
    std::vector<int> inputs;
    std::vector<int> outputs;

    Stmt
        ( std::vector<int> const &inputs_
        , std::vector<int> const &outputs_
        )
        : inputs(inputs_)
        , outputs(outputs_)
    {}

    virtual ~Stmt() = default;

    virtual std::string to_string() const = 0;
    virtual void accept(IRVisitor *visitor) = 0;
};

struct StmtConst : Stmt {
    float value;

    StmtConst
        ( int output_
        , float value_
        )
        : Stmt({output_}, {})
        , value(value_)
    {}

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

    StmtVariable
        ( int output_
        , std::string const &var_name_
        )
        : Stmt({output_}, {})
        , var_name(var_name_)
    {}

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
    Opcode opcode;

    StmtOp
        ( std::vector<int> const &inputs_
        , std::vector<int> const &outputs_
        , Opcode opcode_
        )
        : Stmt(inputs_, outputs_)
        , opcode(opcode_)
    {}

    virtual void accept(IRVisitor *visitor) override {
        return visitor->visit(this);
    }

    virtual std::string to_string() const override {
        return fmt::format("{} = [{}] {}"
                , fmt::join(outputs, ", ")
                , magic_enum::enum_name(opcode)
                , fmt::join(inputs, ", ")
                );
    }
};
