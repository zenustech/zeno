#include "Program.h"
#include "split_str.h"
#include <magic_enum.hpp>
#include <sstream>
#include <cassert>
#include <string>

struct Assembler {
    Operand assemble_operand(std::string const &ident) {
        Operand operand;
        switch (ident[0]) {
        case '#': operand.type = OperandType::imm; break;
        case '$': operand.type = OperandType::reg; break;
        case '@': operand.type = OperandType::mem; break;
        }
        std::stringstream ss(ident.substr(1));
        if (operand.type == OperandType::imm)
            ss >> operand.value;
        else
            ss >> operand.index;
        return operand;
    }

    void bind_channel(int memid, std::string const &attr) {
        if (prog.channels.size() < memid + 1)
            prog.channels.resize(memid + 1);
        prog.channels[memid] = attr;
    }

    std::optional<Instruction> assemble_inst(std::string const &line) {
        auto tokens = split_str(line, ' ');
        assert(tokens.size() > 1);
        if (tokens[0] == "bind") {
            assert(tokens.size() > 3);
            int memid = 0;
            std::stringstream(tokens[1]) >> memid;
            bind_channel(memid, tokens[2]);
            return std::nullopt;
        }
        assert(tokens.size() > 2);
        Instruction inst;
        auto opcode = magic_enum::enum_cast<Opcode>(tokens[0]);
        if (!opcode.has_value()) {
            printf("ERROR: invalid opcode: %s\n", tokens[0].c_str());
            return std::nullopt;
        }
        inst.opcode = opcode.value();
        inst.dst = assemble_operand(tokens[1]);
        if (tokens.size() > 2)
            inst.lhs = assemble_operand(tokens[2]);
        if (tokens.size() > 3)
            inst.rhs = assemble_operand(tokens[3]);
        return inst;
    }

    Program prog;

    void assemble(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0)
                continue;
            auto inst = assemble_inst(line);
            if (inst.has_value()) {
                prog.insts.push_back(inst.value());
            }
        }
    }
};

Program assemble_program(std::string const &lines) {
    Assembler a;
    a.assemble(lines);
    return a.prog;
}
