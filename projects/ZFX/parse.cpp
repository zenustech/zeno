#include "program.h"
#include <sstream>
#include <cassert>
#include <string>

static std::vector<std::string> split_str(std::string const &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(s);
  while (std::getline(iss, token, delimiter))
    tokens.push_back(token);
  return tokens;
}

static Operand parse_operand(std::string const &ident) {
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

Instruction parse_instruction(std::string const &line) {
    auto tokens = split_str(line, ' ');
    assert(tokens.size() == 4);
    Instruction inst;
    inst.opcode = magic_enum::enum_cast<Opcode>(tokens[0]).value();
    inst.dst = parse_operand(tokens[1]);
    inst.lhs = parse_operand(tokens[2]);
    inst.rhs = parse_operand(tokens[3]);
    return inst;
}

Program parse_program(std::string const &lines) {
    Program prog;
    for (auto const &line: split_str(lines, '\n')) {
        if (line.size() != 0)
            prog.insts.push_back(parse_instruction(line));
    }
    return prog;
}
