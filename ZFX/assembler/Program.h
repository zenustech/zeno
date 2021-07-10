#pragma once

#include "common.h"
#include "assembler/Executable.h"
#include <map>

struct Program {
    std::unique_ptr<ExecutableInstance> executable;
    std::map<std::string, int> symtable;

    std::vector<char> constmem;
};

std::unique_ptr<Program> assemble_program(std::string const &lines);
