#pragma once

#include "program.h"
#include <string>

Instruction assemble_instruction(std::string const &line);
Program assemble_program(std::string const &lines);

std::string compile_program(
    std::map<std::string, std::string> const &inityping,
    std::string const &code);
