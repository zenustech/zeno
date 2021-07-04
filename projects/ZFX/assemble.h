#pragma once

#include "program.h"
#include <string>

Instruction assemble_instruction(std::string const &line);
Program assemble_program(std::string const &lines);
