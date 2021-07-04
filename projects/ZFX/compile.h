#pragma once

#include "program.h"
#include <string>

Instruction compile_instruction(std::string const &line);
Program compile_program(std::string const &lines);
