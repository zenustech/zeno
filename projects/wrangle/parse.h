#pragma once

#include "program.h"
#include <string>

Instruction parse_instruction(std::string const &line);
Program parse_program(std::string const &lines);
