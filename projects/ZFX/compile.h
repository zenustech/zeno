#pragma once

#include "program.h"
#include <string>

Program assemble_program(std::string const &lines);
std::string compile_program(std::string const &code);
