#pragma once

#include "program.h"
#include <string>

Program assemble_program(std::string const &lines);
std::string zfx_to_assembly(std::string const &code);
Program compile_program(std::string const &code);
