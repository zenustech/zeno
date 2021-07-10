#pragma once

#include "common.h"
#include "assembler/Executable.h"

ExecutableInstance assemble_program(std::string const &lines);
