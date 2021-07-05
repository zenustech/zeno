#include "compile.h"
#include <string>
#include <map>

Program compile_program(std::string const &code) {
    return assemble_program(zfx_to_assembly(code));
}
