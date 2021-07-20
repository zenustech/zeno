#include "Program.h"
#include <string>
#include <memory>
#include <map>

static std::map<std::string, std::unique_ptr<Program>> cache;

Program *compile_program(std::string const &code) {
    auto it = cache.find(code);
    if (it != cache.end()) {
        return it->second.get();
    }
    auto prog = std::make_unique<Program>(
        assemble_program(zfx_to_assembly(code)));
    auto rawptr = prog.get();
    cache[code] = std::move(prog);
    return rawptr;
}
