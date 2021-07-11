#pragma once

#include "common.h"
#include <algorithm>
#include <tuple>
#include <map>

namespace zfx {

std::tuple
    < std::string
    , std::vector<std::string>
    > compile_to_assembly
    ( std::string const &code
    );

template <class Prog>
struct Program {
    std::unique_ptr<Prog> prog;
    std::vector<std::string> symbols;

    void set_channel_pointer(std::string const &name, float *ptr) {
        auto it = std::find(symbols.begin(), symbols.end(), name);
        int chid = it - symbols.begin();
        prog->set_channel_pointer(chid, ptr);
    }

    void execute() {
        prog->execute();
    }
};

template <class Prog>
struct Compiler {
    std::map<std::string, std::unique_ptr<Program<Prog>>> cache;

    Program<Prog> *compile(std::string const &code) {
        auto it = cache.find(code);
        if (it != cache.end()) {
            return it->second.get();
        }
        auto ptr = nocache_compile(code);
        auto raw_ptr = ptr.get();
        cache[code] = std::move(ptr);
        return raw_ptr;
    }

    auto nocache_compile(std::string const &code) {
        auto 
            [ assem
            , symbols
            ] = compile_to_assembly
            ( code
            );
        auto prog = std::make_unique<Program<Prog>>();
        prog->prog = Prog::assemble(assem);
        prog->symbols = symbols;
        return prog;
    }
};

}
