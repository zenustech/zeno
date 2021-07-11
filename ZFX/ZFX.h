#pragma once

#include "common.h"
#include <tuple>
#include <algorithm>

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
auto compile_to(std::string const &code) {
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
}
