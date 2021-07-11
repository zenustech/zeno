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
    , std::map<std::string, int> symdims
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

    template <class ...Ts>
    Program<Prog> *compile
        ( std::string const &code
        , std::map<std::string, int> const &symdims = {}
        ) {
        std::stringstream ss;
        ss << code << "<EOF>";
        for (auto const &[name, dim]: symdims) {
            ss << ':' << name << '=' << dim;
        }
        auto key = ss.str();

        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second.get();
        }

        auto 
            [ assem
            , symbols
            ] = compile_to_assembly
            (code, std::forward<Ts>(ts)...);
        auto prog = std::make_unique<Program<Prog>>();
        prog->prog = Prog::assemble(assem);
        prog->symbols = symbols;

        auto raw_ptr = prog.get();
        cache[key] = std::move(prog);
        return raw_ptr;
    }
};

}
