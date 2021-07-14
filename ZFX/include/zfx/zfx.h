#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <map>

namespace zfx {

struct Options {
    std::map<std::string, int> symdims;
    std::map<std::string, int> pardims;

    void define_symbol(std::string const &name, int dimension) {
        symdims[name] = dimension;
    }

    void define_param(std::string const &name, int dimension) {
        pardims[name] = dimension;
    }

    void dump(std::ostream &os) const {
        for (auto const &[name, dim]: symdims) {
            os << '/' << name << '/' << dim;
        }
        for (auto const &[name, dim]: pardims) {
            os << '\\' << name << '\\' << dim;
        }
    }
};

std::tuple
    < std::string
    , std::vector<std::pair<std::string, int>>
    , std::vector<std::pair<std::string, int>>
    , std::vector<std::string>
    > compile_to_assembly
    ( std::string const &code
    , Options const &options
    );

template <class Prog>
struct Program {
    std::unique_ptr<Prog> prog;
    std::vector<std::pair<std::string, int>> symbols;
    std::vector<std::pair<std::string, int>> params;

    static inline constexpr size_t SimdWidth = Prog::SimdWidth;

    auto const &get_symbols() const {
        return symbols;
    }

    auto const &get_params() const {
        return symbols;
    }

    int symbol_id(std::string const &name, int dim) const {
        auto it = std::find(
            symbols.begin(), symbols.end(), std::pair{name, dim});
        return it != symbols.end() ? it - symbols.begin() : -1;
    }

    int param_id(std::string const &name, int dim) const {
        auto it = std::find(
            params.begin(), params.end(), std::pair{name, dim});
        return it != params.end() ? it - params.begin() : -1;
    }

    decltype(auto) make_context() {
        return prog->make_context();
    }
};

template <class Prog>
struct Compiler {
    std::map<std::string, std::unique_ptr<Program<Prog>>> cache;

    Program<Prog> *compile
        ( std::string const &code
        , Options const &options
        ) {
        std::ostringstream ss;
        ss << code << "<EOF>";
        options.dump(ss);
        auto key = ss.str();

        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second.get();
        }

        auto 
            [ assem
            , symbols
            , params
            , constants
            ] = compile_to_assembly
            ( code
            , options
            );
        auto prog = std::make_unique<Program<Prog>>();
        prog->prog = Prog::assemble(assem);
        prog->symbols = symbols;  // symbols are attributes in glsl
        prog->params = params;  // params are uniforms in glsl

        auto raw_ptr = prog.get();
        cache[key] = std::move(prog);
        return raw_ptr;
    }
};

}
