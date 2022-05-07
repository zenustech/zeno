//
// Created by admin on 2022/5/6.
//

#pragma once

#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

namespace zfx {

    struct Options {

        // Options two
        Options() {}

        Options(int a) {}
    };

    struct Program {
        std::vector<std::pair<std::string, int>> symbols;
        std::vector<std::pair<std::string, int>> params;
        std::string assembly;

        auto const& get_assembly() {
            return assembly;
        }

        auto const& get_symbols() {
            return symbols;
        }

        auto const& get_params() {

        }

        void

    };

    struct Compiler {
        std::map<std::string, std::unique_ptr<Program>> cache;

        Program *compile(const std::string &code, const Options &option) {
            std::ostringstream ss;
            ss << code << "<EOF>";

            auto key = ss.str();
            auto it = cache.find(key);
            if (it != cache.end()) {
                return it->second;
            }

        }

    };


}

