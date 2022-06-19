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
//可以单独设置一个表示arch的同文件
    struct Options {
        //一些基本设置参数加优化选项

        // Options two
        int simd_width;
        int arch_max_regs{16};
        bool const_fold {false};
        bool kill_unreachable_code {false};
        constexpr struct {} for_x64{};

        constexpr struct {} cuda{};
        Options(decltype(for_x64)) {}

        Options(int a) {}
    };
    class CompileError  {

      private:
        std::string message;

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

