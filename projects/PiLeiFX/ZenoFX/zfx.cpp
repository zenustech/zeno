//
// Created by admin on 2022/5/8.
//
#include "include/zfx.h"
#include <tuple>
#include "parser.h"
namespace zfx {

std::tuple<> compile_to_assembly (const std::string& code, const Options& options) {
#ifdef ZFX_PRINT_IR
    std::cout << "start zfx" << std::endl;
    std::cout << code << std::endl;
#endif
#ifdef ZFX_PRINT_IR
    std::cout << "Parse Ast" << std::endl;
    auto asts = parse(code);
}


}

