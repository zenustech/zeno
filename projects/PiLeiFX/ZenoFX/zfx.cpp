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
    std::cout << "scanner Token" << std::endl;
    //here may output print tokens;
#endif
#ifdef ZFX_PRINT_IR
    std::cout << "Parse Ast" << std::endl;

    //here i want to print ast tree;
#endif
#ifdef ZFX_PRINT_IR
    std::cout << "Transform IR" << std::endl;
    //output IR;
#endif
//There is a problem
    //begin Semantic Analysis
#ifdef ZFX_PRINT_IR
    std::cout << "Controlcheck" << std::endl;
#endif
#ifdef ZFX_PRINT_IR
    std::cout << "SymbolCheck" << std::endl;
#endif
#ifdef ZFX_PRINT_IR
    std::cout << "TypeCheck" << std::endl;
#endif

#ifdef ZFX_PRINT_IR
    std::cout << "Assemble" << std::endl;
#endif
}


}

