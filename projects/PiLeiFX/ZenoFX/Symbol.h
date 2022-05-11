//
// Created by admin on 2022/5/11.
//
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <any>

namespace zfx {

    enum class SymKind{};
    class VarSymbol;
    class FunctionSymbol;
    class Symbol {
        std::string name;
        //

    };

    class SymbolVisitor {
        virtual std::any visitVarSymbol(VarSymbol& sym, std::string additional);
    };

}
