//
// Created by admin on 2022/5/11.
//
/*
 * this is Symbol table
 * save variable name , variable kind and some other information
 * if there are other needs in the future , we will add
 * */
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <any>

namespace zfx {

    enum class SymKind{Variable, Function};
    class VarSymbol;
    class FunctionSymbol;
    class Symbol {
      public:
        std::string name;

        Symbol(const std::string& name, SymKind kind) {}

        virtual std::any accept(SymbolVisitor& visitor, std::string additional) = 0;
    };

    class SymbolVisitor {
      public:
        virtual std::any visitVarSymbol(VarSymbol& sym, std::string additional);
        virtual std::any visitFunctionSymbol(FunctionSymbol& sym, std::string additional);

    };

    class VarSymbol : public Symbol {
      public:
        VarSymbol(const std::string& name, SymKind kind) : Symbol(name, kind){

        }

        std::any accept(SymbolVisitor& visitor, std::string additional) {
            return visitor.visitVarSymbol(*this, additional);
        }
    };

    class FunctionSymbol : public Symbol {
      public:
        FunctionSymbol(const std::string name, SymKind kind) : Symbol(name, kind) : Symbol(name, kind){

        }

        std::any accept(SymbolVisitor& visitor, std::string additional) {
            return visitor.visitFunctionSymbol(*this, additional);
        }

    };
     class SymbolDumper : public SymbolVisitor{

     };
    //extern std::map<std::string, std::shared_ptr<FunctionSymbol>> built_ins;

}
