//
// Created by admin on 2022/5/11.
//
/*
 * this is Symbol table
 * save variable name , variable kind and variable dimensional
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
    enum class SymDim{
        zero = 0,
        OneDimensional = 1,
        TwoDimensional = 2,
        ThreeDimensional = 3,
        FourDimensional = 4
    };
    class VarSymbol;
    class FunctionSymbol;
    class Symbol {
      public:
        std::string name;
        SymKind kind;
        SymDim dim;
        Symbol(const std::string& name, SymKind kind, SymDim dim) {}

        virtual std::any accept(SymbolVisitor& visitor, std::string additional) = 0;
    };

    class SymbolVisitor {
      public:
        virtual std::any visitVarSymbol(VarSymbol& sym, std::string additional);
        virtual std::any visitFunctionSymbol(FunctionSymbol& sym, std::string additional);

    };

    class VarSymbol : public Symbol {
      public:
        VarSymbol(const std::string& name, SymKind kind, SymDim dim) : Symbol(name, kind, dim){

        }

        std::any accept(SymbolVisitor& visitor, std::string additional) {
            return visitor.visitVarSymbol(*this, additional);
        }
    };

    class FunctionSymbol : public Symbol {
      public:
        FunctionSymbol(const std::string name, SymKind kind, SymDim dim) : Symbol(name, kind, kind) {

        }

        std::any accept(SymbolVisitor& visitor, std::string additional) {
            return visitor.visitFunctionSymbol(*this, additional);
        }

    };
     class SymbolDumper : public SymbolVisitor{

     };
    //extern std::map<std::string, std::shared_ptr<FunctionSymbol>> built_ins;

}
