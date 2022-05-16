//
// Created by admin on 2022/5/11.
//

#pragma once
#include <memory>
#include <map>
/*
 * This class is an interface to complete a codegen in the Statement class
 * */
namespace zfx {
    Value* BinaryExprAst::codegen() {
       if (op == '=') {
           VariableExprAst* LHSE = static_cast<>
           if (!LHSE) {
               std::cout << "destination of '=' must be" << std::endl;

           }
           Value* Val = RHS->codegen();
           if (!Val) {
               return nullptr;
           }
       }
    }

    static AllocaInst *createEntryBlock(Function* function, std::string& VarName) {

    }

    Value* NumberExprAst::codegen() {

    }

    Value* UnaryExprAst::codegen() {
        Value* OperandV = Operand->codegen();
        if (!OperandV) {
            return nullptr;
        }
    }
}
