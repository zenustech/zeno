//
// Created by admin on 2022/5/13.
//

#pragma once
//this is a trial stack machine base Ast tree
#include "Lexical.h
#include "Ast.h"
#include "Symbol.h"
#include <memory>
#include <any>
namespace zfx {
    enum class OpCode {
        //
    };

    class BCModule {
        //This is a bitcode module
        std::shared_ptr<FunctionSymbol> _main;//

        std::vector<std::any> consts;//常量值

        BCModule() {
            //construct function 将内置函数加入到常量池
        }
    };

    classBCModuleDumper {

    };

    class StackFrame {

    };
    class BCGenerator : public AstVisitor {
      public:
        std::shared_ptr<BCModule> m;//编译后生成的模型
        std::shared_ptr<FunctionSymbol> functionSym;

        BCGenerator() {
            this->m = std::make_shared<BCModule>();
        }

         std::any visitVariable(Variable& variable, std::string additional) override {

        }

         std::any visitFunctionCall(FunctionCall& functionCall, std::string additional) override {

         }

         std::any visitBinary(Binary& binary, std::strig additional) {

         }

         std::any visitUnary(Unary& unary, std::string additional) override {

         }

         std::any visitTenary(Tenary& tenary, std::string additional) override {

         }

         std::any visitAssign(AssignStmt& assign, std::string additional) override {

         }

         std::any visitLiteral(Literal& literal, std::string additional) override {

         }

         std::any visitIfStmt(ExprIfStmt& exprIfStmt, std::string additional) override {

         }

    };

    class VM {
      public:
        //this

        int32_t execute(const BCModule& bcModule) {
            //找到入口函数
            std::shared_ptr<FunctionSymbol> functionSym;
            if (bcModule._main == nullptr) {
                std::cout << "Can not find main function" << std::endl;
                return -1;
            }
        }
    };
}
