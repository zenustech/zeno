//
// Created by admin on 2022/5/6.
//

#pragma once

#include "Lexical.h"
#include "Location.h"
#include <iostream>
#include <memory>
#include <string>
#include <any>
#include <memory>
#include <vector>
namespace zfx {

    enum VarType {
        Symbol,
        Parameter
    };

    enum Op {

    };
    class AstVisitor {

    };

    class AstNode {
      public:
        Position beginPos;
        Position endPos;
        virtual void dump() = 0;
        virtual std::any accept(AstVisitor &visitor, std::string additional = "") = 0;
        virtual ~AstNode();
    };
//语句
    class Statement: AstNode {

    };
    //声明，子类为$ @
    class Decl :public AstNode {
      public:
        VarType varType;//变量类型
        std::string name;//变量名称
        expilic Decl(VarType varType) : varType (varType) {

        }
    };

    class VariableDecl : public Decl {
        //变量类型
        //变量初始化的形式，
      public:
        VarType varType;
        //std::shared_ptr<AstNode> init;//变量初始化的语句
        explicit VariableDecl(VarType varType , const std::string& name) : Decl(varType), name(name) {

        }
        std::any accept(AstVisitor& visitor) {

        }

        void dump(std::string prefix) {

        }
    };

    class Statement : public AstNode {

    };

    class Expression:public AstNode {
      public:
        std::any constValue;//本表达式的常量值，用作后面的常量折叠等分析
    };

    class ExpressionStatement : public Statement {
      public:
        std::shared_ptr<AstNode> exp;


    };
    /*
     * 二元表达式
     * */
    class Binary {
      public:
        Op op;
        //左边表达式
        //右边表达式
        Binary() {

        }

        std::any accept(AstVisitor& visitor, const std::string &additional) {

        }
    };

    class Unary {
      public:
        Op op;
        //表达式
        Unary() {

        }

        std::any accept(AstVisitor& visitor, const std::string &additional) {

        }
    };

}

