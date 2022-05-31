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
/*
enum class Ast_Node_Type {
        Ast_Type_Statement,
        Ast_Type_Declaration,
        Ast_Type_Identifier,
        Ast_Type_Integer_Literal,
        Ast_Type_Binary_Op,
        Ast_Type_Unary_Op
    };
    */
/*
    class UnaryOpStmt;
    class BinaryOpStmt;
    class TernaryOpStmt;
    class FunctionCallStmt;
    class AssignStmt;
    class SymbolStmt;
    class SymbolStmt;
    class ParamSymbolStmt;
    class TempSymbolStmt;
    class LiterialStnt;
    class FrontedIfStmt;
    class FrontedElseStmt;
    class FrontedElseIfStmt;
    class FrontedEndIfStmt;

    class AstVisitor {
      public:
        virtual ~visitor() {

        }

        virtual std::any visitVariable(Variable& variable, std::string additional = "");

        virtual std::any visitFunctionCall(FunctionCall& functionCall, std::string additional = "");

        virtual std::any visitBinary(Binary& binary, std::strig additional = "");

        virtual std::any visitUnary(Unary& unary, std::string additional = "");

        virtual std::any visitTenary(Tenary& tenary, std::string additional = "");

        virtual std::any visitAssign(AssignStmt& assign, std::string additional = "");

        virtual std::any visitLiteral(Literal& literal, std::string additional = "");

        virtual std::any visitIfStmt(ExprIfStmt& exprIfStmt, std::string additional = "");

    };

    class AsmVisitor {
      public:
        virtual ~AsmVisitor() {

        }

        virtual std::any visitAsm

    };

    class AstNode {
      public:
        Position beginPos;//
        Position endPos;

        AstNode(const Position& beginPos, const Position& endPos) : beginPos(beginPos), endPos(endPos) {};
        virtual std::any accept(AstVisitor& visitor, std::string additional = "") = 0;
    };

    class Statement : public AstNode {
      public:
        int id;
        int dim = 0;
        Statement(const Position& beginPos, const Position& endPos, int id, int dim)
            : AstNode(beginPos, endPos),id(id), dim(dim) {

        }
    };

    class UnaryOpStmt : public Statement {

    };

    class Ast_Identifier {

    };

    class Expression : public AstNode{
        Expression(const Position& beginPos, const Position& endPos)
    };
    */
/*
 * when we declare a variable , we will create a Ast Node : VariableDecl, if this Ast node has initialization
 * we will create a single node represent initialization, but if only to declare we will only create one node;
 * */
/*
    class VariableDecl : public Expression {
        std::string name;

        //std::shared_ptr<>;

        std::string toString() {
            return this->name;
        }
    };

    class FunctionCall : public Expression {
      public:
        std::string name;//function name
        std::vector<std::shared_ptr<AstNode>> arguments;

        FunctionCall() {

        }

        virtual std::string
    };

    class Binary {
      public:
        Op op;
        std::shared_ptr<AstNode> exp1;// left expression
        std::shared_ptr<AstNode> exp2;// right expression
        Binary() {

        }
        std::string toString() {

        }
    };

    class Unary {
      public:
        Op op;
        bool isPrefix;//whether is prefix operation;
        Unary() {

        }
    };

    class Tenary {
        //Tenary Operation;
      public:
        std::shared_ptr<AstNode> cond;
        std::shared_ptr<AstNode> lhs;
        std::shared_ptr<AstNode> rhs;
    };
    */
/*
 * Notice assignStmt is right associative
 *
 * */
/*
    class AssignStmt {
      public:
        std::shared_ptr<Expression> lhs;
        std::shared_ptr<Expression> value_to_assign;
    };


    class ExprIfStmt {

    };


    class IntegerLiteral : public Expression {
        int32_t value;

    };

    class FloatLiteral : public Expression {

    };

    class AsmStatement {

    };

    class AsmAssignStmt : public AsmStatement{

    };

    class AsmLoadConstStmt : public AsmStatement {

    };

    class AsmTernaryOpStmt : public AsmStatement {

    };

    class AsmBinaryOpStmt : public AsmStatement {

    };

    class AsmUnaryOpStmt : public AsmStatement {

    };

    class AsmFuncCallStmt : public AsmStatement {

    };

    class AsmLocalLoadStmt : public AsmStatement {

    };

    class AsmLocalStoreStmt : public AsmStatement {

    };

    class AsmGlobalLoadStmt : public AsmStatement {

    };

    class AsmGlobalStoreStmt : public AsmStatement {

    };

    class AsmParamLoadStmt : public AsmStatement {

    };

    class AsmIfStmt : public AsmStatement {

    };
    */
/*
    class Ast {
      public:
        using Iter = std::vector<zfx::Token> ::iterator
        Ast() {

        }
      private:
        Token token;
    };

    inline std::unique_ptr<Ast> make_ast() {

    }
    */

    class AstVisitor {

    };

    class AstNode {
      public:
        Position beginPos;
        Position endPos;

        virtual std::any accept(AstNode &visitor, std::string additional = "") = 0;

    };
//语句
    class Statement: AstNode {

    };
    //声明，子类为$ @
    class Decl {

    };

    class
}

