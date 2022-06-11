//
// Created by admin on 2022/5/7.
//
/*
 *Prog : statementList? EOF;
 *statementList : (variableDecl | functionDecl | expressionStatement);
 * variableDecl : '$'|'@' Identifier
 * statement : block | expressionStatement | ifStatement | forStatement
 * emptyStatement | functionDecl | variableDecl
 *
 * ifStatement : 'if' '(' expression ') statement ('else' statement)?;
 * forStatement :
 * variableStatement:
 *
 * expression:assignment;
 *Identifier : [a-zA-z][a-zA-Z0-9]*;
 * IntegerLiteral : '0' | [1-9][0-9]*
 * */
#pragma once
#include "Lexical.h"
#include "Ast.h"
#include "Location.h"
#include <memory>
#include <vector>
#include <string>

namespace zfx {

    class Parser {
      public:
        Scanner& scanner;
        Parser(Scanner& scanner) : scanner(scanner) {

        }

        //begin Parser and Generate Ast
        std::vector<std::string> Error;
        std::vector<std::string> Warnings;

        void addError(const std::string msg, Position pos) {

        }

        void addWarnings(const std::string msg, Position pos) {

        }

        std::shared_ptr<AstNode> parseVariableDecl() {
            auto t = this->scanner.next();
            //解析$或者@
            if (t.kind = TokenKind::Decl) {

            }
        }

        std::shared_ptr<AstNode> parseAssignment() {

        }

        std::shared_ptr<AstNode> parseBinary(int32_t prec) {

        }

        std::shared_ptr<AstNode> parseUnary() {

            auto t = this->scanner.peak();
            if (t.kind == TokenKind::Op) {
                //前缀的一元表达式
                //跳过运算符
                this->scanner.next();
                auto exp = this->parseUnary();
                //return std::make_shared<Unary>
            }
        }
    };


}

