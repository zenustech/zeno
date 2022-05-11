//
// Created by admin on 2022/5/7.
//
/*
 *Prog : statementList? EOF;
 *statementList : statement+;
 * statement : block | expressionStatement | ifStatement | forStatement
 * emptyStatement | functionDecl | variableDecl
 *
 * ifStatement : 'if' '(' expression ') statement ('else' statement)?;
 * forStatement :
 * variableStatement:
 * variableDecl : (Identifier|
 * expression:assignment;
 *Identifier : [a-zA-z][a-zA-Z0-9]*;
 * IntegerLiteral : '0' | [1-9][0-9]*
 * */
#pragma once
#include "Lexical.h"
#include "Ast.h"
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
/*
        auto parse() {
            std::vector<std::string> asts;

            return asts;
        }

*/

        std::vector<std::shared_ptr<AstNode>> parseStatement() {

        }

        std::shared_ptr<AstNode> parseVariable() {

        }

        std::shared_ptr<AstNode> parseAssignment() {

        }

        std::shared_ptr<AstNode> parseIfStatement() {

        };

        std::shared_ptr<AstNode> parseBinary() {

        }

        std::shared_ptr<AstNode> parseUnary() {

        }

        std::shared_ptr<AstNode> parseTenary() {

        }

        std::shared_ptr<AstNode> parseFunctionCall() {
            
        }
    };
/*
    std::vector<std::string> parse(const std::string& code) {

    }
    */
}

