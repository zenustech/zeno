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

        AstNode parse_atom() {

        }

        AstNode parse_operator() {

        }

        AstNode parse_compound() {

        }

        AstNode parse_factor() {

        }

        AstNode parse_term() {

        }

        AstNode parse_side() {

        }

        AstNode parse_cond() {

        }

        AstNode parse_andexpr() {

        }

        AstNode parse_orexpr() {

        }

        AstNode parse_expr() {

        }

        AstNode parse_stmt() {

        }
    };


}

