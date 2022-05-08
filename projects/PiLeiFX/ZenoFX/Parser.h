//
// Created by admin on 2022/5/7.
//
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

        std::shared_ptr<AstNode> parseVariable() {}

        std::shared_ptr<AstNode> parseAssignment() {

        }
    };

    std::vector<std::string> parse(const std::string& code) {

    }
}

