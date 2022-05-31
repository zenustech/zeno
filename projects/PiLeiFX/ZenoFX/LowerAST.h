
//
// Created by admin on 2022/5/18.
//
#pragma once
#include "Ast.h"
#include "IR/IR.h"
#include "Stmt.h"
#include <map>
#include <sstream>
/*
 * convent ast to statement
 * */
namespace zfx {
    std::tuple<> lower_ast(std::vector<AstNode> asts,
                       std::map<std::string, int> const& symdims,
                       std::map<std::string, int> const& pardims);
}

