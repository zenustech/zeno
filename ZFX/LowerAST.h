#pragma once

#include "IR.h"
#include "AST.h"
#include <tuple>

std::tuple
    < std::unique_ptr<IR>
    , std::vector<std::string>
    , std::vector<std::string>
    > lower_ast
    ( std::vector<AST::Ptr> asts
    );
