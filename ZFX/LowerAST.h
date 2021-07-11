#pragma once

#include "IR.h"
#include "AST.h"
#include <tuple>
#include <map>

std::tuple
    < std::unique_ptr<IR>
    > lower_ast
    ( std::vector<AST::Ptr> asts
    );
