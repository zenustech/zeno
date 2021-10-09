#pragma once

#include "IR.h"
#include "AST.h"
#include <tuple>

namespace zfx {

std::tuple
    < std::unique_ptr<IR>
    , std::vector<std::pair<std::string, int>>
    , std::vector<std::pair<std::string, int>>
    , std::map<int, std::string>
    > lower_ast
    ( std::vector<AST::Ptr> asts
    , std::map<std::string, int> const &symdims
    , std::map<std::string, int> const &pardims
    );

}
