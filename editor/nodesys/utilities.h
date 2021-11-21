#pragma once


#include <zeno/common.h>
#include <string>
#include <vector>


ZENO_NAMESPACE_BEGIN

std::string find_unique_name
    ( std::vector<std::string> const &names
    , std::string const &base
    );

ZENO_NAMESPACE_END
