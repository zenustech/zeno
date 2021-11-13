#pragma once


#include <zeno/zan/map.h>
#include <cstdlib>
#include <string>
#include <set>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_find_unique_name {


static std::string find_unique_name
    ( zan::is_range_of<std::string> auto const &names
    , std::string const &base
    )
{
    std::set<std::string> found;
    for (auto &&name: names) {
        if (name.starts_with(base)) {
            found.insert(name.substr(base.size()));
        }
    }

    if (found.empty())
        return base + '1';

    for (int i = 1; i <= found.size() + 1; i++) {
        std::string is = std::to_string(i);
        if (!found.contains(is)) {
            return base + is;
        }
    }

    return base + '0' + std::to_string(std::rand());
}


}
}
ZENO_NAMESPACE_BEGIN
