#include "utilities.h"
#include <set>


ZENO_NAMESPACE_BEGIN

std::string find_unique_name
    ( std::vector<std::string> const &names
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

ZENO_NAMESPACE_END
