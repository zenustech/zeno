#pragma once


#include <z2/ztd/stdafx.h>


namespace z2::dop {


struct DopContext {
    std::set<std::string> visited;

    void insert(std::string const &name) {
        visited.insert(name);
    }

    bool contains(std::string const &name) const {
        return visited.contains(name);
    }

    void erase(std::string const &name) {
        visited.erase(name);
    }
}


}  // namespace z2::dop
