#pragma once

#include <map>
#include "Exception.h"
#include "format.h"

namespace ztd {

template <class K, class V>
struct Map : std::map<K, V> {
    using std::map<K, V>::map;

    V &at(K const &k) {
        auto it = find(k);
        if (it == end()) {
            throw Exception(toString("KeyError: ", k));
        }
        return it->second;
    }

    V const &at(K const &k) const {
        auto it = find(k);
        if (it == end()) {
            throw Exception(toString("KeyError: ", k));
        }
        return it->second;
    }
};

}
