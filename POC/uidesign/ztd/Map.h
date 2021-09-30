#pragma once

#include <map>
#include "Exception.h"
#include "format.h"

namespace ztd {

#if 1
template <class K, class V>
using Map = std::map<K, V>;
#else
template <class K, class V>
struct Map : std::map<K, V> {
    using std::map<K, V>::map;

    V &at(K const &k) {
        auto it = this->find(k);
        if (it == this->end()) {
            throw Exception(toString("KeyError: ", k));
        }
        return it->second;
    }

    V const &at(K const &k) const {
        auto it = this->find(k);
        if (it == this->end()) {
            throw Exception(toString("KeyError: ", k));
        }
        return it->second;
    }
};
#endif

}
