#pragma once

#include <map>
#include <z2/ztd/error.h>

namespace z2::ztd {

template <class K, class V>
struct map : std::map<K, V> {
    using std::map<K, V>::map;

    V &at(K const &k) {
        auto it = this->find(k);
        if (it == this->end()) {
            throw make_error("KeyError: ", k);
        }
        return it->second;
    }

    V const &at(K const &k) const {
        auto it = this->find(k);
        if (it == this->end()) {
            throw make_error("KeyError: ", k);
        }
        return it->second;
    }
};

}
