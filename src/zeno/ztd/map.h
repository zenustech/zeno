#pragma once

#include <map>
#include <zeno/ztd/error.h>

ZENO_NAMESPACE_BEGIN
namespace ztd {

template <class K, class V>
struct map : std::map<K, V> {
    using std::map<K, V>::map;

    V &at(K const &k) {
        auto it = this->find(k);
        [[unlikely]] if (it == this->end()) {
            throw format_error("KeyError: {}", k);
        }
        return it->second;
    }

    V const &at(K const &k) const {
        auto it = this->find(k);
        [[unlikely]] if (it == this->end()) {
            throw format_error("KeyError: {}", k);
        }
        return it->second;
    }
};

}
ZENO_NAMESPACE_END
