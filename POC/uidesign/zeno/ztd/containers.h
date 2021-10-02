#pragma once

#include <map>
#include <vector>
#include "exception.h"
#include "format.h"

namespace ztd {

template <class K, class V>
struct Map : std::map<K, V> {
    using std::map<K, V>::map;

    V &at(K const &k) {
        auto it = this->find(k);
        if (it == this->end()) {
            throw makeException("KeyError: ", k);
        }
        return it->second;
    }

    V const &at(K const &k) const {
        auto it = this->find(k);
        if (it == this->end()) {
            throw makeException("KeyError: ", k);
        }
        return it->second;
    }
};

template <class T>
struct Vector : std::vector<T> {
    using std::vector<T>::vector;

    T &at(size_t i) {
        auto n = this->size();
        if (i >= n) {
            throw makeException("IndexError: ", i, " >= ", n);
        }
        return this->operator[](i);
    }

    T const &at(size_t i) const {
        auto n = this->size();
        if (i >= n) {
            throw makeException("IndexError: ", i, " >= ", n);
        }
        return this->operator[](i);
    }
};

}
