#pragma once

#include <vector>
#include <zs/ztd/error.h>

namespace zs::ztd::ztd {

template <class T>
struct vector : std::vector<T> {
    using std::vector<T>::vector;

    T &at(size_t i) {
        auto n = this->size();
        [[unlikely]] if (i >= n) {
            throw format_error("IndexError: {} >= {}", i, n);
        }
        return this->operator[](i);
    }

    T const &at(size_t i) const {
        auto n = this->size();
        [[unlikely]] if (i >= n) {
            throw format_error("IndexError: {} >= {}", i, n);
        }
        return this->operator[](i);
    }
};

}
