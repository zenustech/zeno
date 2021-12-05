#pragma once

#include <vector>
#include <zeno/ztd/error.h>

ZENO_NAMESPACE_BEGIN
namespace ztd {

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

    T &touch(size_t i) {
        auto n = this->size();
        [[unlikely]] if (i >= n) {
            this->resize(i + 1);
        }
        return this->operator[](i);
    }

    T touch(size_t i) const {
        auto n = this->size();
        [[unlikely]] if (i >= n) {
            return T{};
        }
        return this->operator[](i);
    }
};

}
ZENO_NAMESPACE_END
