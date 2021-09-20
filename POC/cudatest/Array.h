#pragma once

namespace fdb {

template <class T, size_t N>
struct Array {
    T m[N]{};

    FDB_CONSTEXPR void store(size_t idx, T val) {
        m[idx] = val;
    }

    FDB_CONSTEXPR T load(size_t idx) const {
        return m[idx];
    }
};

}
