#pragma once

#include <memory>
#include "vec.h"
#include "schedule.h"

namespace fdb::densegrid {

template <typename T, int Log2Res, bool IsOffseted = false>
struct DenseGrid {
    static constexpr int Log2ResX = Log2Res;
    static constexpr int Log2ResY = Log2Res;
    static constexpr int Log2ResZ = Log2Res;
    static constexpr bool Offseted = IsOffseted;
    using ValueType = T;

private:
    T m_data[1ul << 3 * Log2Res]{};

public:
    constexpr DenseGrid() = default;
    constexpr DenseGrid(DenseGrid const &) = delete;
    constexpr DenseGrid(DenseGrid &&) = default;
    constexpr DenseGrid &operator=(DenseGrid const &) = delete;
    constexpr DenseGrid &operator=(DenseGrid &&) = default;

    inline constexpr ValueType const *data() const { return m_data; }
    inline constexpr ValueType *data() { return m_data; }
    inline constexpr size_t size() { return 1ul << 3 * Log2Res; }

protected:
    constexpr T *linearized_pointer(vec3i ijk) const {
        size_t i = ijk[0] & ((1l << Log2ResX) - 1);
        size_t j = ijk[1] & ((1l << Log2ResY) - 1);
        size_t k = ijk[2] & ((1l << Log2ResZ) - 1);
        auto offset = i | (j << Log2ResX) | (k << Log2ResX + Log2ResY);
        return const_cast<T *>(data() + offset);
    }

public:
    constexpr T const &at(vec3i ijk) const {
        return *linearized_pointer(ijk);
    }

    constexpr T &at(vec3i ijk) {
        return *linearized_pointer(ijk);
    }

    constexpr ValueType get(vec3i ijk) const {
        return at(ijk);
    }

    constexpr void set(vec3i ijk, ValueType const &val) {
        at(ijk) = val;
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        int beg = 0, end = 1 << Log2Res;
        if constexpr (IsOffseted) {
            end >>= 1;
            beg = -end;
        }
        ndrange_for(pol, vec3i(beg), vec3i(end), [&] (auto ijk) {
            auto &value = at(ijk);
            func(ijk, value);
        });
    }
};

}
