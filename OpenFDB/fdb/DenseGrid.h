#pragma once

#include <array>
#include "vec.h"
#include "schedule.h"

namespace fdb::densegrid {

template <typename T, size_t Log2Res>
struct DenseGrid {
    static constexpr size_t Log2ResX = Log2Res;
    static constexpr size_t Log2ResY = Log2Res;
    static constexpr size_t Log2ResZ = Log2Res;
    using ValueType = T;

private:
    std::array<T, (1ul << 3 * Log2Res)> m_data;

public:
    DenseGrid() = default;
    ~DenseGrid() = default;
    DenseGrid(DenseGrid const &) = delete;
    DenseGrid(DenseGrid &&) = default;
    DenseGrid &operator=(DenseGrid const &) = delete;
    DenseGrid &operator=(DenseGrid &&) = default;

    decltype(auto) data() const { return m_data.data(); }
    decltype(auto) data() { return m_data.data(); }
    decltype(auto) size() { return m_data.size(); }

protected:
    T *address(vec3i ijk) const {
        size_t i = ijk[0] & ((1l << Log2ResX) - 1);
        size_t j = ijk[1] & ((1l << Log2ResY) - 1);
        size_t k = ijk[2] & ((1l << Log2ResZ) - 1);
        auto offset = i | (j << Log2ResX) | (k << Log2ResX + Log2ResY);
        return const_cast<T *>(m_data.data() + offset);
    }

public:
    T const &at(vec3i ijk) const {
        return *this->address(ijk);
    }

    T &at(vec3i ijk) {
        return *this->address(ijk);
    }

    ValueType get(vec3i ijk) const {
        return at(ijk);
    }

    void set(vec3i ijk, ValueType const &val) {
        at(ijk) = val;
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        ndrange_for(pol, vec3i(0),
                1 << vec3i(Log2ResX, Log2ResY, Log2ResZ), [&] (auto ijk) {
            auto &value = at(ijk);
            func(ijk, value);
        });
    }
};

}
