#include <array>

template <size_t Log2Res, typename T>
struct DenseGrid {
    static constexpr size_t Log2ResX = Log2Res;
    static constexpr size_t Log2ResY = Log2Res;
    static constexpr size_t Log2ResZ = Log2Res;
    using ValueType = T;

    std::array<T, (1 << (3 * Log2Res))> m_data;

    DenseGrid() = default;
    ~DenseGrid() = default;
    DenseGrid(DenseGrid const &) = delete;
    DenseGrid(DenseGrid &&) = default;
    DenseGrid &operator=(DenseGrid const &) = delete;
    DenseGrid &operator=(DenseGrid &&) = default;

    T *address(vec3i ijk) const {
        size_t i = ijk[0] & ((1l << Log2ResX) - 1);
        size_t j = ijk[1] & ((1l << Log2ResY) - 1);
        size_t k = ijk[2] & ((1l << Log2ResZ) - 1);
        auto offset = i | (j << Log2ResX) | (k << (Log2ResX + Log2ResY));
        return m_data.data() + offset;
    }

    T &at(vec3i ijk) const {
        return *this->address(ijk);
    }

    auto get(vec3i ijk) const {
        return at(ijk);
    }

    void set(vec3i ijk, ValueType const &val) const {
        at(ijk) = val;
    }
};
