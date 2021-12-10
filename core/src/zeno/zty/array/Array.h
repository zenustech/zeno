#pragma once


#include <vector>
#include <string>
#include <variant>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct Array {
    using variant_type = std::variant
    < std::vector<float>
    , std::vector<int>
    , std::vector<math::vec3f>
    >;
    
    variant_type m_data;

    Array() = default;
    Array(Array &&) = default;
    Array(Array const &) = default;
    Array &operator=(Array &&) = default;
    Array &operator=(Array const &) = default;
    ~Array() = default;

    template <class T>
        requires (std::is_constructible_v<variant_type, std::vector<T>>)
    Array(std::initializer_list<T> const &t)
        : m_data(std::vector<T>(t))
    {}

    template <class T>
        requires (std::is_constructible_v<variant_type, T>)
    Array(T &&t)
        : m_data(std::forward<T>(t))
    {}

    inline auto &get() & {
        return m_data;
    }

    inline auto const &get() const & {
        return m_data;
    }

    inline auto &&get() && {
        return std::move(m_data);
    }

    template <class T>
    inline std::vector<T> &get() & {
        return std::get<std::vector<T>>(m_data);
    }

    template <class T>
    inline std::vector<T> const &get() const & {
        return std::get<std::vector<T>>(m_data);
    }

    template <class T>
    inline std::vector<T> get() && {
        return std::get<std::vector<T>>(std::move(m_data));
    }

    template <class T>
    inline std::vector<T> &emplace(std::vector<T> init) {
        return m_data.emplace<std::vector<T>>(std::move(init));
    }

    template <class T>
    inline std::vector<T> &emplace() {
        return m_data.emplace<std::vector<T>>();
    }
};

using BoolArray = std::vector<bool>;


Array arrayMathOp(std::string const &type, Array const &arr1);
Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2);
Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2, Array const &arr3);
BoolArray arrayBoolOp(std::string const &type, Array const &arr1, Array const &arr2);
BoolArray arrayBoolOp(std::string const &type, BoolArray const &arr1, BoolArray const &arr2);
BoolArray arrayBoolNotOp(BoolArray const &arr1);
Array arrayFromBoolOp(BoolArray const &arr1);
Array arraySelectOp(BoolArray const &arr1, Array const &arr2, Array const &arr3);
Array arraySelectOp(BoolArray const &arr1, Array const &arr2);


}
ZENO_NAMESPACE_END
