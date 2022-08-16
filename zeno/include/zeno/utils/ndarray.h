#pragma once

#include <zeno/utils/vec.h>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace zeno {

template <size_t N, class T, class AllocT>
struct ndarray {
    static_assert(N >= 1);

    using value_type = T;
    using size_type = size_t;
    inline static constexpr size_type dimension = N;
    using shape_type = vec<dimension, size_type>;
    using index_type = vec<dimension, size_type>;
    using allocator = AllocT;
    using base_type = std::vector<value_type, allocator>;

    static size_type _shape_product(shape_type const &shape_) noexcept {
        size_type ret = shape_[0];
        for (size_type i = 1; i < dimension; i++) {
            ret *= shape_[i];
        }
        return ret;
    }

    size_type _index_linearize(index_type const &index_) noexcept {
        size_type ret = index_[0];
        for (size_type i = 1; i < dimension; i++) {
            ret *= m_shape[i];
            ret += index_[i];
        }
        return ret;
    }

    size_type _index_linearize_safe(index_type const &index_) const {
        size_type ret = index_[0];
        for (size_type i = 1; i < dimension; i++) {
            ret *= m_shape[i];
            if (index_[i] >= m_shape[i])
                throw std::out_of_range("ndarray::at");
            ret += index_[i];
        }
        return ret;
    }

    shape_type m_shape{};
    base_type m_arr;

    shape_type shape() const noexcept {
        return m_shape;
    }

    ndarray &reshape(shape_type const &shape_) {
        m_shape = shape_;
        m_arr.resize(_shape_product(shape_));
        return *this;
    }

    ndarray() = default;

    explicit ndarray(shape_type const &shape_)
        : m_shape(shape_), m_arr(_shape_product(shape_)) {
    }

    explicit ndarray(shape_type const &shape_, value_type const &value_)
        : m_shape(shape_), m_arr(_shape_product(shape_), value_) {
    }

    value_type &operator[](index_type const &index_) noexcept {
        return m_arr[_index_linearize(index_)];
    }

    value_type &at(index_type const &index_) {
        return m_arr[_index_linearize_safe(index_)];
    }

    value_type const &operator[](index_type const &index_) const noexcept {
        return m_arr[_index_linearize(index_)];
    }

    value_type const &at(index_type const &index_) const {
        return m_arr[_index_linearize_safe(index_)];
    }

    //template <size_t M>
    //struct _accessor {
        //ndarray &m_that;
        //vec<M, size_type> m_indices;

        //static vec<M + 1, size_type> _vec_concat(vec<M, size_type> const &a_, size_type b_) noexcept {
            //vec<M + 1, size_type> ret;
            //for (size_type i = 0; i < M; i++)
                //ret[i] = a_[i];
            //ret[M] = b_;
            //return ret;
        //}

        //std::enable_if_t<M + 1 == N, value_type &> operator[](size_type i_) noexcept {
            //return m_that[_vec_concat(m_indices, i_)];
        //}

        //std::enable_if_t<M + 1 != N, _accessor<M + 1>> operator[](size_type i_) noexcept {
            //return _accessor<M + 1>{m_that, _vec_concat(m_indices, i_)};
        //}
    //};

    //std::enable_if_t<N != 1, _accessor<1>> operator[](size_type i_) noexcept {
        //return _accessor<0>{*this, {i_}};
    //}

    //std::enable_if_t<N == 1, value_type &> operator[](size_type i_) noexcept {
        //return (*this)[i];
    //}
};

}
