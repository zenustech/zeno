#pragma once

#include <zeno/zycl/zycl.h>
#include <zeno/zycl/helper.h>


#ifdef ZENO_SYCL_IS_EMULATED

ZENO_NAMESPACE_BEGIN
namespace zycl {

template <class T>
struct vector : protected std::vector<T> {
    using std::vector<T>::vector;

    template <access::mode mode>
    auto get_access(auto &&cgh) {
        return functor_accessor([=, this] (id<1> idx) -> decltype(auto) {
            return this->operator[](idx);
        });
    }

    inline auto &as_vector() {
        return static_cast<std::vector<T> &>(*this);
    }

    inline auto const &to_vector() const {
        return static_cast<std::vector<T> const &>(*this);
    }
};

}
ZENO_NAMESPACE_END

#else

ZENO_NAMESPACE_BEGIN
namespace zycl {

void vector_from_buffer(auto &vec, auto &buf, size_t size) {
    vec.clear();
    vec.reserve(size);
    auto hacc = buf.template get_access<access::mode::read>();
    for (size_t i = 0; i < size; i++) {
        vec.push_back(hacc[i]);
    }
}

void buffer_from_vector(auto &buf, auto const &vec) {
    size_t size = vec.size();
    buf = std::remove_cvref_t<decltype(buf)>(std::max(size, 1));
    auto hacc = buf.template get_access<access::mode::discard_write>();
    for (size_t i = 0; i < size; i++) {
        hacc[i] = vec[i];
    }
}

template <class Vector, class Buf>
struct _M_as_vector : Vector {
    Buf &_M_buf;

    explicit _M_as_vector(Buf &buf, size_t size) : _M_buf(buf) {
        vector_from_buffer(*this, _M_buf, size);
    }

    _M_as_vector(_M_as_vector const &) = delete;
    _M_as_vector &operator=(_M_as_vector const &) = delete;
    _M_as_vector(_M_as_vector &&) = default;
    _M_as_vector &operator=(_M_as_vector &&) = default;

    ~_M_as_vector() {
        buffer_from_vector(_M_buf, *this);
    }
};

template <class Vector>
auto _M_make_as_vector(auto &buf) {
    return _M_as_vector<Vector, decltype(buf)>(buf);
}

template <class Vector>
auto _M_make_to_vector(auto &buf) {
    Vector vec;
    vector_from_buffer(vec, buf);
    return vec;
}

template <class T>
struct vector {
    buffer<T, 1> _M_buf;
    size_t _M_size;

    vector(vector const &) = default;
    vector &operator=(vector const &) = default;
    vector(vector &&) = default;
    vector &operator=(vector &&) = default;

    size_t capacity() const {
        return _M_buf.size();
    }

    size_t size() const {
        return _M_size;
    }

    void resize(size_t size) {
        _M_buf = buffer<T, 1>(std::max(size, (size_t)1));
        _M_size = size;
    }

    vector() : _M_buf(1), _M_size(0) {
    }

    explicit vector(size_t size) : _M_buf(std::max(size, (size_t)1)), _M_size(size) {
    }

    template <access::mode mode>
    auto get_access(auto &&cgh) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(cgh)>, host_handler>)
            return _M_buf.template get_access<mode>();
        else
            return _M_buf.template get_access<mode>(cgh);
    }

    template <class Vector = std::vector<T>>
    auto as_vector() {
        return _M_make_as_vector<Vector>(_M_buf);
    }

    template <class Vector = std::vector<T>>
    auto to_vector() {
        return _M_make_to_vector<Vector>(_M_buf);
    }
};

}
ZENO_NAMESPACE_END

#endif
