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

    auto &as_vector() {
        return static_cast<std::vector<T> &>(*this);
    }

    auto const &as_const_vector() const {
        return static_cast<std::vector<T> const &>(*this);
    }
};

}
ZENO_NAMESPACE_END

#else

ZENO_NAMESPACE_BEGIN
namespace zycl {

void vector_from_buffer(auto &vec, auto const &buf) {
    size_t size = buf.size();
    vec.clear();
    vec.reserve(size);
    auto hacc = make_access<access::mode::read>(buf);
    size_t i = 0;
    for (size_t i = 0; i < size; i++) {
        vec.push_back(hacc[i]);
    }
}

void buffer_from_vector(auto &buf, auto const &vec) {
    size_t size = vec.size();
    buf = std::remove_cvref_t<decltype(buf)>(size);
    auto hacc = make_access<access::mode::discard_write>(buf);
    size_t i = 0;
    for (size_t i = 0; i < size; i++) {
        hacc[i] = vec[i];
    }
}

template <class Vector, class Buf>
struct _M_as_vector : Vector {
    Buf &_M_buf;

    explicit _M_as_vector(Buf &buf) : _M_buf(buf) {
        vector_from_buffer(*this, _M_buf);
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
auto _M_make_as_const_vector(auto &buf) {
    Vector vec;
    vector_from_buffer(vec, buf);
    return vec;
}

template <class T>
struct vector {
    buffer<T, 1> _M_buf;

    template <access::mode mode>
    auto get_access(auto &&cgh) {
        auto a_buf = make_access<mode>(cgh, _M_buf);
        return functor_accessor([=] (id<1> idx) -> decltype(auto) {
            return a_buf[idx];
        });
    }

    auto as_vector() {
        return _M_make_as_vector<std::vector<T>>(_M_buf);
    }

    auto as_const_vector() {
        return _M_make_as_const_vector<std::vector<T>>(_M_buf);
    }
};

}
ZENO_NAMESPACE_END

#endif
