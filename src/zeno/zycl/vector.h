#pragma once

#include <zeno/zycl/zycl.h>
#include <zeno/zycl/helper.h>


ZENO_NAMESPACE_BEGIN
namespace zycl {

template <class Buf>
void buffer_from_vector(Buf &buf, auto const &range) {
    auto size = range.size();
    buf = Buf(size);
    auto hacc = make_accessor<access::mode::write>(buf);
    size_t i = 0;
    for (auto &&x: range) {
        hacc[i++] = x;
    }
}

template <class Vector, class Buf>
struct _M_as_vector : Vector {
    Buf &_M_buf;

    _M_as_vector(Buf &buf) : _M_buf(buf) {
        vector_from_buffer(*this, _M_buf);
    }

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
        auto a_buf = make_access(cgh, _M_buf);
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
