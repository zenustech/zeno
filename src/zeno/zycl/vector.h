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

template <class Vector, class Parent>
struct _M_as_vector : Vector {
    Parent *_M_parent;

    explicit _M_as_vector(Parent *parent) : _M_parent(parent) {}
    _M_as_vector(_M_as_vector const &) = delete;
    _M_as_vector &operator=(_M_as_vector const &) = delete;
    _M_as_vector(_M_as_vector &&) = default;
    _M_as_vector &operator=(_M_as_vector &&) = default;

    ~_M_as_vector() {
        size_t size = Vector::size();
        _M_parent->_M_size = size;
        _M_parent->_M_buf = decltype(_M_parent->_M_buf)(std::max(size, (size_t)1));
        auto hacc = _M_parent->_M_buf.template get_access<access::mode::discard_write>();
        for (size_t i = 0; i < size; i++) {
            hacc[i] = (*this)[i];
        }
    }
};

template <class T>
class _M_vector_init_kernel;

template <class T>
struct vector {
    mutable buffer<T, 1> _M_buf;
    size_t _M_size;

    vector(vector const &) = default;
    vector &operator=(vector const &) = default;
    vector(vector &&) = default;
    vector &operator=(vector &&) = default;

    size_t capacity() const {
        return _M_buf.size();
    }

    size_t _M_recapacity(size_t size) {
        _M_buf = buffer<T, 1>(std::max(size, (size_t)1));
        _M_size = size;
    }

    size_t size() const {
        return _M_size;
    }

    void resize(size_t size) {
        if (size > capacity()) {
            _M_recapacity(size);
        }
        _M_size = size;
    }

    void shrink_to_fit() {
        if (_M_size < capacity())
            _M_recapacity(_M_size);
    }

    size_t clear() {
        _M_size = 0;
    }

    vector() : _M_buf(1), _M_size(0) {
    }

    explicit vector(size_t size) : _M_buf(std::max(size, (size_t)1)), _M_size(size) {
        _M_initialize(0, _M_size);
    }

    void _M_initialize(size_t beg, size_t end) {
        queue().submit([&] (handler &cgh) {
            auto dacc = _M_buf.template get_access<access::mode::discard_write>();
            cgh.parallel_for<_M_vector_init_kernel<T>>(range<1>(_M_size), [=] (item<1> idx) {
                ::new (static_cast<void *>(&dacc[idx])) T{};
            });
        });
    }

    template <access::mode mode>
        requires (mode == access::mode::read)
    auto get_access(auto &&cgh) const {
        static_cast<vector *>(this)->get_access<mode>(cgh);
    }

    template <access::mode mode>
    auto get_access(auto &&cgh) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(cgh)>, host_handler>)
            return _M_buf.template get_access<mode>();
        else
            return _M_buf.template get_access<mode>(cgh);
    }

    template <class Vector>
    void _M_copy_to_vector(Vector &vec) const {
        size_t size = _M_size;
        vec.reserve(size);
        auto hacc = _M_buf.template get_access<access::mode::read>();
        for (size_t i = 0; i < size; i++) {
            vec.push_back(hacc[i]);
        }
    }

    template <class Vector = std::vector<T>>
    auto as_vector() {
        _M_as_vector<Vector, vector> vec(this);
        _M_copy_to_vector(vec);
        return vec;
    }

    template <class Vector = std::vector<T>>
    auto to_vector() const {
        Vector vec;
        _M_copy_to_vector<Vector>(vec);
        return vec;
    }
};

}
ZENO_NAMESPACE_END

#endif
