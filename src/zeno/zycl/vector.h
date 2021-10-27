#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <utility>
#include <optional>


ZENO_NAMESPACE_BEGIN
namespace zycl {

struct host_handler {};

template <access::mode mode = access::mode::read_write, class Cgh>
auto make_accessor(Cgh &&cgh, auto &&buf) {
    if constexpr (std::is_same_v<std::remove_cvref_t<Cgh>, host_handler>)
        return buf.template get_accessor<mode>();
    else
        return buf.template get_accessor<mode>(cgh);
}

template <access::mode mode = access::mode::read_write>
auto make_accessor(auto &&buf) {
    return make_accessor(host_handler{}, std::forward<decltype(buf)>(buf));
}

template <class F>
struct functor_accessor {
    F f;

    constexpr functor_accessor(F &&f) : f(std::forward<F>(f)) {
    }

    constexpr decltype(auto) operator[](auto &&t) {
        return f(std::forward<decltype(t)>(t));
    }
};

template <class T>
struct vector {
    buffer<T, 1> _M_buf;
    mutable std::optional<std::vector<T>> _M_host;

    template <access::mode mode>
    auto get_accessor(auto &&cgh) {
        auto a_buf = make_accessor(cgh, _M_buf);
        return functor_accessor([=] (id<1> idx) -> decltype(auto) {
            return a_buf[idx];
        });
    }

    auto &host_begin() const {
        if (!_M_host.has_value())
            _M_host.emplace();
        return _M_host.value();
    }

    void host_end() const {
        if (!_M_host.has_value())
            return;
        buffer_from_range(_M_buf, _M_host->begin(), _M_host->end());
        _M_host = std::nullopt;
    }
};

}
ZENO_NAMESPACE_END
