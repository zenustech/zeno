#pragma once

#include <zeno/common.h>
#include <type_traits>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _span_h {

template <class Base, class Range>
struct span {
    Base _M_base;
    Range _M_size;

    constexpr span(Base &&base, Range &&size)
        : _M_base(std::move(base)), _M_size(std::move(size)) {
    }

    constexpr Range size() const {
        return _M_size;
    }

    constexpr decltype(auto) operator[](auto &&t) const {
        return _M_base[std::forward<decltype(t)>(t)];
    }
};


template <class Base, class Range>
span(Base &&base, Range &&size) -> span<std::remove_cvref_t<Base>, std::remove_cvref_t<Range>>;

}
}
ZENO_NAMESPACE_END
