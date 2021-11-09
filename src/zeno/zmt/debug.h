#pragma once


#include <zeno/common.h>
#include <zeno/zmt/format.h>
#include <iostream>


ZENO_NAMESPACE_BEGIN
namespace zmt {

template <class Os>
struct Debug {
    Os &os;

    Debug(Os &os) : os(os) {
    }

    struct _M_had {
        Os &os;

        inline _M_had(Os &os) : os(os) {}

        inline _M_had &operator<<(auto &&t) const {
            os << ' ';
            os << std::forward<decltype(t)>(t);
        }

        inline ~_M_had() {
            os << std::endl;
        }
    };

    inline _M_had operator<<(auto &&t) const {
        os << std::forward<decltype(t)>(t);
        return {os};
    }
};

inline auto debug() {
    return Debug(std::cout) << "(zmt::debug)";
}

inline auto error() {
    return Debug(std::cerr) << "(zmt::error)";
}

}
ZENO_NAMESPACE_END
