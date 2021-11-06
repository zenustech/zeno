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

        _M_had(Os &os) : os(os) {}

        _M_had &operator<<(auto &&t) const {
            os << ' ';
            os << std::forward<decltype(t)>(t);
        }

        ~_M_had() {
            os << std::endl;
        }
    };

    _M_had operator<<(auto &&t) const {
        os << std::forward<decltype(t)>(t);
        return {os};
    }
};

}
ZENO_NAMESPACE_END
