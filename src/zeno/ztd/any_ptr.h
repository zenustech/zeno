#pragma once


#include <memory>
#include <optional>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _any_ptr_h {

struct any_ptr : std::shared_ptr<void> {
    using std::shared_ptr<void>::shared_ptr;

    template <class T>
    std::shared_ptr<T> as() {
    }
};

}
}
ZENO_NAMESPACE_END
