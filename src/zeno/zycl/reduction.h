#pragma once

#include <zeno/zycl/zycl.h>
#include <type_traits>
#include <optional>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zycl {


auto make_reduction(auto &&buf, auto ident, auto &&binop) {
    sycl::reduction::property::initialize_to_identity props;
    return reduction(buf, ident, binop, props);
}

}
ZENO_NAMESPACE_END
