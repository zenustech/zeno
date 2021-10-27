#pragma once

#include <zeno/common.h>

#if defined(ZENO_WITH_SYCL)
#include <CL/sycl.hpp>

ZENO_NAMESPACE_BEGIN
inline namespace __zeno_real_sycl {
namespace sycl = cl::sycl;
}
ZENO_NAMESPACE_END
#else
#pragma message("<zeno/sycl/sycl.h> is using fake sycl, add -DZENO_WITH_SYCL flag to use true sycl instead")
#include "__fake_sycl.h"
#endif
