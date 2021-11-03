#include <zeno/zycl/instance.h>

ZENO_NAMESPACE_BEGIN
namespace zycl {

queue default_queue() {
    static queue que;
#ifndef ZENO_SYCL_IS_EMULATED
    std::once_flag flg;
    std::call_once(flg, [&] {
        std::cout << "SYCL device: " << que.get_device().get_info<zycl::info::device::name>()
            << ", backend: " << que.get_backend() << std::endl;
    });
#endif
    return que;
}

}
ZENO_NAMESPACE_END
