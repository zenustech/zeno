#include <zeno/zycl/instance.h>

ZENO_NAMESPACE_BEGIN
namespace zycl {
inline namespace ns_instance {

queue default_queue() {
    static queue que;
#ifdef ZENO_WITH_SYCL
    static std::once_flag flg;
    std::call_once(flg, [&] {
        std::cout << "SYCL device: " << que.get_device().get_info<zycl::info::device::name>() << std::endl;
    });
#endif
    return que;
}

}
}
ZENO_NAMESPACE_END
