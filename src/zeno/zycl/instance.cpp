#include <zeno/zycl/instance.h>
#include <zeno/zmt/log.h>

ZENO_NAMESPACE_BEGIN
namespace zycl {
inline namespace ns_instance {

queue default_queue() {
    static queue que;
#ifdef ZENO_WITH_SYCL
    static std::once_flag flg;
    std::call_once(flg, [&] {
        ZENO_ZMT_INFO("SYCL device: {}", que.get_device().get_info<zycl::info::device::name>());
    });
#endif
    return que;
}

}
}
ZENO_NAMESPACE_END
