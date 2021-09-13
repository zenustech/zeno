#include "sycl_sink.h"


namespace fdb {


CommandQueue &getQueue() {
    static auto p = std::make_unique<CommandQueue>();
    return *p;
}


}
