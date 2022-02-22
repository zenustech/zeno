#include <zeno/extra/GlobalStatus.h>

namespace zeno {

ZENO_API std::string GlobalStatus::toJson() const {
    return "wtf";
}

ZENO_API void GlobalStatus::fromJson(std::string_view json) {
}

}
