#include <zeno/utils/Exception.h>

namespace zeno {

void trigger_gdb();

ZENO_API Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
    trigger_gdb();
}

ZENO_API Exception::~Exception() noexcept = default;

ZENO_API char const *Exception::what() const noexcept {
    return msg.c_str();
}

}
