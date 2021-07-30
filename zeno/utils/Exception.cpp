#include <zeno/utils/Exception.h>

namespace zeno {

ZENO_API Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
}

ZENO_API Exception::~Exception() noexcept = default;

ZENO_API char const *Exception::what() const noexcept {
    return msg.c_str();
}

}
