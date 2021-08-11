#include <zeno/utils/Exception.h>
#include <spdlog/spdlog.h>

namespace zeno {

void print_traceback();
void trigger_gdb();

ZENO_API BaseException::BaseException(std::string_view msg) noexcept
    : msg(msg) {
}

ZENO_API Exception::Exception(std::string_view msg) noexcept
    : BaseException(msg) {
    spdlog::error("exception occurred: {}", msg);
    print_traceback();
    trigger_gdb();
}

ZENO_API BaseException::~BaseException() noexcept = default;

ZENO_API char const *BaseException::what() const noexcept {
    return msg.c_str();
}

}
