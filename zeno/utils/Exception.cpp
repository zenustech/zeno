#include <zeno/utils/Exception.h>
#include <spdlog/spdlog.h>

namespace zeno {

void print_traceback();
void trigger_gdb();

ZENO_API Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
    spdlog::error("exception occurred: {}", msg);
    print_traceback();
    trigger_gdb();
}

ZENO_API Exception::~Exception() noexcept = default;

ZENO_API char const *Exception::what() const noexcept {
    return msg.c_str();
}

}
