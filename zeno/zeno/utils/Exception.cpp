#include <zeno/utils/Exception.h>
#include <zeno/utils/logger.h>
#if __has_include(<backward.hpp>)
#include <backward.hpp>
#endif

namespace zeno {

ZENO_API BaseException::BaseException(std::string_view msg) noexcept
    : msg(msg) {
}

ZENO_API Exception::Exception(std::string_view msg) noexcept
    : BaseException(msg) {
    log_error("Exception: {}", msg);

#if __has_include(<backward.hpp>)
    using namespace backward;
    StackTrace st;
    st.load_here(32);
    Printer p;
    p.print(st);
#endif
}

ZENO_API BaseException::~BaseException() noexcept = default;

ZENO_API char const *BaseException::what() const noexcept {
    return msg.c_str();
}

}
