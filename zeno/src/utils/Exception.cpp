#include <zeno/utils/Exception.h>
#include <zeno/utils/logger.h>
//#include <zeno/utils/cppdemangle.h>
//#ifdef ZENO_FAULTHANDLER
//#include <backward.hpp>
//#endif

namespace zeno {

ZENO_API BaseException::BaseException(std::string_view msg) noexcept
    : msg(msg) {
}

ZENO_API Exception::Exception(std::string_view msg) noexcept
    : BaseException(msg) {
    //log_error("[{}] {}", cppdemangle(typeid(*this)), msg);

//#ifdef ZENO_FAULTHANDLER
    //using namespace backward;
    //StackTrace st;
    //st.load_here(32);
    //st.skip_n_firsts(3);
    //Printer p;
    //p.print(st);
//#endif
}

ZENO_API BaseException::~BaseException() noexcept = default;

ZENO_API char const *BaseException::what() const noexcept {
    return msg.c_str();
}

}
