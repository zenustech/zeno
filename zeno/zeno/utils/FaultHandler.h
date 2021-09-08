#pragma on

#include <zeno/utils/Exception.h>
#include <functional>

namespace zeno {

ZENO_API void print_traceback(int skip);
ZENO_API void signal_catcher(std::function<void()> const &callback);

}
