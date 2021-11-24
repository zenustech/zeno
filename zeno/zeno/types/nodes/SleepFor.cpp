#include <zeno/dop/dop.h>
#include <thread>
#include <chrono>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void SleepFor(dop::FuncContext *ctx) {
    auto ms = value_cast<int>(ctx->inputs.at(0));
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


ZENO_DOP_DEFUN(SleepFor, {}, {{
    "misc", "sleep for a given time (milliseconds)",
}, {
    {"time_ms", "int"},
}, {
}});


}
}
ZENO_NAMESPACE_END
