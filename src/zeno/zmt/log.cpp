#include <zeno/zmt/log.h>
#include <zeno/zmt/print.h>

ZENO_NAMESPACE_BEGIN
namespace zmt {

static log_level loglev;

void set_log_level(log_level lev) {
    loglev = lev;
}

void output_log(log_level lev, std::string_view msg) {
    if (lev > loglev) {
        zmt::println("({}) {}", "TDIWCEF"[(int)lev], msg);
    }
}

}
ZENO_NAMESPACE_END
