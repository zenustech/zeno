#pragma once

#include <zeno/extra/GlobalStatus.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <stdexcept>
#include <string>

namespace zeno {

struct GraphException {
    std::string nodeName;
    std::exception_ptr ep;

    GlobalStatus evalStatus() const {
        try {
            std::rethrow_exception(ep);
        } catch (ErrorException const &e) {
            log_error("==> error during {}: {}", nodeName, e.what());
            return {nodeName, e.getError()};
        } catch (std::exception const &e) {
            log_error("==> exception during {}: {}", nodeName, e.what());
            return {nodeName, std::make_shared<StdError>(std::current_exception())};
        } catch (...) {
            log_error("==> unknown exception during {}", nodeName);
            return {nodeName, std::make_shared<StdError>(std::current_exception())};
        }
        return {};
    }

    template <class Func>
    static void translated(Func &&func, std::string const &nodeName) {
        try {
            func();
        } catch (GraphException const &ge) {
            throw ge;
        } catch (...) {
            throw GraphException{nodeName, std::current_exception()};
        }
    }

    template <class Func>
    static void catched(Func &&func, GlobalStatus &globalStatus) {
        try {
            func();
        } catch (GraphException const &ge) {
            globalStatus = ge.evalStatus();
        }
    }
};

}
