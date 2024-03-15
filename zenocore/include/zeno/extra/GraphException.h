#pragma once

#include <zeno/extra/GlobalError.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <zeno/utils/helper.h>
#include <stdexcept>
#include <string>

namespace zeno {

struct GraphException {
    ObjPath nodePath;
    std::string param;
    std::exception_ptr ep;

    GlobalError evalStatus() const {
        try {
            std::rethrow_exception(ep);
        } catch (ErrorException const &e) {
            log_error("==> error during [{}]: {}", objPathToStr(nodePath), e.what());
            return GlobalError(nodePath, e.getError(), param);
        } catch (std::exception const &e) {
            log_error("==> exception during [{}]: {}", objPathToStr(nodePath), e.what());
            return GlobalError(nodePath, std::make_shared<StdError>(std::current_exception()), param);
        } catch (...) {
            log_error("==> exception during [{}]: <unknown>", objPathToStr(nodePath));
            return GlobalError(nodePath, std::make_shared<StdError>(std::current_exception()), param);
        }
        return GlobalError(ObjPath(), nullptr);
    }

    template <class Func>
    static void translated(Func &&func, INode* const pNode, std::string param = "") {
        try {
            func();
        } catch (GraphException const &ge) {
            throw ge;
        } catch (...) {
            throw GraphException{ pNode->get_path(), param, std::current_exception() };
        }
    }

    template <class Func>
    static void catched(Func &&func, GlobalError &globalStatus) {
        try {
            func();
        } catch (GraphException const &ge) {
            globalStatus = ge.evalStatus();
        }
    }
};

}
