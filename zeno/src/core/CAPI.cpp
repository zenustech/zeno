#include <zeno/zeno.h>
#include <zeno/core/CAPI.h>
#include <zeno/utils/memory.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/compile_opts.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <set>
#include <stdexcept>
#include <memory>

using namespace zeno;

namespace {

    template <class T>
    class LUT {
        std::set<std::shared_ptr<T>> lut;

    public:
        template <class ...Ts>
        uint64_t create(Ts &&...ts) {
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lut.insert(std::make_shared<T>(std::forward<Ts>(ts)...)).first->get()));
        }

        std::shared_ptr<T> const &access(uint64_t key) const {
            auto it = lut.find(make_stale_shared(reinterpret_cast<T *>(static_cast<uint64_t>(key))));
            if (ZENO_UNLIKELY(it == lut.end()))
                throw makeError<KeyError>(std::to_string(key), cppdemangle(typeid(T)));
            return *it;
        }

        void destroy(uint64_t key) {
            auto it = lut.find(make_stale_shared(reinterpret_cast<T *>(static_cast<uint64_t>(key))));
            if (ZENO_UNLIKELY(it == lut.end()))
                throw makeError<KeyError>(std::to_string(key), cppdemangle(typeid(T)));
            lut.erase(it);
        }
    };

    class LastError {
        uint32_t errcode;
        std::string message;

    public:
        template <class Func>
        uint32_t catched(Func const &func) noexcept {
            errcode = 0;
            message.clear();
            try {
                func();
            } catch (std::exception const &e) {
                errcode = 1;
                message = e.what();
            } catch (...) {
                errcode = 1;
                message = "(got unknown exception type)";
            }
        }

        const char *what() noexcept {
            return message.c_str();
        }

        uint32_t code() noexcept {
            return errcode;
        }
    };

    LUT<Session> lutSession;
    LUT<Graph> lutGraph;
    LastError lastError;
}

#ifdef __cplusplus
extern "C" {
#endif

ZENO_CAPI Zeno_Error Zeno_GetLastErrorCode() ZENO_CAPI_NOEXCEPT {
    return lastError.code();
}

ZENO_CAPI const char *Zeno_GetLastErrorStr() ZENO_CAPI_NOEXCEPT {
    return lastError.what();
}

ZENO_CAPI Zeno_Error Zeno_CreateGraph(Zeno_Graph *graphRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        std::shared_ptr<Graph> graph(getSession().createGraph().release());
        *graphRet_ = lutGraph.create(std::move(graph));
    });
}

ZENO_CAPI Zeno_Error Zeno_DestroyGraph(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutGraph.destroy(graph_);
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphLoadJson(Zeno_Graph graph_, const char *jsonStr_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutGraph.access(graph_)->loadGraph(jsonStr_);
    });
}

#ifdef __cplusplus
}
#endif
