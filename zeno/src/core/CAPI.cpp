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
#include <cstring>

using namespace zeno;

namespace {

    template <class T>
    class LUT {
        std::set<std::shared_ptr<T>> lut;

    public:
        uint64_t create(std::shared_ptr<T> p) {
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lut.insert(std::move(p)).first->get()));
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
                message = "(unknown)";
            }
            return errcode;
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
    LUT<IObject> lutObject;
    LastError lastError;
    std::map<std::string, std::shared_ptr<IObject>> tempNodeRes;
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

ZENO_CAPI Zeno_Error Zeno_GraphCallTempNode(Zeno_Graph graph_, const char *nodeType_, const char *const *inputKeys_, const Zeno_Object *inputObjects_, size_t inputCount_, size_t *outputCountRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        std::map<std::string, std::shared_ptr<IObject>> inputs;
        for (size_t i = 0; i < inputCount_; i++) {
            inputs.emplace(inputKeys_[i], lutObject.access(inputObjects_[i]));
        }
        tempNodeRes = lutGraph.access(graph_)->callTempNode(nodeType_, inputs);
        *outputCountRet_ = tempNodeRes.size();
    });
}

ZENO_CAPI Zeno_Error Zeno_GetLastTempNodeResult(const char **outputKeys_, Zeno_Object *outputObjects_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto it = tempNodeRes.begin();
        for (size_t i = 0; i < tempNodeRes.size(); i++) {
            outputKeys_[i] = it->first.c_str();
            outputObjects_[i] = lutObject.create(std::move(it->second));
            ++it;
        }
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectInt(Zeno_Object *objectRet_, const int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        if (dim_ == 1)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(value_[0]));
        else if (dim_ == 2)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(vec2i(value_[0], value_[1])));
        else if (dim_ == 3)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(vec3i(value_[0], value_[1], value_[2])));
        else if (dim_ == 4)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(vec4i(value_[0], value_[1], value_[2], value_[3])));
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectFloat(Zeno_Object *objectRet_, const float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        if (dim_ == 1)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(value_[0]));
        else if (dim_ == 2)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(vec2f(value_[0], value_[1])));
        else if (dim_ == 3)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(vec3f(value_[0], value_[1], value_[2])));
        else if (dim_ == 4)
            *objectRet_ = lutObject.create(std::make_shared<NumericObject>(vec4f(value_[0], value_[1], value_[2], value_[3])));
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectString(Zeno_Object *objectRet_, const char *str_, size_t strLen_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        *objectRet_ = lutObject.create(std::make_shared<StringObject>(std::string(str_, strLen_)));
    });
}

ZENO_CAPI Zeno_Error Zeno_DestroyObject(Zeno_Object object_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutObject.destroy(object_);
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectInt(Zeno_Object object_, int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto ptr = dynamic_cast<NumericObject *>(optr);
        if (ZENO_UNLIKELY(ptr == nullptr))
            throw makeError<TypeError>(typeid(NumericObject), typeid(*optr), "get object as numeric");
        if (dim_ == 1) {
            auto const &val = ptr->get<int>();
            value_[0] = val;
        } else if (dim_ == 2) {
            auto const &val = ptr->get<vec2i>();
            value_[0] = val[0];
            value_[1] = val[1];
        } else if (dim_ == 3) {
            auto const &val = ptr->get<vec3i>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
        } else if (dim_ == 4) {
            auto const &val = ptr->get<vec4i>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
            value_[3] = val[3];
        }
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectFloat(Zeno_Object object_, float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto ptr = dynamic_cast<NumericObject *>(optr);
        if (ZENO_UNLIKELY(ptr == nullptr))
            throw makeError<TypeError>(typeid(NumericObject), typeid(*optr), "get object as numeric");
        if (dim_ == 1) {
            auto const &val = ptr->get<float>();
            value_[0] = val;
        } else if (dim_ == 2) {
            auto const &val = ptr->get<vec2f>();
            value_[0] = val[0];
            value_[1] = val[1];
        } else if (dim_ == 3) {
            auto const &val = ptr->get<vec3f>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
        } else if (dim_ == 4) {
            auto const &val = ptr->get<vec4f>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
            value_[3] = val[3];
        }
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectString(Zeno_Object object_, char *strBuf_, size_t *strLenRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto ptr = dynamic_cast<StringObject *>(optr);
        if (ZENO_UNLIKELY(ptr == nullptr))
            throw makeError<TypeError>(typeid(StringObject), typeid(*optr), "get object as string");
        auto &str = ptr->get();
        if (strBuf_ != nullptr)
            memcpy(strBuf_, str.data(), std::min(str.size(), *strLenRet_));
        *strLenRet_ = str.size();
    });
}

#ifdef __cplusplus
}
#endif
