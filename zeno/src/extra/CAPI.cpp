#include <zeno/zeno.h>
#include <zeno/extra/CAPI.h>
#include <zeno/utils/memory.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/compile_opts.h>
#include <zeno/utils/log.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/zeno_p.h>
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
        std::map<std::shared_ptr<T>, uint32_t> lut;

    public:
        uint64_t create(std::shared_ptr<T> p) {
            T *raw_p = p.get();
            auto [it, succ] = lut.emplace(std::move(p), 0);
            ++it->second;
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw_p));
        }

        std::shared_ptr<T> const &access(uint64_t key) const {
            T *raw_p = reinterpret_cast<T *>(static_cast<uint64_t>(key));
            auto it = lut.find(make_stale_shared(raw_p));
            if (ZENO_UNLIKELY(it == lut.end()))
                throw makeError<KeyError>(std::to_string(key), cppdemangle(typeid(T)));
            return it->first;
        }

        void destroy(uint64_t key) {
            T *raw_p = reinterpret_cast<T *>(static_cast<uint64_t>(key));
            auto it = lut.find(make_stale_shared(raw_p));
            if (ZENO_UNLIKELY(it == lut.end()))
                throw makeError<KeyError>(std::to_string(key), cppdemangle(typeid(T)));
            if (--it->second <= 0)
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
                log_debug("Zeno API catched error: {}", message);
            } catch (...) {
                errcode = 1;
                message = "(unknown)";
                log_debug("Zeno API catched unknown error");
            }
            return errcode;
        }

        const char *what() noexcept {
            return message.empty() ? "(success)" : message.c_str();
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
    std::shared_ptr<Graph> currentGraph;

    static auto &getObjFactory() {
        static std::map<std::string, Zeno_Object (*)(void *)> impl;
        return impl;
    }

    static auto &getObjDefactory() {
        static std::map<std::string, void *(*)(Zeno_Object)> impl;
        return impl;
    }

    static auto &getCFuncPtrs() {
        static std::map<std::string, void *(*)(void *)> impl;
        return impl;
    }
}

extern "C" {

ZENO_CAPI Zeno_Error Zeno_GetLastError(const char **msgRet_) ZENO_CAPI_NOEXCEPT {
    *msgRet_ = lastError.what();
    return lastError.code();
}

ZENO_CAPI Zeno_Error Zeno_CreateGraph(Zeno_Graph *graphRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto graph = getSession().createGraph();
        *graphRet_ = lutGraph.create(std::move(graph));
    });
}

ZENO_CAPI Zeno_Error Zeno_DestroyGraph(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutGraph.destroy(graph_);
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphIncReference(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutGraph.create(lutGraph.access(graph_));
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphGetSubGraph(Zeno_Graph graph_, Zeno_Graph *retGraph_, const char *subName_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        *retGraph_ = lutGraph.create(lutGraph.access(graph_)->getSubnetGraph(subName_)->shared_from_this());
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

ZENO_CAPI Zeno_Error Zeno_CreateObjectPrimitive(Zeno_Object *objectRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        *objectRet_ = lutObject.create(std::make_shared<PrimitiveObject>());
    });
}

ZENO_CAPI Zeno_Error Zeno_DestroyObject(Zeno_Object object_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutObject.destroy(object_);
    });
}

ZENO_CAPI Zeno_Error Zeno_ObjectIncReference(Zeno_Object object_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        lutObject.create(lutObject.access(object_));
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectLiterialType(Zeno_Object object_, int *typeRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        *typeRet_ = [&] {
            auto optr = lutObject.access(object_).get();
            if (auto strptr = dynamic_cast<StringObject *>(optr)) {
                return 1;
            }
            if (auto numptr = dynamic_cast<NumericObject *>(optr)) {
                if (numptr->is<int>())
                    return 11;
                if (numptr->is<vec2i>())
                    return 12;
                if (numptr->is<vec3i>())
                    return 13;
                if (numptr->is<vec4i>())
                    return 14;
                if (numptr->is<float>())
                    return 21;
                if (numptr->is<vec2f>())
                    return 22;
                if (numptr->is<vec3f>())
                    return 23;
                if (numptr->is<vec4f>())
                    return 24;
            }
            return 0;
        }();
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

ZENO_CAPI Zeno_Error Zeno_GetObjectPrimData(Zeno_Object object_, Zeno_PrimMembType primArrType_, const char *attrName_, void **ptrRet_, size_t *lenRet_, Zeno_PrimDataType *typeRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
        auto memb = invoker_variant(static_cast<size_t>(primArrType_),
            &PrimitiveObject::verts,
            &PrimitiveObject::points,
            &PrimitiveObject::lines,
            &PrimitiveObject::tris,
            &PrimitiveObject::quads,
            &PrimitiveObject::loops,
            &PrimitiveObject::polys,
            &PrimitiveObject::uvs);
        std::string attrName = attrName_;
        std::visit([&] (auto const &memb) {
            auto &attArr = memb(*prim);
            attArr.template attr_visit<AttrAcceptAll>(attrName, [&] (auto &arr) {
                *ptrRet_ = reinterpret_cast<void *>(arr.data());
                *lenRet_ = arr.size();
                using T = std::decay_t<decltype(arr[0])>;
                *typeRet_ = static_cast<Zeno_PrimDataType>(variant_index<AttrAcceptAll, T>::value);
            });
        }, memb);
    });
}

ZENO_CAPI Zeno_Error Zeno_AddObjectPrimAttr(Zeno_Object object_, Zeno_PrimMembType primArrType_, const char *attrName_, Zeno_PrimDataType dataType_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
        auto memb = invoker_variant(static_cast<size_t>(primArrType_),
            &PrimitiveObject::verts,
            &PrimitiveObject::points,
            &PrimitiveObject::lines,
            &PrimitiveObject::tris,
            &PrimitiveObject::quads,
            &PrimitiveObject::loops,
            &PrimitiveObject::polys,
            &PrimitiveObject::uvs);
        std::string attrName = attrName_;
        std::visit([&] (auto const &memb) {
            index_switch<std::variant_size_v<AttrAcceptAll>>(static_cast<size_t>(dataType_), [&] (auto dataType) {
                using T = std::variant_alternative_t<dataType.value, AttrAcceptAll>;
                memb(*prim).template add_attr<T>(attrName);
            });
        }, memb);
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectPrimDataKeys(Zeno_Object object_, Zeno_PrimMembType primArrType_, size_t *lenRet_, const char **keysRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
        auto memb = invoker_variant(static_cast<size_t>(primArrType_),
            &PrimitiveObject::verts,
            &PrimitiveObject::points,
            &PrimitiveObject::lines,
            &PrimitiveObject::tris,
            &PrimitiveObject::quads,
            &PrimitiveObject::loops,
            &PrimitiveObject::polys,
            &PrimitiveObject::uvs);
        std::visit([&] (auto const &memb) {
            auto &attArr = memb(*prim);
            *lenRet_ = attArr.template num_attrs<AttrAcceptAll>() + 1;
            if (keysRet_ != nullptr) {
                size_t index = 0;
                attArr.template forall_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
                    keysRet_[index++] = key.c_str();
                });
            }
        }, memb);
    });
}

ZENO_CAPI Zeno_Error Zeno_ResizeObjectPrimData(Zeno_Object object_, Zeno_PrimMembType primArrType_, size_t newSize_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto optr = lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
        auto memb = invoker_variant(static_cast<size_t>(primArrType_),
            &PrimitiveObject::verts,
            &PrimitiveObject::points,
            &PrimitiveObject::lines,
            &PrimitiveObject::tris,
            &PrimitiveObject::quads,
            &PrimitiveObject::loops,
            &PrimitiveObject::polys,
            &PrimitiveObject::uvs);
        std::visit([&] (auto const &memb) {
            memb(*prim).resize(newSize_);
        }, memb);
    });
}

ZENO_CAPI Zeno_Error Zeno_InvokeObjectFactory(Zeno_Object *objectRet_, const char *typeName_, void *ffiObj_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto it = getObjFactory().find(typeName_);
        if (ZENO_UNLIKELY(it == getObjFactory().end()))
            throw makeError("invalid typeName [" + (std::string)typeName_ + "] in ObjFactory");
        *objectRet_ = it->second(ffiObj_);
    });
}

ZENO_CAPI Zeno_Error Zeno_InvokeObjectDefactory(Zeno_Object object_, const char *typeName_, void **ffiObjRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto it = getObjDefactory().find(typeName_);
        if (ZENO_UNLIKELY(it == getObjDefactory().end()))
            throw makeError("invalid typeName [" + (std::string)typeName_ + "] in ObjDefactory");
        *ffiObjRet_ = it->second(object_);
    });
}

ZENO_CAPI Zeno_Error Zeno_InvokeCFunctionPtr(void *ffiObjArg_, const char *typeName_, void **ffiObjRet_) ZENO_CAPI_NOEXCEPT {
    return lastError.catched([=] {
        auto it = getCFuncPtrs().find(typeName_);
        if (ZENO_UNLIKELY(it == getCFuncPtrs().end()))
            throw makeError("invalid typeName [" + (std::string)typeName_ + "] in CFuncPtrs");
        *ffiObjRet_ = it->second(ffiObjArg_);
    });
}

}

namespace zeno {

ZENO_API Zeno_Object capiLoadObjectSharedPtr(std::shared_ptr<IObject> const &objPtr_) {
    return lutObject.create(objPtr_);
}

ZENO_API void capiEraseObjectSharedPtr(Zeno_Object object_) {
    lutObject.destroy(object_);
}

ZENO_API std::shared_ptr<IObject> capiFindObjectSharedPtr(Zeno_Object object_) {
    return lutObject.access(object_);
}

ZENO_API Zeno_Graph capiLoadGraphSharedPtr(std::shared_ptr<Graph> const &graPtr_) {
    return lutGraph.create(graPtr_);
}

ZENO_API void capiEraseGraphSharedPtr(Zeno_Graph graph_) {
    lutGraph.destroy(graph_);
}

ZENO_API std::shared_ptr<Graph> capiFindGraphSharedPtr(Zeno_Graph graph_) {
    return lutGraph.access(graph_);
}

ZENO_API int capiRegisterObjectFactory(std::string const &typeName_, Zeno_Object (*factory_)(void *)) {
    getObjFactory().emplace(typeName_, factory_);
    return 1;
}

ZENO_API int capiRegisterObjectDefactory(std::string const &typeName_, void *(*defactory_)(Zeno_Object)) {
    getObjDefactory().emplace(typeName_, defactory_);
    return 1;
}

ZENO_API int capiRegisterCFunctionPtr(std::string const &typeName_, void *(*cfunc_)(void *)) {
    getCFuncPtrs().emplace(typeName_, cfunc_);
    return 1;
}

ZENO_API Zeno_Error capiLastErrorCatched(std::function<void()> const &func) noexcept {
    return lastError.catched([&] {
        func();
    });
}

}
