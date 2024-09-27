#include <zeno/zeno.h>
#include <zeno/extra/CAPI.h>

using namespace zeno;

namespace PyZeno{
LUT<Session> lutSession;
LUT<Graph> lutGraph;
LUT<IObject> lutObject;
LastError lastError;
std::map<std::string, std::shared_ptr<IObject>> tempNodeRes;
std::shared_ptr<Graph> currentGraph;
}

extern "C" {
ZENO_CAPI Zeno_Error Zeno_GetLastError(const char **msgRet_) ZENO_CAPI_NOEXCEPT {
    *msgRet_ = PyZeno::lastError.what();
    return PyZeno::lastError.code();
}

ZENO_CAPI Zeno_Error Zeno_CreateGraph(Zeno_Graph *graphRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto graph = getSession().createGraph("");
        *graphRet_ = PyZeno::lutGraph.create(std::move(graph));
    });
}

ZENO_CAPI Zeno_Error Zeno_DestroyGraph(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        PyZeno::lutGraph.destroy(graph_);
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphIncReference(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        PyZeno::lutGraph.create(PyZeno::lutGraph.access(graph_));
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphGetSubGraph(Zeno_Graph graph_, Zeno_Graph *retGraph_, const char *subName_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        *retGraph_ = PyZeno::lutGraph.create(PyZeno::lutGraph.access(graph_)->getSubnetGraph(subName_)->shared_from_this());
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphLoadJson(Zeno_Graph graph_, const char *jsonStr_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        //PyZeno::lutGraph.access(graph_)->loadGraph(jsonStr_);
    });
}

ZENO_CAPI Zeno_Error Zeno_GraphCallTempNode(Zeno_Graph graph_, const char *nodeType_, const char *const *inputKeys_, const Zeno_Object *inputObjects_, size_t inputCount_, size_t *outputCountRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        std::map<std::string, std::shared_ptr<IObject>> inputs;
        for (size_t i = 0; i < inputCount_; i++) {
            inputs.emplace(inputKeys_[i], PyZeno::lutObject.access(inputObjects_[i]));
        }
        PyZeno::tempNodeRes = PyZeno::lutGraph.access(graph_)->callTempNode(nodeType_, inputs);
        *outputCountRet_ = PyZeno::tempNodeRes.size();
    });
}

ZENO_CAPI Zeno_Error Zeno_GetLastTempNodeResult(const char **outputKeys_, Zeno_Object *outputObjects_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto it = PyZeno::tempNodeRes.begin();
        for (size_t i = 0; i < PyZeno::tempNodeRes.size(); i++) {
            outputKeys_[i] = it->first.c_str();
            outputObjects_[i] = PyZeno::lutObject.create(std::move(it->second));
            ++it;
        }
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectInt(Zeno_Object *objectRet_, const int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        if (dim_ == 1)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(value_[0]));
        else if (dim_ == 2)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(zeno::vec2i(value_[0], value_[1])));
        else if (dim_ == 3)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(zeno::vec3i(value_[0], value_[1], value_[2])));
        else if (dim_ == 4)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(zeno::vec4i(value_[0], value_[1], value_[2], value_[3])));
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectFloat(Zeno_Object *objectRet_, const float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        if (dim_ == 1)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(value_[0]));
        else if (dim_ == 2)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(zeno::vec2f(value_[0], value_[1])));
        else if (dim_ == 3)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(zeno::vec3f(value_[0], value_[1], value_[2])));
        else if (dim_ == 4)
            *objectRet_ = PyZeno::lutObject.create(std::make_shared<NumericObject>(zeno::vec4f(value_[0], value_[1], value_[2], value_[3])));
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectString(Zeno_Object *objectRet_, const char *str_, size_t strLen_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        *objectRet_ = PyZeno::lutObject.create(std::make_shared<StringObject>(std::string(str_, strLen_)));
    });
}

ZENO_CAPI Zeno_Error Zeno_CreateObjectPrimitive(Zeno_Object *objectRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        *objectRet_ = PyZeno::lutObject.create(std::make_shared<PrimitiveObject>());
    });
}

ZENO_CAPI Zeno_Error Zeno_DestroyObject(Zeno_Object object_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        PyZeno::lutObject.destroy(object_);
    });
}

ZENO_CAPI Zeno_Error Zeno_ObjectIncReference(Zeno_Object object_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        PyZeno::lutObject.create(PyZeno::lutObject.access(object_));
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectLiterialType(Zeno_Object object_, int *typeRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        *typeRet_ = [&] {
            auto optr = PyZeno::lutObject.access(object_).get();
            if (auto strptr = dynamic_cast<StringObject *>(optr)) {
                return 1;
            }
            if (auto numptr = dynamic_cast<NumericObject *>(optr)) {
                if (numptr->is<int>())
                    return 11;
                if (numptr->is<zeno::vec2i>())
                    return 12;
                if (numptr->is<zeno::vec3i>())
                    return 13;
                if (numptr->is<zeno::vec4i>())
                    return 14;
                if (numptr->is<float>())
                    return 21;
                if (numptr->is<zeno::vec2f>())
                    return 22;
                if (numptr->is<zeno::vec3f>())
                    return 23;
                if (numptr->is<zeno::vec4f>())
                    return 24;
            }
            return 0;
        }();
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectInt(Zeno_Object object_, int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto ptr = dynamic_cast<NumericObject *>(optr);
        if (ZENO_UNLIKELY(ptr == nullptr))
            throw zeno::makeError<TypeError>(typeid(NumericObject), typeid(*optr), "get object as numeric");
        if (dim_ == 1) {
            auto const &val = ptr->get<int>();
            value_[0] = val;
        } else if (dim_ == 2) {
            auto const &val = ptr->get<zeno::vec2i>();
            value_[0] = val[0];
            value_[1] = val[1];
        } else if (dim_ == 3) {
            auto const &val = ptr->get<zeno::vec3i>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
        } else if (dim_ == 4) {
            auto const &val = ptr->get<zeno::vec4i>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
            value_[3] = val[3];
        }
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectFloat(Zeno_Object object_, float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto ptr = dynamic_cast<NumericObject *>(optr);
        if (ZENO_UNLIKELY(ptr == nullptr))
            throw zeno::makeError<TypeError>(typeid(NumericObject), typeid(*optr), "get object as numeric");
        if (dim_ == 1) {
            auto const &val = ptr->get<float>();
            value_[0] = val;
        } else if (dim_ == 2) {
            auto const &val = ptr->get<zeno::vec2f>();
            value_[0] = val[0];
            value_[1] = val[1];
        } else if (dim_ == 3) {
            auto const &val = ptr->get<zeno::vec3f>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
        } else if (dim_ == 4) {
            auto const &val = ptr->get<zeno::vec4f>();
            value_[0] = val[0];
            value_[1] = val[1];
            value_[2] = val[2];
            value_[3] = val[3];
        }
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectString(Zeno_Object object_, char *strBuf_, size_t *strLenRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto ptr = dynamic_cast<StringObject *>(optr);
        if (ZENO_UNLIKELY(ptr == nullptr))
            throw zeno::makeError<TypeError>(typeid(StringObject), typeid(*optr), "get object as string");
        auto &str = ptr->get();
        if (strBuf_ != nullptr)
            memcpy(strBuf_, str.data(), std::min(str.size(), *strLenRet_));
        *strLenRet_ = str.size();
    });
}

ZENO_CAPI Zeno_Error Zeno_GetObjectPrimData(Zeno_Object object_, Zeno_PrimMembType primArrType_, const char *attrName_, void **ptrRet_, size_t *lenRet_, Zeno_PrimDataType *typeRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw zeno::makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
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
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw zeno::makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
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
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw zeno::makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
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
    return PyZeno::lastError.catched([=] {
        auto optr = PyZeno::lutObject.access(object_).get();
        auto prim = dynamic_cast<PrimitiveObject *>(optr);
        if (ZENO_UNLIKELY(prim == nullptr))
            throw zeno::makeError<TypeError>(typeid(PrimitiveObject), typeid(*optr), "get object as primitive");
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
    return PyZeno::lastError.catched([=] {
        auto it = PyZeno::getObjFactory().find(typeName_);
        if (ZENO_UNLIKELY(it == PyZeno::getObjFactory().end()))
            throw zeno::makeError("invalid typeName [" + (std::string)typeName_ + "] in ObjFactory");
        *objectRet_ = it->second(ffiObj_);
    });
}

ZENO_CAPI Zeno_Error Zeno_InvokeObjectDefactory(Zeno_Object object_, const char *typeName_, void **ffiObjRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto it = PyZeno::getObjDefactory().find(typeName_);
        if (ZENO_UNLIKELY(it == PyZeno::getObjDefactory().end()))
            throw zeno::makeError("invalid typeName [" + (std::string)typeName_ + "] in ObjDefactory");
        *ffiObjRet_ = it->second(object_);
    });
}

ZENO_CAPI Zeno_Error Zeno_InvokeCFunctionPtr(void *ffiObjArg_, const char *typeName_, void **ffiObjRet_) ZENO_CAPI_NOEXCEPT {
    return PyZeno::lastError.catched([=] {
        auto it = PyZeno::getCFuncPtrs().find(typeName_);
        if (ZENO_UNLIKELY(it == PyZeno::getCFuncPtrs().end()))
            throw zeno::makeError("invalid typeName [" + (std::string)typeName_ + "] in CFuncPtrs");
        *ffiObjRet_ = it->second(ffiObjArg_);
    });
}

}

namespace zeno {

ZENO_API Zeno_Object capiLoadObjectSharedPtr(std::shared_ptr<IObject> const &objPtr_) {
    return PyZeno::lutObject.create(objPtr_);
}

ZENO_API void capiEraseObjectSharedPtr(Zeno_Object object_) {
    PyZeno::lutObject.destroy(object_);
}

ZENO_API std::shared_ptr<IObject> capiFindObjectSharedPtr(Zeno_Object object_) {
    return PyZeno::lutObject.access(object_);
}

ZENO_API Zeno_Graph capiLoadGraphSharedPtr(std::shared_ptr<Graph> const &graPtr_) {
    return PyZeno::lutGraph.create(graPtr_);
}

ZENO_API void capiEraseGraphSharedPtr(Zeno_Graph graph_) {
    PyZeno::lutGraph.destroy(graph_);
}

ZENO_API std::shared_ptr<Graph> capiFindGraphSharedPtr(Zeno_Graph graph_) {
    return PyZeno::lutGraph.access(graph_);
}

ZENO_API int capiRegisterObjectFactory(std::string const &typeName_, Zeno_Object (*factory_)(void *)) {
    PyZeno::getObjFactory().emplace(typeName_, factory_);
    return 1;
}

ZENO_API int capiRegisterObjectDefactory(std::string const &typeName_, void *(*defactory_)(Zeno_Object)) {
    PyZeno::getObjDefactory().emplace(typeName_, defactory_);
    return 1;
}

ZENO_API int capiRegisterCFunctionPtr(std::string const &typeName_, void *(*cfunc_)(void *)) {
    PyZeno::getCFuncPtrs().emplace(typeName_, cfunc_);
    return 1;
}

ZENO_API Zeno_Error capiLastErrorCatched(std::function<void()> const &func) noexcept {
    return PyZeno::lastError.catched([&] {
        func();
    });
}

}
