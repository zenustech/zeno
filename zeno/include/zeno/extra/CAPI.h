#pragma once

#include <stdint.h>
#include <stddef.h>

#include <cstring>
#include <set>
#include <stdexcept>
#include <memory>

#include "../utils/api.h"  // <zeno/utils/api.h>
#include <zeno/zeno.h>
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

#ifdef __cplusplus
extern "C" {
#endif

#define ZENO_CAPI ZENO_API

#ifdef __cplusplus
#define ZENO_CAPI_NOEXCEPT noexcept
#else
#define ZENO_CAPI_NOEXCEPT
#endif

typedef uint64_t Zeno_Graph;
typedef uint64_t Zeno_Object;
typedef uint32_t Zeno_Error;

enum Zeno_PrimMembType {
    Zeno_PrimMembType_verts = 0,
    Zeno_PrimMembType_points,
    Zeno_PrimMembType_lines,
    Zeno_PrimMembType_tris,
    Zeno_PrimMembType_quads,
    Zeno_PrimMembType_loops,
    Zeno_PrimMembType_polys,
    Zeno_PrimMembType_uvs,
};

enum Zeno_PrimDataType {
    Zeno_PrimDataType_vec3f = 0,
    Zeno_PrimDataType_float,
    Zeno_PrimDataType_vec3i,
    Zeno_PrimDataType_int,
    Zeno_PrimDataType_vec2f,
    Zeno_PrimDataType_vec2i,
    Zeno_PrimDataType_vec4f,
    Zeno_PrimDataType_vec4i,
};

ZENO_CAPI Zeno_Error Zeno_GetLastError(const char **msgRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateGraph(Zeno_Graph *graphRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_DestroyGraph(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetCurrentGraph(Zeno_Graph *graphRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GraphIncReference(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GraphGetSubGraph(Zeno_Graph graph_, Zeno_Graph *retGraph_, const char *subName_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GraphLoadJson(Zeno_Graph graph_, const char *jsonStr_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GraphCallTempNode(Zeno_Graph graph_, const char *nodeType_, const char *const *inputKeys_, const Zeno_Object *inputObjects_, size_t inputCount_, size_t *outputCount_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetLastTempNodeResult(const char **outputKeys_, Zeno_Object *outputObjects_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateObjectInt(Zeno_Object *objectRet_, const int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateObjectFloat(Zeno_Object *objectRet_, const float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateObjectString(Zeno_Object *objectRet_, const char *str_, size_t strLen_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_DestroyObject(Zeno_Object object_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_ObjectIncReference(Zeno_Object object_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectLiterialType(Zeno_Object object_, int *typeRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectInt(Zeno_Object object_, int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectFloat(Zeno_Object object_, float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectString(Zeno_Object object_, char *strBuf_, size_t *strLenRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectPrimData(Zeno_Object object_, Zeno_PrimMembType primArrType_, const char *attrName_, void **ptrRet_, size_t *lenRet_, Zeno_PrimDataType *typeRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_AddObjectPrimAttr(Zeno_Object object_, Zeno_PrimMembType primArrType_, const char *attrName_, Zeno_PrimDataType dataType_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectPrimDataKeys(Zeno_Object object_, Zeno_PrimMembType primArrType_, size_t *lenRet_, const char **keysRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_ResizeObjectPrimData(Zeno_Object object_, Zeno_PrimMembType primArrType_, size_t newSize_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_InvokeObjectFactory(Zeno_Object *objectRet_, const char *typeName_, void *ffiObj_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_InvokeObjectDefactory(Zeno_Object object_, const char *typeName_, void **ffiObjRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_InvokeCFunctionPtr(void *ffiObjArg_, const char *typeName_, void **ffiObjRet_) ZENO_CAPI_NOEXCEPT;

enum ZS_DataType {
    ZS_DataType_int = 0,
    ZS_DataType_float,
    ZS_DataType_double
};

ZENO_CAPI Zeno_Error ZS_CreateObjectZsSmallVecInt(Zeno_Object *objectRet_, const int *value_, size_t dim_x_, size_t dim_y_) ZENO_CAPI_NOEXCEPT; 
ZENO_CAPI Zeno_Error ZS_GetObjectZsVecData(Zeno_Object object_, void **ptrRet_, size_t *dims_Ret_, size_t *dim_xRet_, size_t *dim_yRet_, ZS_DataType *typeRet_, void** data_ptr) ZENO_CAPI_NOEXCEPT; 

#ifdef __cplusplus
}
#endif

namespace PyZeno
{
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
            throw zeno::makeError<zeno::KeyError>(std::to_string(key), zeno::cppdemangle(typeid(T)));
        return it->first;
    }

    void destroy(uint64_t key) {
        T *raw_p = reinterpret_cast<T *>(static_cast<uint64_t>(key));
        auto it = lut.find(make_stale_shared(raw_p));
        if (ZENO_UNLIKELY(it == lut.end()))
            throw zeno::makeError<zeno::KeyError>(std::to_string(key), zeno::cppdemangle(typeid(T)));
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
            zeno::log_debug("Zeno API catched error: {}", message);
        } catch (...) {
            errcode = 1;
            message = "(unknown)";
            zeno::log_debug("Zeno API catched unknown error");
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

extern LUT<zeno::Session> lutSession;
extern LUT<zeno::Graph> lutGraph;
extern LUT<zeno::IObject> lutObject;
extern LastError lastError;
extern std::map<std::string, std::shared_ptr<zeno::IObject>> tempNodeRes;
extern std::shared_ptr<zeno::Graph> currentGraph;

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