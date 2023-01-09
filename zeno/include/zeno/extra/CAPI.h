#pragma once

#include <stdint.h>
#include <stddef.h>
#include "../utils/api.h"  // <zeno/utils/api.h>

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

#ifdef __cplusplus
}
#endif
