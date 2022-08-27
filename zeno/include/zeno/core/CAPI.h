#pragma once

#include <stdint.h>
#include <stddef.h>
#include "../utils/api.h"  // <zeno/utils/api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t Zeno_Graph;
typedef uint64_t Zeno_Object;
typedef uint32_t Zeno_Error;

#define ZENO_CAPI ZENO_API

#ifdef __cplusplus
#define ZENO_CAPI_NOEXCEPT noexcept
#else
#define ZENO_CAPI_NOEXCEPT
#endif

ZENO_CAPI Zeno_Error Zeno_GetLastErrorCode() ZENO_CAPI_NOEXCEPT;
ZENO_CAPI const char *Zeno_GetLastErrorStr() ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateGraph(Zeno_Graph *graphRet_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_DestroyGraph(Zeno_Graph graph_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GraphLoadJson(Zeno_Graph graph_, const char *jsonStr_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GraphCallTempNode(Zeno_Graph graph_, const char *nodeType_, const char *const *inputKeys_, const Zeno_Object *inputObjects_, size_t inputCount_, size_t *outputCount_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetLastTempNodeResult(const char **outputKeys_, Zeno_Object *outputObjects_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateObjectInt(Zeno_Object *objectRet_, const int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateObjectFloat(Zeno_Object *objectRet_, const float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_CreateObjectString(Zeno_Object *objectRet_, const char *str_, size_t strLen_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_DestroyObject(Zeno_Object object_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectInt(Zeno_Object object_, int *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectFloat(Zeno_Object object_, float *value_, size_t dim_) ZENO_CAPI_NOEXCEPT;
ZENO_CAPI Zeno_Error Zeno_GetObjectString(Zeno_Object object_, char *strBuf_, size_t *strLenRet_) ZENO_CAPI_NOEXCEPT;

#ifdef __cplusplus
}
#endif
