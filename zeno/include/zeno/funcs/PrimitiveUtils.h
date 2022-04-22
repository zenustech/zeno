#pragma once

#include <zeno/utils/api.h>
#include <zeno/types/PrimitiveObject.h>
#include <string>

namespace zeno {

ZENO_API void primTriangulateQuads(PrimitiveObject *prim);
ZENO_API void primTriangulate(PrimitiveObject *prim, bool with_uv = true);
ZENO_API void primPolygonate(PrimitiveObject *prim, bool with_uv = true);

ZENO_API void primCalcNormal(PrimitiveObject *prim, float flip = 1.0f);

ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMerge(
        std::vector<zeno::PrimitiveObject *> const &primList,
        std::string const &tagAttr = {});

ZENO_API void primLineSort(PrimitiveObject *prim, bool reversed = false);
ZENO_API void primLineDistance(PrimitiveObject *prim, std::string resAttr, int start = 0);

ZENO_API void primFilterVerts(PrimitiveObject *prim, std::string tagAttr, int tagValue, bool isInversed = false);
ZENO_API void primRevampVerts(PrimitiveObject *prim, std::vector<int> const &revamp, std::vector<int> const *unrevamp_p = nullptr);
ZENO_API void primFilterFaces(PrimitiveObject *prim, std::string tagAttr, int tagValue, bool isInversed = false);

ZENO_API void primTranslate(PrimitiveObject *prim, vec3f const &offset);
ZENO_API void primScale(PrimitiveObject *prim, vec3f const &scale);

}
