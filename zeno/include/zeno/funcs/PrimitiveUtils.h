#pragma once

#include <zeno/utils/api.h>
#include <zeno/types/PrimitiveObject.h>
#include <string>

namespace zeno {

ZENO_API void primTriangulateQuads(PrimitiveObject *prim);
ZENO_API void primTriangulate(PrimitiveObject *prim, bool with_uv = true);
ZENO_API void primPolygonate(PrimitiveObject *prim, bool with_uv = true);
ZENO_API void primSepTriangles(PrimitiveObject *prim, bool keepTriFaces = true, bool withUVattr = true);

ZENO_API void primCalcNormal(PrimitiveObject *prim, float flip = 1.0f);
ZENO_API void primDecodeUVs(PrimitiveObject *prim);
ZENO_API void primLoopUVsToVerts(PrimitiveObject *prim);

ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMerge(std::vector<zeno::PrimitiveObject *> const &primList, std::string const &tagAttr = {});
ZENO_API std::shared_ptr<PrimitiveObject> primDuplicate(PrimitiveObject *parsPrim, PrimitiveObject *meshPrim, std::string dirAttr, std::string tanAttr, std::string radAttr, std::string onbType, float radius, int seed);

ZENO_API void primLineSort(PrimitiveObject *prim, bool reversed = false);
ZENO_API void primLineDistance(PrimitiveObject *prim, std::string resAttr, int start = 0);

ZENO_API void primFilterVerts(PrimitiveObject *prim, std::string tagAttr, int tagValue, bool isInversed = false);
ZENO_API void primRevampVerts(PrimitiveObject *prim, std::vector<int> const &revamp, std::vector<int> const *unrevamp_p = nullptr);

ZENO_API std::vector<std::shared_ptr<PrimitiveObject>> primUnmergeVerts(PrimitiveObject *prim, std::string tagAttr);
ZENO_API void primSimplifyTag(PrimitiveObject *prim, std::string tagAttr);
ZENO_API void primColorByTag(PrimitiveObject *prim, std::string tagAttr, std::string clrAttr);

ZENO_API void primTranslate(PrimitiveObject *prim, vec3f const &offset);
ZENO_API void primScale(PrimitiveObject *prim, vec3f const &scale);

ZENO_API std::pair<vec3f, vec3f> primBoundingBox(PrimitiveObject *prim);

ZENO_API void primRandomize(PrimitiveObject *prim, std::string attr, std::string dirAttr, std::string randType, std::string combType, float scale, int seed);
ZENO_API void primPerlinNoise(PrimitiveObject *prim, std::string inAttr, std::string outAttr, std::string outType, float scale, float detail, float roughness, float disortion, vec3f offset);

ZENO_API std::shared_ptr<PrimitiveObject> primScatter(
    PrimitiveObject *prim, std::string type, int npoints, bool interpAttrs, int seed);

}
