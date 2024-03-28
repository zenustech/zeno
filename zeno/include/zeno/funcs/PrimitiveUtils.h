#pragma once

#include <zeno/utils/api.h>
#include <zeno/types/PrimitiveObject.h>
#include <string>

namespace zeno {

ZENO_API PrimitiveObject* primParsedFrom(const char *binData, std::size_t binSize);

ZENO_API void primTriangulateQuads(PrimitiveObject *prim);
ZENO_API void primTriangulate(PrimitiveObject *prim, bool with_uv = true, bool has_lines = true, bool with_attr = true);
ZENO_API void primTriangulateIntoPolys(PrimitiveObject *prim);
ZENO_API void primPolygonate(PrimitiveObject *prim, bool with_uv = true);

ZENO_API void primSepTriangles(PrimitiveObject *prim, bool smoothNormal = true, bool keepTriFaces = true);
//ZENO_API void primSmoothNormal(PrimitiveObject *prim, bool isFlipped = false);

ZENO_API void primFlipFaces(PrimitiveObject *prim);
ZENO_API void primCalcNormal(PrimitiveObject *prim, float flip = 1.0f, std::string nrmAttr = "nrm");
//ZENO_API void primCalcInsetDir(PrimitiveObject *prim, float flip = 1.0f, std::string nrmAttr = "nrm");

ZENO_API void primWireframe(PrimitiveObject *prim, bool removeFaces = false, bool toEdges = false);
ZENO_API void primEdgeBound(PrimitiveObject *prim, bool removeFaces = false, bool toEdges = false);
ZENO_API void primKillDeadVerts(PrimitiveObject *prim);

ZENO_API void primDecodeUVs(PrimitiveObject *prim);
ZENO_API void primLoopUVsToVerts(PrimitiveObject *prim);

ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMerge(std::vector<zeno::PrimitiveObject *> const &primList, std::string const &tagAttr = {});
ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMergeWithFacesetMatid(std::vector<zeno::PrimitiveObject *> const &primList, std::string const &tagAttr = {});
ZENO_API std::shared_ptr<PrimitiveObject> primDuplicate(PrimitiveObject *parsPrim, PrimitiveObject *meshPrim, std::string dirAttr = {}, std::string tanAttr = {}, std::string radAttr = {}, std::string onbType = "XYZ", float radius = 1.f, bool copyParsAttr = true, bool copyMeshAttr = true);

ZENO_API void primLineSort(PrimitiveObject *prim, bool reversed = false);
ZENO_API void primLineDistance(PrimitiveObject *prim, std::string resAttr, int start = 0);
ZENO_API void prim_set_abcpath(PrimitiveObject* prim, std::string path_name);
ZENO_API void prim_set_faceset(PrimitiveObject* prim, std::string faceset_name);

ZENO_API void primFilterVerts(PrimitiveObject *prim, std::string tagAttr, int tagValue, bool isInversed = false, std::string revampAttrO = {}, std::string method = "verts");

ZENO_API void primMarkIsland(PrimitiveObject *prim, std::string tagAttr);
ZENO_API std::vector<std::shared_ptr<PrimitiveObject>> primUnmergeVerts(PrimitiveObject *prim, std::string tagAttr);
ZENO_API std::vector<std::shared_ptr<PrimitiveObject>> primUnmergeFaces(PrimitiveObject *prim, std::string tagAttr);

ZENO_API void primSimplifyTag(PrimitiveObject *prim, std::string tagAttr);
ZENO_API void primColorByTag(PrimitiveObject *prim, std::string tagAttr, std::string clrAttr, int seed = -1);

ZENO_API void primTranslate(PrimitiveObject *prim, vec3f const &offset);
ZENO_API void primScale(PrimitiveObject *prim, vec3f const &scale);

ZENO_API std::pair<vec3f, vec3f> primBoundingBox(PrimitiveObject *prim);

ZENO_API void primRandomize(PrimitiveObject *prim, std::string attr, std::string dirAttr, std::string seedAttr, std::string randType, float base, float scale, int seed);
ZENO_API void primPerlinNoise(PrimitiveObject *prim, std::string inAttr, std::string outAttr, std::string outType, float scale, float detail, float roughness, float disortion, vec3f offset, float average, float strength);

ZENO_API std::shared_ptr<PrimitiveObject> primScatter(
    PrimitiveObject *prim, std::string type, std::string denAttr, float density, float minRadius, bool interpAttrs, int seed);

}
