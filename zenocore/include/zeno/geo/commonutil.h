#pragma once

#include <zeno/core/coredata.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/HeatmapObject.h>
//abi兼容（除了PrimitiveObject...)

namespace zeno
{
    ZENO_API void primTriangulate(PrimitiveObject* prim, bool with_uv = true, bool has_lines = true, bool with_attr = true);
    ZENO_API void primTriangulateQuads(PrimitiveObject* prim);
    ZENO_API std::unique_ptr<zeno::PrimitiveObject> PrimMerge(Vector<zeno::PrimitiveObject*> const& primList, String const& tagAttr = "", bool tag_on_vert = true, bool tag_on_face = false);
    ZENO_API void primPolygonate(PrimitiveObject* prim, bool with_uv = true);
    ZENO_API void primFlipFaces(PrimitiveObject* prim, bool only_face = false);
    ZENO_API void primCalcNormal(zeno::PrimitiveObject* prim, float flip = 1.0f, String nrmAttr = "nrm");
    ZENO_API zeno::Vector<std::unique_ptr<PrimitiveObject>> get_prims_from_list(ListObject* spList);
    ZENO_API zeno::Vector<std::unique_ptr<PrimitiveObject>> PrimUnmergeFaces(PrimitiveObject* prim, String tagAttr);
    ZENO_API void primKillDeadVerts(PrimitiveObject* prim);
    ZENO_API void prim_set_abcpath(PrimitiveObject* prim, zeno::String path_name);
    ZENO_API void prim_set_faceset(PrimitiveObject* prim, zeno::String faceset_name);
    ZENO_API void primSampleHeatmap(
        PrimitiveObject* prim,
        const zeno::String& srcChannel,
        const zeno::String& dstChannel,
        HeatmapObject* heatmap,
        float remapMin,
        float remapMax);
    ZENO_API void primSampleTexture(
        PrimitiveObject* prim,
        const zeno::String& srcChannel,
        const zeno::String& srcSource,
        const zeno::String& dstChannel,
        PrimitiveObject* img,
        const zeno::String& wrap,
        zeno::Vec3f borderColor,
        float remapMin,
        float remapMax
    );
    ZENO_API std::unique_ptr<zeno::PrimitiveObject> primMergeWithFacesetMatid(Vector<zeno::PrimitiveObject*> const& primList, String const& tagAttr = "", bool tag_on_vert = true, bool tag_on_face = false);
    ZENO_API void prim_copy_faceset_to_matid(PrimitiveObject* prim);
    ZENO_API void primWireframe(PrimitiveObject* prim, bool removeFaces = false, bool toEdges = false);
    ZENO_API std::unique_ptr<PrimitiveObject> readImageFile(zeno::String const& path);
    ZENO_API std::unique_ptr<PrimitiveObject> readExrFile(zeno::String const& path);
    ZENO_API std::unique_ptr<PrimitiveObject> readImageFile(zeno::String const& path);
    ZENO_API std::unique_ptr<PrimitiveObject> readPFMFile(zeno::String const& path);
    ZENO_API void write_pfm(const zeno::String& path, PrimitiveObject* image);
    ZENO_API void write_jpg(const zeno::String& path, PrimitiveObject* image);
}