#ifndef __GEOMETRY_UTIL_H__
#define __GEOMETRY_UTIL_H__

#include <vector>
#include <tuple>
#include <array>
#include <zeno/types/ListObject.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/types/IGeometryObject.h>
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace zeno
{
    using namespace zeno::reflect;

    enum Rotate_Orientaion
    {
        Orientaion_YZ,
        Orientaion_XY,
        Orientaion_ZX
    };

    enum PointSide {
        UnDecided,
        Below,
        Above,
        Both,   //在平面上
    };

    struct DividePoint
    {
        vec3f pos;  //分割点的坐标
        int from;   //分割点所在线段的起点
        int to;     //分割点所在线段的终点，并有from < to，如果分割点恰好是顶点，则from=to
    };

    enum DivideKeep {
        Keep_Both,
        Keep_Below,
        Keep_Above
    };

    struct DivideFace
    {
        std::vector<int> face_indice;
        PointSide side;
    };

    struct GeometryObject;

    bool prim_remove_point(GeometryObject* prim, int ptnum);
    ZENO_API std::vector<vec3f> calc_point_normals(GeometryObject* prim, const std::vector<vec3i>& tris, float flip = 1.0f);
    ZENO_API std::vector<vec3f> calc_triangles_tangent(
        GeometryObject* geo,
        bool has_uv,
        const std::vector<vec3i>& tris,
        const std::vector<vec3f>& pos,
        const std::vector<vec3f>& uv,
        const std::vector<vec3f>& uv0,
        const std::vector<vec3f>& uv1,
        const std::vector<vec3f>& uv2
        );
    ZENO_API std::vector<vec3f> compute_vertex_tangent(
        const std::vector<vec3i>& tris,
        const std::vector<vec3f>& tang,
        const std::vector<vec3f>& pos
        );
    ZENO_API glm::mat4 calc_rotate_matrix(
        float xangle,
        float yangle,
        float zangle,
        Rotate_Orientaion orientaion
        );
    ZENO_API std::pair<vec3f, vec3f> geomBoundingBox(GeometryObject* geo);
    ZENO_API std::optional<std::pair<vec3f, vec3f>> geomBoundingBox2(GeometryObject* prim);
    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> mergeObjects(
        zeno::ListObject* spList,
        std::string const& tagAttr = {},
        bool tag_on_vert = true,
        bool tag_on_face = false);
    ZENO_API void geom_set_abcpath(GeometryObject_Adapter* prim, zeno::String path_name);
    ZENO_API void geom_set_faceset(GeometryObject_Adapter* prim, zeno::String faceset_name);
    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> fuseGeometry(zeno::GeometryObject_Adapter* input, float threshold);
    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> constructGeom(const std::vector<std::vector<zeno::vec3f>>& faces);
    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> scatter(
        zeno::GeometryObject_Adapter* input,
        const std::string& sampleRegion,
        const int count,
        int seed);
    ZENO_API std::unique_ptr<zeno::GeometryObject_Adapter> divideObject(
        zeno::GeometryObject_Adapter* input_object,
        DivideKeep keep,
        zeno::vec3f center_pos,
        zeno::vec3f direction
    );
    ZENO_API bool dividePlane(const std::vector<vec3f>& face_pts, float A, float B, float C, float D,
        /*out*/std::vector<DivideFace>& split_faces,
        /*out*/std::map<int, DividePoint>& split_infos,
        /*out*/PointSide& which_side_if_failed
        );
}


#endif