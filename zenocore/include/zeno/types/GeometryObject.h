#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <vector>
#include <string>
#include <zeno/types/AttrVector.h>
#include <zeno/core/common.h>
#include <zeno/core/IObject.h>
#include <zeno/core/FunctionManager.h>

namespace zeno
{

using Geo_Attribute = std::pair<std::string, AttrValue>;

struct PrimitiveObject;

struct Geo_Attributes : std::vector<Geo_Attribute>
{
    template <class T>
    T get_attr(const std::string& name) const {
        for (auto iter = this->begin(); iter != this->end(); iter++) {
            if (iter->first == name) {
                return iter->second;
            }
        }
        throw makeError<zeno::KeyError>("");
    }
};

struct HEdge;
struct Face;
struct Point;

struct HEdge {
    std::string id;
    HEdge* pair = 0, *next = 0;
    int point = -1;
    int face = -1;
};

struct Face {
    HEdge* h = 0;      //any h-edge of this face.
    std::map<std::string, zfxvariant> attr;
};

struct Point {
    vec3f pos = { 0, 0, 0 };
    vec3f normal = { 0, 0, 0 };
    std::map<std::string, zfxvariant> attr;
    std::set<HEdge*> edges;    //all h-edge starting from this point.
};

class GeometryObject : public IObjectClone<GeometryObject> {
public:
    ZENO_API GeometryObject();
    ZENO_API GeometryObject(const GeometryObject& rhs);
    ZENO_API GeometryObject(PrimitiveObject* prim);
    ZENO_API std::shared_ptr<PrimitiveObject> toPrimitive() const;
    ZENO_API ~GeometryObject();

    int get_point_count() const;
    int get_face_count() const;
    std::vector<vec3f> get_points() const;

    bool has_point_attr(std::string const& name) const;
    std::vector<zfxvariant> get_point_attr(std::string const& name) const;

    void set_points_pos(const ZfxVariable& val, ZfxElemFilter& filter);
    void set_points_normal(const ZfxVariable& val, ZfxElemFilter& filter);

    //API:
    //给定 face_id 和 vert_id，返回顶点索引编号 point_idx。
    int facepoint(int face_id, int vert_id) const;

    //通过 face_id，获取此 face 所有 points 索引编号。
    zfxintarr facepoints(int face_id);

    //返回包含指定 point 的 prim 列表。
    zfxintarr pointfaces(int point_id);
    zfxintarr pointvertex(int point_id);

    bool createFaceAttr(const std::string& attr_name, const zfxvariant& defl);
    bool setFaceAttr(const std::string& attr_name, const zfxvariant& val);
    std::vector<zfxvariant> getFaceAttr(const std::string& attr_name) const;
    bool deleteFaceAttr(const std::string& attr_name);

    bool createPointAttr(const std::string& attr_name, const zfxvariant& defl);
    bool setPointAttr(const std::string& attr_name, const zfxvariant& val);
    std::vector<zfxvariant> getPointAttr(const std::string& attr_name) const;
    bool deletePointAttr(const std::string& attr_name);

    int addpoint(zfxvariant pos = zfxfloatarr({0,0,0}));
    void addprim();
    int addvertex(int face_id, int point_id);

    bool remove_point(int ptnum);
    bool remove_faces(const std::set<int>& faces, bool includePoints);

    int npoints() const;
    int nfaces() const;
    int nvertices() const;
    int nvertices(int face_id) const;

    //vertex先不考虑

private:
    void initFromPrim(PrimitiveObject* prim);
    HEdge* checkHEdge(int fromPoint, int toPoint);
    std::tuple<Point*, HEdge*, HEdge*> getPrev(HEdge* outEdge);
    int getNextOutEdge(int fromPoint, int currentOutEdge);
    int getPointTo(HEdge* hedge) const;

    std::vector<std::shared_ptr<Face>> m_faces;

    std::unordered_map<std::string, std::shared_ptr<HEdge>> m_hEdges;

    std::vector<std::shared_ptr<Point>> m_points;
    bool m_bTriangle = true;
};

}

#endif