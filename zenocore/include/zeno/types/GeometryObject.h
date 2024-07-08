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
    bool remove_point(int ptnum);

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