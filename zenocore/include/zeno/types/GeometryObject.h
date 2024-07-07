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

struct HEdge {
    int pair = -1, next = -1;
    int point = -1;  //the point which pointed by the hedge.
    int face = -1;
};

struct Face {
    int h = -1;      //any h-edge of this face.
};

struct Point {
    std::set<int> edges;    //all h-edge starting from this point.
};

class GeometryObject : public IObjectClone<GeometryObject> {
public:
    ZENO_API GeometryObject();
    ZENO_API GeometryObject(PrimitiveObject* prim);
    ZENO_API std::shared_ptr<PrimitiveObject> toPrimitive() const;
    int get_point_count() const;
    int get_face_count() const;
    std::vector<vec3f> get_points() const;

    bool has_point_attr(std::string const& name) const;

    template <class T>
    auto const& point_attr(std::string const& name) const {
        return m_points_data.attr<T>(name);
    }

    void set_points_pos(const ZfxVariable& val, ZfxElemFilter& filter);
    void set_points_normal(const ZfxVariable& val, ZfxElemFilter& filter);
    bool remove_point(int ptnum);

private:
    void initFromPrim(PrimitiveObject* prim);
    int checkHEdge(int fromPoint, int toPoint);
    int getNextOutEdge(int fromPoint, int currentOutEdge);

    std::vector<Face> m_faces;
    AttrVector<int> m_faces_data;   //no based value

    std::vector<HEdge> m_hEdges;

    std::vector<Point> m_points;
    AttrVector<vec3f> m_points_data;    //corresponding with points.

    bool m_bTriangle = true;
};

}

#endif