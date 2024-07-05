#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <vector>
#include <string>
#include <zeno/core/common.h>
#include <zeno/core/IObject.h>

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


class GeometryObject : public IObjectClone<GeometryObject> {
public:

    //struct Vertex
    //{
    //    int index = 0;  //index to m_points;
    //    Geo_Attributes attr;
    //};
    //struct Primitive
    //{
    //    std::vector<Vertex> vertices;
    //    Geo_Attributes attr;
    //};

    struct HEdge {
        int pair = -1, next = -1;
        int point = -1;  //the point which pointed by the hedge.
        int face = -1;
    };

    struct Face {
        int h = -1;      //any h-edge of this face.
    };

    struct Point {
        vec3f pos;
        //Geo_Attributes attr;  //TODO: attr supporting
        int hEdge = -1;      //any h-edge starting from this Point
    };

    ZENO_API GeometryObject();
    ZENO_API GeometryObject(const GeometryObject& geo);
    ZENO_API GeometryObject(PrimitiveObject* prim);

private:
    void initFromPrim(PrimitiveObject* prim);
    int checkHEdge(int fromPoint, int toPoint);
    int getNextOutEdge(int fromPoint, int currentOutEdge);
    int visit_allHalfEdge_from(int fromPoint, std::function<bool(int)> f);

    std::vector<Point> m_points;
    std::vector<Face> m_faces;
    std::vector<HEdge> m_hEdges;
};

}

#endif