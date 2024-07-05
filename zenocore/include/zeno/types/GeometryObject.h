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
        std::set<int> edges;    //all h-edge starting from this point.
        //int hEdge = -1;       //any h-edge starting from this Point
    };

    ZENO_API GeometryObject();
    ZENO_API GeometryObject(const GeometryObject& geo);
    ZENO_API GeometryObject(PrimitiveObject* prim);
    ZENO_API std::shared_ptr<PrimitiveObject> toPrimitive() const;

private:
    void initFromPrim(PrimitiveObject* prim);
    int checkHEdge(int fromPoint, int toPoint);
    int getNextOutEdge(int fromPoint, int currentOutEdge);

    std::vector<Point> m_points;
    std::vector<Face> m_faces;
    std::vector<HEdge> m_hEdges;
};

}

#endif