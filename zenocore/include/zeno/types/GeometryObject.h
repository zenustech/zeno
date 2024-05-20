#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <vector>
#include <string>
#include <zeno/core/common.h>
#include <zeno/core/IObject.h>

namespace zeno
{

using Geo_Attribute = std::pair<std::string, AttrValue>;

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
    struct Point
    {
        vec3f pos;
        Geo_Attributes attr;
    };
    struct Vertex
    {
        int index = 0;  //index to m_points;
        Geo_Attributes attr;
    };
    struct Primitive
    {
        std::vector<Vertex> vertices;
        Geo_Attributes attr;
    };

    GeometryObject();
    GeometryObject(const GeometryObject& geo);

private:
    std::vector<Primitive> m_prims;
    std::vector<Point> m_points;
};

}

#endif