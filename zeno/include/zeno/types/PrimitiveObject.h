#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/AttrVector.h>
#include <zeno/utils/type_traits.h>
#include <zeno/utils/vec.h>
#include <optional>
#include <variant>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace zeno {

struct MaterialObject;
struct InstancingObject;
/*
    Assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
    compute the sum of the cross product of every triangle of a triangle
    fan of the polygon (even in the concave parts):
      normal = vector3(0, 0, 0);
    for(int i=1; i<n-1; ++i)
        normal += cross(p[i+1]-p[0],p[i]-p[0]);
    normal /= norm(n);
    This is derived from the Stoke's theorem.
    The length of the normal (before normalization)
    is equal to two times the signed area enclosed by the loop.
    So you are guaranteed that it will not be null (if the area is not null)
    and you are guaranteed that it will point in the right direction.

*/
static zeno::vec3f polyNormal(std::vector<zeno::vec3f> & verts, std::vector<int> &poly)
{
    zeno::vec3f normal = zeno::vec3f(0,0,0);
    for(int i=1;i<poly.size()-1;i++)
    {
        normal += zeno::cross(verts[poly[i+1]]-verts[poly[0]], verts[poly[i]]-verts[poly[0]]);
    }
    normal = -zeno::normalize(normal);
    return normal;
}
static char isReflex(zeno::vec3f a, zeno::vec3f b, zeno::vec3f c, zeno::vec3f n)
{
    
    return zeno::dot(zeno::cross(b - a, c - b), n) <= 0 ? 1 : 0;
}
static void markReflex(std::vector<zeno::vec3f> &verts, std::vector<int> &poly, std::vector<char> &reflexMark,
                       zeno::vec3f n) 
{
    reflexMark.resize(poly.size());
    for(int i=0;i<poly.size();i++)
    {
        int prev = i==0? (poly.size()-1) : (i-1);
        int next = i==(poly.size()-1)? 0 : (i+1);
        reflexMark[i] = isReflex(verts[poly[prev]], verts[poly[i]], verts[poly[next]], n);
    }
}
static void takeOutTriangle(std::vector<zeno::vec3f> & verts, std::vector<int> &poly, std::vector<char>& reflexMark, std::vector<zeno::vec3i> & triangles)
{
    if(poly.size()==3)
    {
        triangles.push_back(zeno::vec3i(poly[0], poly[1], poly[2]));
        poly.resize(0);
        return;
    }else
    {
        //find 3 consecutive points which satisfy:
        //if there are no reflex points, any 3 points
        //if there are at least one reflex point, a must be the reflex point, and b must be a non-reflex point
        char anyReflex = 0;
        for(auto r:reflexMark)
        {
            //printf("r : %d\n", r);
            anyReflex = anyReflex | r;
        }


        int a = 0;
        int b = 1;
        int c = 2;
        while (zeno::length(zeno::cross(verts[poly[b]] - verts[poly[a]], verts[poly[c]] - verts[poly[b]])) <=
               0.0000001)
        {
            a++;
            b++;
            c++;
        }
        
        if(anyReflex == true)
        {
            while(a < poly.size())
            {
                b = (a + 1) % poly.size();
                if(reflexMark[a] == true && reflexMark[b] == false)
                    break;
                a++;
            }
        }
        c = (a+2) % poly.size();

        triangles.push_back(zeno::vec3i(poly[a], poly[b], poly[c]));
        //printf("tridx : %d, %d, %d\n", poly[a], poly[b], poly[c]);
        //printf("tridx : %d, %d, %d\n", a, b, c);
        //remove b from the poly loop
        poly.erase(poly.begin()+b);
        //recompute reflexMark
        auto n = polyNormal(verts, poly);
        //printf("n:%f, %f, %f\n", n[0], n[1], n[2]);
        markReflex(verts, poly, reflexMark, n);
        //recursively take out triangle from the rest of things
        takeOutTriangle(verts, poly, reflexMark, triangles);
    }
}
static std::vector<vec3i> polygonDecompose(std::vector<zeno::vec3f> & verts, std::vector<int> &poly) {
    std::vector<char> reflexMark;
    auto n = polyNormal(verts, poly);
    markReflex(verts, poly, reflexMark, n);
    std::vector<vec3i> triangles;
    takeOutTriangle(verts, poly, reflexMark, triangles);
    return triangles;
}

struct PrimitiveObject : IObjectClone<PrimitiveObject> {
    AttrVector<vec3f> verts;
    AttrVector<int> points;
    AttrVector<vec2i> lines;
    AttrVector<vec3i> tris;
    AttrVector<vec4i> quads;

    AttrVector<int> loops;
    AttrVector<vec2i> polys;
    AttrVector<vec2i> edges;

    AttrVector<vec2f> uvs;

    std::shared_ptr<MaterialObject> mtl;
    std::shared_ptr<InstancingObject> inst;

    // deprecated:
    template <class Accept = std::variant<vec3f, float>, class F>
    void foreach_attr(F &&f) {
        std::string pos_name = "pos";
        f(pos_name, verts.values);
        verts.foreach_attr<Accept>(std::move(f));
    }

    // deprecated:
    template <class Accept = std::variant<vec3f, float>, class F>
    void foreach_attr(F &&f) const {
        std::string const pos_name = "pos";
        f(pos_name, verts.values);
        verts.foreach_attr<Accept>(std::move(f));
    }

    // deprecated:
    size_t num_attrs() const {
        return 1 + verts.num_attrs();
    }

    // deprecated:
    auto attr_keys() const {
        auto keys = verts.attr_keys();
        keys.insert(keys.begin(), "pos");
        return keys;
    }

    // deprecated:
    template <class T>
    auto &add_attr(std::string const &name) {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        } else {
            if (name == "pos") throw makeError<TypeError>(
                typeid(vec3f), typeid(T), "attribute 'pos' must be vec3f");
        }
        return verts.add_attr<T>(name);
    }

    // deprecated:
    template <class T>
    auto &add_attr(std::string const &name, T const &value) {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        } else {
            if (name == "pos") throw makeError<TypeError>(
                typeid(vec3f), typeid(T), "attribute 'pos' must be vec3f");
        }
        if (verts.attr_is<T>(name)) {
            return verts.attr<T>(name);
        } else {
            auto &ret = verts.add_attr<T>(name);
            ret.assign(size(), value);
            return ret;
        }
    }

    // deprecated:
    template <class T>
    auto const &attr(std::string const &name) const {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        } else {
            if (name == "pos") throw makeError<TypeError>(
                typeid(vec3f), typeid(T), "attribute 'pos' must be vec3f");
        }
        return verts.attr<T>(name);
    }

    // deprecated:
    template <class T>
    auto &attr(std::string const &name) {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return verts.values;
        } else {
            if (name == "pos") throw makeError<TypeError>(
                typeid(vec3f), typeid(T), "attribute 'pos' must be vec3f");
        }
        return verts.attr<T>(name);
    }

    // deprecated:
    auto const &attr(std::string const &name) const {
        //if (name == "pos") return verts.values;
        return verts.attr(name);
    }

    // deprecated:
    auto &attr(std::string const &name) {
        //if (name == "pos") return verts.values;
        return verts.attr(name);
    }

    // deprecated:
    template <class Accept = std::variant<vec3f, float>, class F>
    auto attr_visit(std::string const &name, F const &f) const {
        if (name == "pos") {
            return f(verts.values);
        } else {
            return verts.attr_visit<Accept>(name, f);
        }
    }

    // deprecated:
    template <class Accept = std::variant<vec3f, float>, class F>
    auto attr_visit(std::string const &name, F const &f) {
        if (name == "pos") {
            return f(verts.values);
        } else {
            return verts.attr_visit<Accept>(name, f);
        }
    }

    // deprecated:
    bool has_attr(std::string const &name) const {
        if (name == "pos") return true;
        return verts.has_attr(name);
    }

    // deprecated:
    template <class T>
    bool attr_is(std::string const &name) const {
        if constexpr (std::is_same_v<T, vec3f>) {
            if (name == "pos") return true;
        } else {
            if (name == "pos") return false;
        }
        return verts.attr_is<T>(name);
    }

    size_t size() const {
        return verts.size();
    }

    void resize(size_t size) {
        return verts.resize(size);
    }
    // end of deprecated
};

} // namespace zeno
