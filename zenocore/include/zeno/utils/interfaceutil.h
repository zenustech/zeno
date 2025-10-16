#pragma once

#include <zeno/core/coredata.h>
#include <zeno/utils/helper.h>
#include <zeno/utils/vec.h>

namespace zeno
{
    template<class T>
    std::vector<T> zeVec2stdVec(const zeno::Vector<T>& rhs) {
        std::vector<T> vec(rhs.size());
        for (int i = 0; i < rhs.size(); i++) {
            vec[i] = rhs[i];
        }
        return vec;
    }

    template<class T>
    zeno::Vector<T> stdVec2zeVec(const std::vector<T>& rhs) {
        zeno::Vector<T> vec(rhs.size());
        for (int i = 0; i < rhs.size(); i++) {
            vec[i] = rhs[i];
        }
        return vec;
    }

    inline std::string zsString2Std(const zeno::String& zs) { return std::string(zs.c_str()); }
    inline zeno::String stdString2zs(const std::string& ss) { return zeno::String(ss.c_str()); }

    inline zeno::vec2f toVec2f(const zeno::Vec2f& rhs) { return zeno::vec2f(rhs.x, rhs.y); }
    inline zeno::Vec2f toAbiVec2f(const zeno::vec2f& rhs) { return zeno::Vec2f{ rhs[0], rhs[1] }; }

    inline zeno::vec2i toVec2i(const zeno::Vec2i& rhs) { return zeno::vec2i(rhs.x, rhs.y); }
    inline zeno::Vec2i toAbiVec2i(const zeno::vec2i& rhs) { return zeno::Vec2i{ rhs[0], rhs[1] }; }

    inline zeno::vec3f toVec3f(const zeno::Vec3f& rhs) { return zeno::vec3f(rhs.x, rhs.y, rhs.z); }
    inline zeno::Vec3f toAbiVec3f(const zeno::vec3f& rhs) { return zeno::Vec3f{ rhs[0], rhs[1], rhs[2] }; }

    inline zeno::vec3i toVec3i(const zeno::Vec3i& rhs) { return zeno::vec3i(rhs.x, rhs.y, rhs.z); }
    inline zeno::Vec3i toAbiVec3i(const zeno::vec3i& rhs) { return zeno::Vec3i{ rhs[0], rhs[1], rhs[2] }; }

    inline zeno::vec4f toVec4f(const zeno::Vec4f& rhs) { return zeno::vec4f(rhs.x, rhs.y, rhs.z, rhs.w); }
    inline zeno::Vec4f toAbiVec4f(const zeno::vec4f& rhs) { return zeno::Vec4f{ rhs[0], rhs[1], rhs[2], rhs[3] }; }

    inline zeno::vec4i toVec4i(const zeno::Vec4i& rhs) { return zeno::vec4i(rhs.x, rhs.y, rhs.z, rhs.w); }
    inline zeno::Vec4i toAbiVec4i(const zeno::vec4i& rhs) { return zeno::Vec4i{ rhs[0], rhs[1], rhs[2], rhs[3] }; }


    #define get_input2_string_(name)    zeno::zsString2Std(get_input2_string(name))
    #define get_input2_vec2i_(name)     zeno::toVec2i(get_input2_vec2i(name))
    #define get_input2_vec2f_(name)     zeno::toVec2f(get_input2_vec2f(name))
    #define get_input2_vec3i_(name)     zeno::toVec3i(get_input2_vec3i(name))
    #define get_input2_vec3f_(name)     zeno::toVec3f(get_input2_vec3f(name))
    #define get_input2_vec4i_(name)     zeno::toVec4i(get_input2_vec4i(name))
    #define get_input2_vec4f_(name)     zeno::toVec4f(get_input2_vec4f(name))

}