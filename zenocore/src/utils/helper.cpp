#include <zeno/utils/helper.h>


namespace zeno {

    ParamType convertToType(std::string const& type) {
        if (type == "string") { return Param_String; }
        else if (type == "bool") { return Param_Bool; }
        else if (type == "int") { return Param_Int; }
        else if (type == "float") { return Param_Float; }
        else if (type == "NumericObject") { return Param_Float; }
        else if (type == "vec2i") { return Param_Vec2i; }
        else if (type == "vec3i") { return Param_Vec3i; }
        else if (type == "vec4i") { return Param_Vec4i; }
        else if (type == "vec2f") { return Param_Vec2f; }
        else if (type == "vec3f") { return Param_Vec3f; }
        else if (type == "vec4f") { return Param_Vec4f; }
        else if (type == "prim") { return Param_Prim; }
        else if (type == "list") { return Param_List; }
        else if (type == "dict") { return Param_Dict; }
        else return Param_Null;
    }

    zvariant str2var(std::string const& defl, ParamType const& type) {
        switch (type) {
        case Param_String: {
        }
        case Param_Int: {
            return std::stoi(defl);
        }
        case Param_Float: {
            return std::stof(defl);
        }
        case Param_Vec2i:
        case Param_Vec3i:
        case Param_Vec4i:
        {
            std::vector<int> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stoi(v));
            }

            if (Param_Vec2i == type) {
                return vec2i(vec[0], vec[1]);
            }
            else if (Param_Vec3i == type) {
                return vec3i(vec[0], vec[1], vec[2]);
            }
            else {
                return vec4i(vec[0], vec[1], vec[2], vec[3]);
            }
        }
        case Param_Vec2f:
        case Param_Vec3f:
        case Param_Vec4f:
        {
            std::vector<float> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stof(v));
            }

            if (Param_Vec2f == type) {
                return vec2f(vec[0], vec[1]);
            }
            else if (Param_Vec3f == type) {
                return vec3f(vec[0], vec[1], vec[2]);
            }
            else {
                return vec4f(vec[0], vec[1], vec[2], vec[3]);
            }
        }
        default:
            return nullptr;
        }
    }

    zany strToZAny(std::string const& defl, ParamType const& type) {
        switch (type) {
        case Param_String: {
            zany res = std::make_shared<zeno::StringObject>(defl);
            return res;
        }
        case Param_Int: {
            return std::make_shared<NumericObject>(std::stoi(defl));
        }
        case Param_Float: {
            return std::make_shared<NumericObject>(std::stof(defl));
        }
        case Param_Vec2i:
        case Param_Vec3i:
        case Param_Vec4i:
        {
            std::vector<int> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stoi(v));
            }

            if (Param_Vec2i == type) {
                return std::make_shared<NumericObject>(vec2i(vec[0], vec[1]));
            }
            else if (Param_Vec3i == type) {
                return std::make_shared<NumericObject>(vec3i(vec[0], vec[1], vec[2]));
            }
            else {
                return std::make_shared<NumericObject>(vec4i(vec[0], vec[1], vec[2], vec[3]));
            }
        }
        case Param_Vec2f:
        case Param_Vec3f:
        case Param_Vec4f:
        {
            std::vector<float> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stof(v));
            }

            if (Param_Vec2f == type) {
                return std::make_shared<NumericObject>(vec2f(vec[0], vec[1]));
            }
            else if (Param_Vec3f == type) {
                return std::make_shared<NumericObject>(vec3f(vec[0], vec[1], vec[2]));
            }
            else {
                return std::make_shared<NumericObject>(vec4f(vec[0], vec[1], vec[2], vec[3]));
            }
        }
        default:
            return nullptr;
        }
    }
}