#include <zeno/utils/helper.h>


namespace zeno {

    ZENO_API ParamType convertToType(std::string const& type) {
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
        else if (type == "colorvec3f") { return Param_Vec3f; }
        else return Param_Null;
    }

    ZENO_API zvariant str2var(std::string const& defl, ParamType const& type) {
        switch (type) {
        case Param_String: {
            return defl;
        }
        case Param_Bool: {
            if (defl == "0" || defl == "false")    return 0;
            if (defl == "1" || defl == "true")     return 1;
            return zvariant();
        }
        case Param_Int: {
            if (defl == "") return 0;
            return std::stoi(defl);
        }
        case Param_Float: {
            if (defl == "") return 0;
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
            return zvariant();
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
                if (vec.size() != 2)
                    return vec2f();
                return vec2f(vec[0], vec[1]);
            }
            else if (Param_Vec3f == type) {
                if (vec.size() != 3)
                    return vec3f();
                return vec3f(vec[0], vec[1], vec[2]);
            }
            else {
                if (vec.size() != 4)
                    return vec4f();
                return vec4f(vec[0], vec[1], vec[2], vec[3]);
            }
            return zvariant();
        }
        default:
            return zvariant();
        }
    }

    EdgeInfo getEdgeInfo(std::shared_ptr<ILink> spLink) {
        EdgeInfo edge;
        auto spOutParam = spLink->fromparam.lock();
        auto spInParam = spLink->toparam.lock();
        if (!spOutParam || !spInParam)
            return edge;

        auto spOutNode = spOutParam->m_wpNode.lock();
        auto spInNode = spInParam->m_wpNode.lock();
        if (!spOutNode || !spInNode)
            return edge;

        const std::string& outNode = spOutNode->get_name();
        const std::string& outParam = spOutParam->name;
        const std::string& inNode = spInNode->get_name();
        const std::string& inParam = spInParam->name;
        edge = { outNode, outParam, "", inNode, inParam, spLink->keyName };
        return edge;
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

    bool isEqual(const zvariant& lhs, const zvariant& rhs, ParamType const type) {
        if (lhs.index() != rhs.index())
            return false;

        std::visit([&](auto&& arg1, auto&& arg2) -> bool {
            using T = std::decay_t<decltype(arg1)>;
            using E = std::decay_t<decltype(arg2)>;
            if constexpr (std::is_same_v<T, int> && std::is_same_v<E, int>) {
                return arg1 == arg2;
            }
            else if constexpr (std::is_same_v<T, float> && std::is_same_v<E, float>) {
                return arg1 == arg2;
            }
            else if constexpr (std::is_same_v<T, std::string> && std::is_same_v<E, std::string>) {
                return arg1 == arg2;
            }
            else if constexpr (std::is_same_v<T, zeno::vec2i> && std::is_same_v<E, zeno::vec2i>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec2f> && std::is_same_v<E, zeno::vec2f>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec3i> && std::is_same_v<E, zeno::vec3i>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg1[2]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec3f> && std::is_same_v<E, zeno::vec3f>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg1[2]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec4i> && std::is_same_v<E, zeno::vec4i>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg1[2] && arg1[3] == arg2[3]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec4f> && std::is_same_v<E, zeno::vec4f>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg1[2] && arg1[3] == arg2[3]);
            }
            else
            {
                return false;
            }
        }, lhs, rhs);
    }
}