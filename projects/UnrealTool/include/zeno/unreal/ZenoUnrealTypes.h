#pragma once

#include "Pair.h"
#include "msgpack/msgpack.h"
#include "zeno/utils/vec.h"
#include "zeno/zeno.h"
#include <cstdint>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <vector>

namespace zeno::unreal {

struct AnyNumeric {
    std::string data_;

    AnyNumeric() = default;

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    AnyNumeric(T value) {
        data_ = std::to_string(value);
    }

    inline float data() const {
        return std::stof(data_);
    }

    template <typename T>
    void pack(T& pack) {
        pack(data_);
    }

};

class ZENO_API Mesh {

public:
    Mesh() = default;
    Mesh(const std::vector<zeno::vec3f>& verts, const std::vector<zeno::vec3i>& trigs);

    std::vector<std::array<AnyNumeric, 3>> vertices;
    std::vector<std::array<int32_t, 3>> triangles;

    template <class T>
    void pack(T& pack) {
        pack(vertices, triangles);
    }

};

enum class EParamType : int8_t {
    Invalid = -1,
    Float = 0,
    Integer,
    Max,
};

static EParamType ConvertStringToEParamType(const std::string& str) {
    if (str == "float") {
        return EParamType::Float;
    } else if (str == "int") {
        return EParamType::Integer;
    }

    return EParamType::Invalid;
}

struct SubnetNodeParamList {
    std::map<std::string, int8_t> params;

    template <class T>
    void pack(T& pack) {
        pack(params);
    }

};

struct NodeParamInput {
    std::map<std::string, AnyNumeric> data;

    template <class T>
    void pack(T& pack) {
        pack(data);
    }
};

struct Diff {
    std::set<std::string> data;

    template <class T>
    void pack(T& pack) {
        pack(data);
    }
};

enum class ESubjectType : int16_t {
    Invalid = -1,
    Mesh = 0,
    Num,
};

struct SubjectContainer {
    std::string Name;
    int16_t Type;
    std::vector<uint8_t> Data;

    ESubjectType GetType() const {
        return static_cast<ESubjectType>(Type);
    }

    template <class T>
    void pack(T& pack) {
        pack(Name, Type, Data);
    }
};

struct SubjectContainerList {
    std::vector<SubjectContainer> Data;

    template <class T>
    void pack(T& pack) {
        pack(Data);
    }
};

extern "C" {
    class Mesh;
    struct SubnetNodeParamList;
    struct NodeParamInput;
}

}
