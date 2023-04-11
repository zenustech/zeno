#pragma once

#include "Pair.h"
#include "zeno/utils/vec.h"
#include "zeno/zeno.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <cstdint>
#include <vector>

namespace zeno::unreal {

struct GenericFieldVisitor {
    explicit GenericFieldVisitor(rapidjson::Document::AllocatorType& allocator);

    template <typename T>
    rapidjson::Value operator()(const std::vector<T>& data) const {
        rapidjson::Value value(rapidjson::kArrayType);

        for (float i : data) {
            value.PushBack(i, allocator);
        }

        return value;
    }

    template <typename ValueType, size_t Size>
    rapidjson::Value operator()(const std::vector<vec<Size, ValueType>>& data) const {
        rapidjson::Value value(rapidjson::kArrayType);

        for (const auto& i : data) {
            value.PushBack(operator()(i), allocator);
        }

        return value;
    }

    template <typename ValueType, size_t Size>
    rapidjson::Value operator()(const vec<Size, ValueType>& data) const {
        rapidjson::Value value(rapidjson::kArrayType);

        for (int i = 0; i < Size; ++i) {
            value.PushBack(data[i], allocator);
        }

        return value;
    }

private:
    rapidjson::Document::AllocatorType& allocator;
};


class Serializable {

protected:
    template <typename T>
    rapidjson::Value SerializeField(const T& field, rapidjson::Document::AllocatorType& allocator) const {
        return GenericFieldVisitor(allocator)(field);
    }

public:
    virtual SimpleCharBuffer Serialize() = 0;
    virtual int8_t Deserialize(const char* json) = 0;
};

class ZENO_API Mesh {

public:
    Mesh() = default;
    Mesh(const std::vector<zeno::vec3f>& verts, const std::vector<zeno::vec3i>& trigs);

    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<int32_t, 3>> triangles;

    template <class T>
    void pack(T& pack) {
        pack(vertices, triangles);
    }

};

extern "C" {
    class Mesh;
}

}
