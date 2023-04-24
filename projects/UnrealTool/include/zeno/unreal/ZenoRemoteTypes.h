#pragma once

#include <vector>
#include <array>
#include <string>
#include <set>
#include <cassert>

namespace zeno::remote {

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

struct Mesh {

    Mesh() = default;
    Mesh(std::vector<std::array<AnyNumeric,3>>&& verts, std::vector<std::array<int32_t, 3>>&& trigs) {
        vertices.swap(verts);
        triangles.swap(trigs);
    }

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

struct Diff {
    std::vector<std::string> data;
    int32_t CurrentHistory;

    template <class T>
    void pack(T& pack) {
        pack(data);
    }
};

enum class ESubjectType : int16_t {
    Invalid = -1,
    Mesh = 0,
    HeightField,
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

struct HeightField {
    int32_t Nx = 0, Ny = 0;
    std::vector<std::vector<uint16_t>> Data;

    HeightField() = default;

    HeightField(int32_t InNx, int32_t InNy, const std::vector<uint16_t>& InData)
        : Nx(InNx)
          , Ny(InNy)
    {
        assert(Nx * Ny == InData.size());
        Data.resize(Ny);
        for (std::vector<uint16_t>& Vy : Data) { Vy.resize(Nx); }
        for (size_t Y = 0; Y < Ny; ++Y) {
            for (size_t X = 0; X < Nx; ++X) {
                const size_t Idx = Y * Ny + X;
                Data[Y][X] = InData[Idx];
            }
        }
    }

    std::vector<uint16_t> ToFlat() const {
        std::vector<uint16_t> Result;
        Result.reserve(Nx * Ny);
        for (const std::vector<uint16_t>& Ry : Data) {
            for (uint16_t Rx : Ry) {
                Result.push_back(Rx);
            }
        }

        return Result;
    }

    template <class T>
    void pack(T& pack) {
        pack(Nx, Ny, Data);
    }
};

struct Dummy {
    template <class T>
    void pack(T& pack) {
        pack();
    }
};

}
