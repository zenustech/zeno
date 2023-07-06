#pragma once

#include <vector>
#include <array>
#include <string>
#include <set>
#include <map>
#include <cassert>

#if defined(__clang__) || _MSC_VER >= 1900
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

// Height data range is [-255.0, 255.0]
// See UE LandscapeDataAccess.h.
#define UE_LANDSCAPE_ZSCALE (1.0f / 128.0f)
#define UE_LANDSCAPE_ZSCALE_INV 128.f

namespace zeno::remote {

enum class ESubjectType : int16_t {
    Invalid = -1,
    Mesh = 0,
    HeightField,
    PointSet,
    Num,
};

template <ESubjectType SubjectType>
struct ZenoSubject {

    constexpr static ESubjectType SubjectType = SubjectType;
    std::map<std::string, std::string> Meta;

    template <typename T>
    void pack(T& pack) {
        pack(Meta);
    }
};

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

struct Mesh : public ZenoSubject<ESubjectType::Mesh> {

    Mesh() = default;
    Mesh(std::vector<std::array<AnyNumeric,3>>&& verts, std::vector<std::array<int32_t, 3>>&& trigs) {
        vertices.swap(verts);
        triangles.swap(trigs);
    }

    std::vector<std::array<AnyNumeric, 3>> vertices;
    std::vector<std::array<int32_t, 3>> triangles;

    /**
     * @return Identify if bdiff meta not found
     */
    inline std::array<float, 4> GetBoundDiff() const {
        auto it = Meta.find("bdiff");
        std::array<float, 4> bdiff {1.f, 1.f, 1.f, 1.f};
        if (it != Meta.end()) {
            sscanf(it->second.c_str(), "%f,%f,%f,%f", &bdiff[0], &bdiff[1], &bdiff[2], &bdiff[3]);
        }
        return bdiff;
    }

    template <class T>
    void pack(T& pack) {
        ZenoSubject::pack(pack);
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

// Get EParamType from std::string
static EParamType GetParamTypeFromString(const std::string& str) {
    if (str == "Float") {
        return EParamType::Float;
    } else if (str == "Integer") {
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

/**
 * @brief SubjectContainer
 * @note  This is a container for subject data, provide Name, Type and Data.
 */
struct SubjectContainer {
    std::string Name;
    int16_t/* ESubjectType */ Type;
    std::vector<uint8_t> Data;

    ESubjectType GetType() const {
        return static_cast<ESubjectType>(Type);
    }

    template <class T>
    void pack(T& pack) {
        pack(Name, Type, Data);
    }
};

/**
 * @brief SubjectContainerList
 * @note  This is a container for SubjectContainer, provide a list of SubjectContainer.
 */
struct SubjectContainerList {
    std::vector<SubjectContainer> Data;

    template <class T>
    void pack(T& pack) {
        pack(Data);
    }
};

/**
 * @brief HeightField
 * @note  This is a container for height field data, provide Nx, Ny, Data and LandscapeScale.
 */
struct HeightField : public ZenoSubject<ESubjectType::HeightField> {
    int32_t Nx = 0, Ny = 0;
    std::vector<std::vector<uint16_t>> Data;
    float LandscapeScaleX = .0f;
    float LandscapeScaleY = .0f;
    float LandscapeScaleZ = .0f;

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
        ZenoSubject::pack(pack);
        pack(Nx, Ny, Data, LandscapeScaleX, LandscapeScaleY, LandscapeScaleZ);
    }
};

struct Dummy : public ZenoSubject<ESubjectType::Invalid> {
    template <class T>
    void pack(T& pack) {
        ZenoSubject::pack(pack);
        pack();
    }
};

/**
 * @brief ParamContainer
 * @note  This is a container for param value, provide type and data.
 */
struct ParamContainer {
    int8_t/* EParamType */ Type;
    std::string Data;

    template <class T>
    void pack(T& pack) {
        pack(Type, Data);
    }
};

template <typename T, uint8_t N>
using TVectorN = std::array<T, N>;
using Vector3f = TVectorN<float, 3>;

/**
 * @brief ParamDescriptor
 * @note  This is a container for param information, provide name and type.
 */
struct ParamDescriptor {
    std::string Name;
    int16_t/* ESubjectType */ Type = 0;

    template <class T>
    void pack(T& pack) {
        pack(Name, Type);
    }
};

/**
 * @brief ParamValue
 * @note  This is a container for param value, it can be either numeric or complex data
 */
struct ParamValue : public ParamDescriptor {
    std::string NumericData;
    std::vector<uint8_t> ComplexData;

    template <class T>
    void pack(T& pack) {
        pack(Name, Type, NumericData, ComplexData);
    }

    template <class T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
    T Cast() const {
        if (Type != static_cast<decltype(Type)>(EParamType::Integer)) {
            throw "ParamValue integer casting runtime check failed.";
        }
        return static_cast<T>(std::stol(NumericData));
    }

    template <class T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
    T Cast() const {
        if (Type != static_cast<decltype(Type)>(EParamType::Float)) {
            throw "ParamValue float casting runtime check failed.";
        }
        return static_cast<T>(std::stod(NumericData));
    }

};

/**
 * @brief The ParamValueBatch struct
 * This is a batch of ParamValue, used for sending multiple parameters at once.
 */
struct ParamValueBatch {
    std::vector<ParamValue> Values;

    template <class T>
    void pack(T& pack) {
        pack(Values);
    }
};

/**
 * @brief The GraphInfo struct
 * This is a struct that contains information about the graph.
 * It is used for sending information about the graph to the client.
 */
struct GraphInfo {
    bool bIsValid = false;
    std::map<std::string, zeno::remote::ParamDescriptor> InputParameters;
    std::map<std::string, zeno::remote::ParamDescriptor> OutputParameters;

    template <class T>
    void pack(T& pack) {
        pack(bIsValid, InputParameters, OutputParameters);
    }
};

/**
 * @brief The GraphRunInfo struct
 * This is a struct that contains information to run a graph.
 */
struct GraphRunInfo {
    ParamValueBatch Values;
    std::string GraphDefinition; // zsl file

    template <class T>
    void pack(T& pack) {
        pack(Values, GraphDefinition);
    }
};

struct PCGPoint {
    Vector3f Position { .0f, .0f, .0f };
    float Density = 1.f;
    // TODO [darc] : support others attributes :

    template <class T>
    void pack(T& pack) {
        pack(Position, Density);
    }
};

struct PointSet : public ZenoSubject<ESubjectType::PointSet> {
    std::vector<PCGPoint> Points;

    template <class T>
    void pack(T& pack) {
        ZenoSubject::pack(pack);
        pack(Points);
    }
};

inline const static std::string NAME_LandscapeInfoSimple = "__Internal_Reserved_LandscapeInfo";
inline const static std::string NAME_SurfaceSample = "__Internal_Reserved_SurfaceSample";

}
