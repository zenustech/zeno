#pragma once

#include <map>

#include "magic_enum.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace EulerAngle 
{

enum struct Measure {
    Degree, Radians
};

enum struct RotationOrder {
    XYZ, XZY,
    YXZ, YZX,
    ZXY, ZYX
};

static std::string MeasureDefaultString() {
    auto name = magic_enum::enum_name(Measure::Radians);
    return std::string(name);
}

static std::string MeasureListString() {
    auto list = magic_enum::enum_names<Measure>();

    std::string result;
    for (auto& ele : list) {
        result += " ";
        result += ele;
    }
    return result;
}

static std::string RotationOrderDefaultString() {
    auto name = magic_enum::enum_name(RotationOrder::YXZ);
    return std::string(name);
}

static std::string RotationOrderListString() {
    auto list = magic_enum::enum_names<RotationOrder>();

    std::string result;
    for (auto& ele : list) {
        result += " ";
        result += ele;
    }
    return result;
}

    inline glm::mat4 rotate(RotationOrder order, Measure measure, const glm::vec3& angleXYZ) {

        auto angles = angleXYZ;

        if ( measure !=  Measure::Radians ) {
            angles[0] = glm::radians(angleXYZ[0]);
            angles[0] = glm::radians(angleXYZ[1]);
            angles[0] = glm::radians(angleXYZ[2]);
        }

        auto orderName = magic_enum::enum_name(order);

        static const std::map<char, uint8_t> kv {
            {'X', 0}, {'Y', 1}, {'Z', 2}
        };

        glm::mat4 rotation = glm::mat4(1.0f);

        for (size_t i=0; i<3; ++i) {
            auto k = orderName[i];

            auto index = kv.at(k);
            auto angle = angles[index];

            auto axis = glm::vec3(0,0,0);
            axis[index] = 1.0f;

            rotation = glm::rotate(rotation, angle, axis);
        }

        return rotation;
    }
}