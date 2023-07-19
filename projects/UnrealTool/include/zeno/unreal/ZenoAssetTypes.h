#pragma once

#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <type_traits>
#include <tuple>

namespace zeno::unreal {

    template<size_t T>
#if 1
    using TVectorFloat = std::array<float, T>;
#else
    struct TVectorFloat {
        float Data[T];

        float& operator[](size_t Index) {
            return Data[Index];
        }

        NOP_STRUCTURE(TVectorFloat, Data);
    };
#endif

    using Vector2f = TVectorFloat<2>;
    using Vector3f = TVectorFloat<3>;

    using ByteArray = std::vector<uint8_t>;

    using Any =
        std::variant<
            bool,
            std::string,
            size_t,
            std::make_signed<size_t>::type,
            Vector2f,
            Vector3f,
            ByteArray,
            struct LandscapeData,
            struct AssetBundle
        >
    ;

    ////////////////////////////////////////////////////////////
    /// Component Types
    ////////////////////////////////////////////////////////////

    /**
     * Contains the transform of an entity
     */
    struct TransformComponent {
        Vector3f Position;
        Vector3f Rotation;
        Vector3f Scale;
    };

    ////////////////////////////////////////////////////////////
    /// Asset Data Types
    ////////////////////////////////////////////////////////////
    struct LandscapeData {
        std::vector<uint16_t> HeightField;
        TransformComponent Transform;
    };

    struct AssetBundle {
        std::map<std::string, Any> Assets;
    };

}// namespace zeno::unreal
