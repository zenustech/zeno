#pragma once

#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace zeno::unreal {

    template<size_t T>
    using TVectorFloat = std::array<float, T>;

    using Vector2f = TVectorFloat<2>;
    using Vector3f = TVectorFloat<3>;

    using Any =
        std::variant<
            bool,
            std::string,
            size_t,
            std::make_signed<size_t>::type,
            Vector2f,
            Vector3f,
            struct LandscapeData
        >
    ;

    struct LandscapeData {
        Vector3f Scale;
        std::vector<uint16_t> HeightField;
    };

    struct AssetBundle {
        std::map<std::string, Any> Assets;
    };

}// namespace zeno::unreal
