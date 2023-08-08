#pragma once

#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <type_traits>
#include <tuple>

namespace zeno::unreal {

    ////////////////////////////////////////////////////////////
    /// Primitive Types (not that primitive, just simple data types)
    ////////////////////////////////////////////////////////////
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
            struct AssetBundle,
            struct PngTextureData,
            struct PointSet
        >
    ;

    enum class EAssetFlag : uint32_t {
        None = 0,
        /** If a data not only used by another data, it is standalone */
        Standalone = 1u << 0,
    };

    ////////////////////////////////////////////////////////////
    /// Component Types
    ////////////////////////////////////////////////////////////

    /**
     * Contains the transform of an entity
     */
    struct TransformComponent {
        Vector3f Position = {0.0f};
        Vector3f Rotation = {0.0f};
        Vector3f Scale = {1.0f};
    };

    ////////////////////////////////////////////////////////////
    /// Asset Data Types
    ////////////////////////////////////////////////////////////
    /**
     * Allows reference to a data in the same asset bundle
     */
    struct SoftDataReference {
        /** Used to locate asset in a bundle */
        std::string Guid;

        template <typename T>
        T* GetChecked(struct AssetBundle& InBundle);
    };

    struct LandscapeData {
        uint32_t Width = 0;
        uint32_t Height = 0;
        std::vector<uint16_t> HeightField;
        TransformComponent Transform;
        SoftDataReference BaseColorTextureRef;

        uint32_t Flags = static_cast<uint32_t>(EAssetFlag::Standalone);
    };

    struct PngTextureData {
        std::vector<uint8_t> Buffer;
        uint32_t Width;
        uint32_t Height;

        uint32_t Flags = static_cast<uint32_t>(EAssetFlag::None);
    };

    struct PointSet {
        enum class Type {
            Misc,
            Tree,
            Grass,
        };
        Type PointType;
        std::vector<TransformComponent> Points;
    };

    struct AssetBundle {
        std::map<std::string, Any> Assets;

        uint32_t Flags = static_cast<uint32_t>(EAssetFlag::None);

        std::string Push(const Any& InData);
    };


#if ZENO_ASSET_TYPE_IMPLEMENTATION
    std::string AssetBundle::Push(const Any& InData) {
        std::string Guid = std::to_string(Assets.size());
        Assets.emplace(Guid, InData);
        return Guid;
    }
    template<typename T>
    T* SoftDataReference::GetChecked(AssetBundle& InBundle) {
        auto& Asset = InBundle.Assets[Guid];
        if (auto* Ptr = std::get_if<T>(&Asset)) {
            return Ptr;
        }
        return nullptr;
    }
#endif
}// namespace zeno::unreal
