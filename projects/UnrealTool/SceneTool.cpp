#include "zeno/core/INode.h"
#include "zeno/core/defNode.h"
#include "zeno/types/DictObject.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/logger.h"

#include "zeno/unreal/UnrealTool.h"
#define ZENO_ASSET_TYPE_IMPLEMENTATION 1
#include "zeno/unreal/ZenoAssetTypes.h"
#undef ZENO_ASSET_TYPE_IMPLEMENTATION

// Note: Should we check libpng version in cmake file?
#include "libpng16/png.h"

#include <alpaca/alpaca.h>

#define CHECK_PTR(ARG)                          \
    if (!ARG) {                                 \
        zeno::log_error("null pointer: " #ARG); \
        return;                                 \
    }

#define CHECK_PTR_RET(ARG)                      \
    if (!ARG) {                                 \
        zeno::log_error("null pointer: " #ARG); \
        return {};                              \
    }

namespace zeno {

    /**
     * Mapping height value in range [-255, 255] to [0, 65535]
     * @param Height float value in range [-255, 255]
     * @return uint16_t value in range [0, 65535]
     */
    uint16_t MapHeightDataF32ToU16(const float Height) {
        constexpr uint16_t uint16Max = std::numeric_limits<uint16_t>::max();
        // LandscapeDataAccess.h:
        // static_cast<uint16>(FMath::RoundToInt(FMath::Clamp<float>(Height *
        // LANDSCAPE_INV_ZSCALE + MidValue, 0.f, MaxValue)))
        auto NewValue = static_cast<uint16_t>(
            std::round(zeno::clamp(Height * UE_LANDSCAPE_ZSCALE_INV + float(0x8000), 0.f, static_cast<float>(uint16Max))));
        return NewValue;
    }

    /**
     * Mapping height value in range [0, 65535] to [-255, 255]
     * @param Height uint16_t value in range [0, 65535]
     * @param Scale  Scale factor
     * @return float value in range [-255, 255]
     */
    float MapHeightDataU16ToF32(const uint16_t Height, const float Scale) {
        // ((float)Height - MidValue) * LANDSCAPE_ZSCALE
        return ((float) Height - 0x8000) * UE_LANDSCAPE_ZSCALE * Scale;
    }

    /**
     * Write grayscale png file
     * @param Width Image width
     * @param Height Image height
     * @param Data Image data
     * @param Filename Output filename
     * @param BitDepth Bit depth, default is 16 (only 8 or 16 is supported)
     */
    void WriteGrayscalePNG(uint32_t Width, uint32_t Height, const void *Data, const char *Filename, uint8_t BitDepth = 8, int32_t ColorType = PNG_COLOR_TYPE_GRAY) {
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        CHECK_PTR(png_ptr);

        png_infop info_ptr = png_create_info_struct(png_ptr);
        CHECK_PTR(info_ptr);

        png_bytep *row_pointers = (png_bytep *) png_malloc(png_ptr, Height * sizeof(png_bytep));
        if (setjmp(png_jmpbuf(png_ptr)) != 0) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            png_free(png_ptr, row_pointers);
            return;
        }

        FILE *fd = fopen(Filename, "wb");
        CHECK_PTR(fd);

        // Not compression by default
        png_set_compression_level(png_ptr, 3);
        png_set_IHDR(png_ptr, info_ptr, Width, Height, BitDepth, ColorType, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_set_write_fn(
            png_ptr, (void *) fd, [](png_structp png_ptr, png_bytep data, png_size_t length) {
                FILE *fp = (FILE *) png_get_io_ptr(png_ptr);
                fwrite(data, 1, length, fp);
            },
            nullptr);

        const uint64_t PixelChannels = ColorType == PNG_COLOR_TYPE_GRAY ? 1 : 4;
        const uint64_t BytesPerPixel = BitDepth * PixelChannels / 8;
        const uint64_t BytesPerRow = BytesPerPixel * Width;

        for (int64_t i = 0; i < Height; i++) {
            row_pointers[i] = (png_bytep) Data + i * BytesPerRow;
        }
        png_set_rows(png_ptr, info_ptr, row_pointers);

        uint32_t Transform = PNG_TRANSFORM_IDENTITY;
        // Little endian swap
        // Default is big endian
#if _WIN32 || _WIN64 || __linux__
        if (BitDepth == 16) {
            Transform |= PNG_TRANSFORM_SWAP_ENDIAN;
        }
#endif

        png_write_png(png_ptr, info_ptr, Transform, nullptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        png_free(png_ptr, row_pointers);
        fclose(fd);
    }

    std::optional<zeno::unreal::ByteArray> ToPNGBuffer(uint32_t Width, uint32_t Height, const void *Data, uint8_t BitDepth = 8, int32_t ColorType = PNG_COLOR_TYPE_GRAY) {
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        CHECK_PTR_RET(png_ptr);

        png_infop info_ptr = png_create_info_struct(png_ptr);
        CHECK_PTR_RET(info_ptr);

        png_bytep *row_pointers = (png_bytep *) png_malloc(png_ptr, Height * sizeof(png_bytep));
        if (setjmp(png_jmpbuf(png_ptr)) != 0) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            png_free(png_ptr, row_pointers);
            return {};
        }

        std::vector<uint8_t> buffer;

        // Not compression by default
        png_set_compression_level(png_ptr, 3);
        png_set_IHDR(png_ptr, info_ptr, Width, Height, BitDepth, ColorType, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_set_write_fn(
            png_ptr, (void *) &buffer, [](png_structp png_ptr, png_bytep data, png_size_t length) {
                auto *pBuffer = (std::vector<uint8_t> *) png_get_io_ptr(png_ptr);
                pBuffer->insert(pBuffer->end(), data, data + length);
            },
            nullptr);

        const uint64_t PixelChannels = ColorType == PNG_COLOR_TYPE_GRAY ? 1 : 4;
        const uint64_t BytesPerPixel = BitDepth * PixelChannels / 8;
        const uint64_t BytesPerRow = BytesPerPixel * Width;

        for (int64_t i = 0; i < Height; i++) {
            row_pointers[i] = (png_bytep) Data + i * BytesPerRow;
        }
        png_set_rows(png_ptr, info_ptr, row_pointers);

        uint32_t Transform = PNG_TRANSFORM_IDENTITY;
        // Little endian swap
        // Default is big endian
#if _WIN32 || _WIN64 || __linux__
        if (BitDepth == 16) {
            Transform |= PNG_TRANSFORM_SWAP_ENDIAN;
        }
#endif

        png_write_png(png_ptr, info_ptr, Transform, nullptr);
        png_free(png_ptr, row_pointers);
        png_destroy_write_struct(&png_ptr, &info_ptr);

        return buffer;
    }

    struct AssetWrapperInZeno : public IObject {
        zeno::unreal::Any Data;
    };

    /**
     * Node to export heightfield to image
     */
    struct ExportHeightfieldToImage : public INode {
        void apply() override {
            std::string channel = get_input2<std::string>("channel");
            std::string output_path = get_input2<std::string>("output_path");
            std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("prim");

            if (!prim->verts.has_attr(channel)) {
                zeno::log_error("No such channel '{}' in primitive vertex", channel);
                return;
            }

            int nx = prim->userData().get2<int>("nx"), ny = prim->userData().get2<int>("ny");
            zeno::log_warn("nx={}, ny={}", nx, ny);

            const auto &heights = prim->verts.attr<float>(channel);

            std::vector<uint16_t> image_data;
            image_data.reserve(heights.size());
            for (const auto height: heights) {
                image_data.push_back(MapHeightDataF32ToU16(height));
            }

            WriteGrayscalePNG(nx, ny, image_data.data(), output_path.c_str(), 16);
        }
    };

    struct ExportTextureToImage : public INode {
        void apply() override {
            std::string channel = get_input2<std::string>("channel");
            std::string output_path = get_input2<std::string>("output_path");
            std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("prim");

            if (!prim->verts.has_attr(channel)) {
                zeno::log_error("No such channel '{}' in primitive vertex", channel);
                return;
            }

            int nx = prim->userData().get2<int>("nx"), ny = prim->userData().get2<int>("ny");
            zeno::log_warn("nx={}, ny={}", nx, ny);

            const auto &heights = prim->verts.attr<zeno::vec3f>(channel);

            std::vector<std::array<uint8_t, 4>> image_data;
            image_data.reserve(heights.size());
            for (const auto color: heights) {
                image_data.push_back(std::array<uint8_t, 4>{uint8_t(color[0] * 0xFF), uint8_t(color[1] * 0xFF), uint8_t(color[2] * 0xFF), 0xFF});
            }

            WriteGrayscalePNG(nx, ny, image_data.data(), output_path.c_str(), 8, PNG_COLOR_TYPE_RGBA);
        }
    };

    struct ExtractVertexAttributeToTexture : public INode {
        void apply() override {
            std::string channel = get_input2<std::string>("channel");
            std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("prim");
            std::string type = get_input2<std::string>("type");

            if (!prim || !prim->verts.has_attr(channel)) {
                zeno::log_error("No such channel '{}' in primitive vertex", channel);
                return;
            }

            int nx = prim->userData().get2<int>("nx"), ny = prim->userData().get2<int>("ny");

            std::optional<zeno::unreal::ByteArray> Array;
            if (type == "HeightField") {
                const auto &heights = prim->verts.attr<float>(channel);

                std::vector<uint16_t> image_data;
                image_data.reserve(heights.size());
                for (const auto height: heights) {
                    image_data.push_back(MapHeightDataF32ToU16(height));
                }

                Array = ToPNGBuffer(nx, ny, image_data.data(), 16);
            } else if (type == "vec3f") {
                const auto &heights = prim->verts.attr<zeno::vec3f>(channel);

                std::vector<std::array<uint8_t, 4>> image_data;
                image_data.reserve(heights.size());
                for (const auto color: heights) {
                    image_data.push_back(std::array<uint8_t, 4>{uint8_t(color[0] * 0xFF), uint8_t(color[1] * 0xFF), uint8_t(color[2] * 0xFF), 0xFF});
                }

                Array = ToPNGBuffer(nx, ny, image_data.data(), 8, PNG_COLOR_TYPE_RGBA);
            }

            if (!Array.has_value()) {
                zeno::log_error("Failed to convert HeightField to PNG buffer");
                return;
            }
            auto PngPtr = std::make_shared<AssetWrapperInZeno>();
            auto &PngBuffer = std::get<zeno::unreal::PngTextureData>(PngPtr->Data = zeno::unreal::PngTextureData{});
            PngBuffer.Buffer = std::move(Array.value());
            PngBuffer.Width = nx;
            PngBuffer.Height = ny;
            set_output("pngTexture", PngPtr);
        }
    };

    struct ExtractVertexHeightField : public INode {
        void apply() override {
            std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("prim");
            const std::string channel = get_input2<std::string>("HeightChannel");
            const std::string nx_channel = get_input2<std::string>("NxUserDataChannel");
            const std::string ny_channel = get_input2<std::string>("NyUserDataChannel");

            const auto &heights = prim->verts.attr<float>(channel);

            std::vector<uint16_t> image_data;
            image_data.reserve(heights.size());
            for (const auto height: heights) {
                image_data.push_back(MapHeightDataF32ToU16(height));
            }

            auto WrapperPtr = std::make_shared<AssetWrapperInZeno>();
            auto &HeightField = std::get<zeno::unreal::LandscapeData>(WrapperPtr->Data = zeno::unreal::LandscapeData{});
            HeightField.Width = prim->userData().get2<int>(nx_channel);
            HeightField.Height = prim->userData().get2<int>(ny_channel);
            HeightField.HeightField = std::move(image_data);

            set_output("LandscapeData", WrapperPtr);
        }
    };

    struct SetLandscapeDataBaseColorTexture : public INode {
        void apply() override {
            std::shared_ptr<AssetWrapperInZeno> WrapperPtr = get_input2<AssetWrapperInZeno>("LandscapeData");
            std::string guid = get_input2<std::string>("guidRef");

            if (!std::holds_alternative<zeno::unreal::LandscapeData>(WrapperPtr->Data)) {
                zeno::log_error("Input LandscapeData is not LandscapeData");
                return;
            }

            auto &LandscapeData = std::get<zeno::unreal::LandscapeData>(WrapperPtr->Data);
            LandscapeData.BaseColorTextureRef.Guid = guid;

            set_output("LandscapeData", WrapperPtr);
        }
    };

    struct CreateAssetBundle : public INode {
        void apply() override {
            std::shared_ptr<DictObject> dict = get_input2<DictObject>("AssetData");
            std::map<std::string, std::shared_ptr<AssetWrapperInZeno>> asset_map = dict->get<AssetWrapperInZeno>();

            std::shared_ptr<AssetWrapperInZeno> wrapper = std::make_shared<AssetWrapperInZeno>();
            zeno::unreal::AssetBundle &Bundle = std::get<zeno::unreal::AssetBundle>(wrapper->Data = zeno::unreal::AssetBundle{});
            for (const std::pair<std::string, std::shared_ptr<AssetWrapperInZeno>> &pair: asset_map) {
                if (pair.second) {
                    // Bundle.Push(pair.second->Data);
                    Bundle.Assets[pair.first] = pair.second->Data;
                }
            }

            set_output("Bundle", wrapper);
        }
    };

    struct PushAssetToBundle : public INode {
        void apply() override {
            std::shared_ptr<AssetWrapperInZeno> BundleWrapper = get_input2<AssetWrapperInZeno>("Bundle");
            std::shared_ptr<AssetWrapperInZeno> AssetWrapper = get_input2<AssetWrapperInZeno>("Asset");

            if (!std::holds_alternative<zeno::unreal::AssetBundle>(BundleWrapper->Data)) {
                zeno::log_error("Input Bundle is not AssetBundle");
                return;
            }

            zeno::unreal::AssetBundle &Bundle = std::get<zeno::unreal::AssetBundle>(BundleWrapper->Data);
            std::string guid = Bundle.Push(AssetWrapper->Data);

            set_output("Bundle", BundleWrapper);
            set_output2("GUID", guid);
        }
    };

    struct CreatePointSet : public INode {
        void apply() override {
            std::string PointType = get_input2<std::string>("PointType");
            std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("Prim");
            auto &posList = prim->verts;
            auto& scaleList = prim->verts.attr<float>("rad");

            zeno::unreal::PointSet PointSet;
            PointSet.Points.reserve(posList.size());
            for (size_t i = 0; i < posList.size(); ++i) {
                // TODO [darc] : add other :
                const vec3f& pos = posList[i];
                float scale = scaleList[i];
                PointSet.Points.push_back( { pos, { 1.0f }, { scale, scale, scale } });
            }
            if (PointType == "Misc") PointSet.PointType = zeno::unreal::PointSet::Type::Misc;
            else if (PointType == "Tree")
                PointSet.PointType = zeno::unreal::PointSet::Type::Tree;
            else if (PointType == "Grass")
                PointSet.PointType = zeno::unreal::PointSet::Type::Grass;

            std::shared_ptr<AssetWrapperInZeno> WrapperPtr = std::make_shared<AssetWrapperInZeno>();
            WrapperPtr->Data = std::move(PointSet);
            set_output("PointSet", WrapperPtr);
        }
    };

    struct SaveAssetBundle : public INode {
        void apply() override {
            std::shared_ptr<AssetWrapperInZeno> asset_bundle = get_input<AssetWrapperInZeno>("asset_bundle");
            std::string output_path = get_input2<std::string>("output_path");

            if (!asset_bundle) {
                zeno::log_error("Asset bundle to save is NULL");
                return;
            }

            if (!std::holds_alternative<zeno::unreal::AssetBundle>(asset_bundle->Data)) {
                zeno::log_error("Asset bundle to save is not AssetBundle");
                return;
            }

            zeno::unreal::ByteArray Buffer;
            auto &Bundle = std::get<zeno::unreal::AssetBundle>(asset_bundle->Data);
            alpaca::serialize<alpaca::options::with_checksum>(Bundle, Buffer);

            // Save to file
            FILE *fd = fopen(output_path.c_str(), "wb");
            CHECK_PTR(fd);
            fwrite(Buffer.data(), 1, Buffer.size(), fd);
            fclose(fd);
        }
    };

    struct ResampleToUnrealLandscape : public INode {

        template<typename T>
        using V = std::vector<T>;

        void apply() override {
            std::shared_ptr<PrimitiveObject> OldPrim = get_input2<PrimitiveObject>("Prim");
            std::shared_ptr<zeno::PrimitiveObject> Prim =
                std::make_shared<zeno::PrimitiveObject>();
            zeno::vec3f Scale = get_input2<zeno::vec3f>("Scale");
            const std::string channel = get_input2<std::string>("HeightChannel");
            const std::string nx_channel = get_input2<std::string>("NxUserDataChannel");
            const std::string ny_channel = get_input2<std::string>("NyUserDataChannel");

            const auto &heights = OldPrim->verts.attr<float>(channel);

            std::vector<uint16_t> image_data;
            image_data.reserve(heights.size());
            for (const auto height: heights) {
                image_data.push_back(MapHeightDataF32ToU16(height));
            }

            auto Nx = OldPrim->userData().get2<int32_t>(nx_channel);
            auto Ny = OldPrim->userData().get2<int32_t>(ny_channel);

            const float centerX = (float) Nx * Scale[0] * 0.5f;
            const float centerY = (float) Ny * Scale[1] * 0.5f;

            Prim->verts.reserve(Nx * Ny);
            for (int y = 0; y < Ny; y++) {
                for (int x = 0; x < Nx; x++) {
                    Prim->verts.push_back(
                        {(float) x * Scale[0] - centerX, 0.0f, (float) y * Scale[1] - centerY});
                }
            }

            const auto& VertAttrs = OldPrim->verts.attr_keys();
            for (const auto& Attr: VertAttrs) {
                if (Attr == channel || Attr == "pos") continue;
                if (std::holds_alternative<V<vec3f>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<vec3f>(Attr);
                }
                else if (std::holds_alternative<V<vec2f>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<vec2f>(Attr);
                }
                else if (std::holds_alternative<V<vec4f>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<vec4f>(Attr);
                }
                else if (std::holds_alternative<V<float>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<float>(Attr);
                }
                else if (std::holds_alternative<V<int>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<int>(Attr);
                }
                else if (std::holds_alternative<V<vec2i>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<vec2i>(Attr);
                }
                else if (std::holds_alternative<V<vec3i>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<vec3i>(Attr);
                }
                else if (std::holds_alternative<V<vec4i>>(OldPrim->verts.attr(Attr))) {
                    Prim->verts.add_attr<vec4i>(Attr);
                }
            }

#pragma omp parallel for
            for (int32_t i = 0; i < Nx * Ny; i++) {
                for (const auto& Attr: VertAttrs) {
                    if (Attr == channel || Attr == "pos") continue;
                    if (std::holds_alternative<V<vec3f>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<vec3f>(Attr)[i] = OldPrim->verts.attr<vec3f>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<vec2f>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<vec2f>(Attr)[i] = OldPrim->verts.attr<vec2f>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<vec4f>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<vec4f>(Attr)[i] = OldPrim->verts.attr<vec4f>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<float>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<float>(Attr)[i] = OldPrim->verts.attr<float>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<int>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<int>(Attr)[i] = OldPrim->verts.attr<int>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<vec2i>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<vec2i>(Attr)[i] = OldPrim->verts.attr<vec2i>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<vec3i>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<vec3i>(Attr)[i] = OldPrim->verts.attr<vec3i>(Attr)[i];
                    }
                    else if (std::holds_alternative<V<vec4i>>(OldPrim->verts.attr(Attr))) {
                        Prim->verts.attr<vec4i>(Attr)[i] = OldPrim->verts.attr<vec4i>(Attr)[i];
                    }
                }
            }

            Prim->tris.reserve((Nx - 1) * (Ny - 1) * 6);
            for (int y = 0; y < Ny - 1; y++) {
                for (int x = 0; x < Nx - 1; x++) {
                    Prim->tris.push_back({y * Nx + x, y * Nx + x + 1, (y + 1) * Nx + x});
                    Prim->tris.push_back(
                        {y * Nx + x + 1, (y + 1) * Nx + x + 1, (y + 1) * Nx + x});
                }
            }

            auto &Arr = Prim->verts.add_attr<float>("height");
            size_t Idx = 0;
            for (const auto Height: image_data) {
                Arr[Idx] = ((float) Height - 0x8000) * UE_LANDSCAPE_ZSCALE *
                           Scale[2];// ((float)Height - MidValue) * LANDSCAPE_ZSCALE
                Prim->verts[Idx] = {Prim->verts[Idx].at(0), Arr[Idx],
                                    Prim->verts[Idx].at(2)};
                Idx++;
            }

            Prim->userData() = OldPrim->userData();

            set_output("Prim", Prim);
        }
    };

    /** static init block */
    namespace {
        ZENDEFNODE(
            ExportHeightfieldToImage,
            {
                {
                    {"prim"},
                    {"string", "channel", "height"},
                    {"writepath", "output_path"},
                },
                {},
                {},
                {"Landscape"},
            });

        ZENDEFNODE(
            ExportTextureToImage,
            {
                {
                    {"prim"},
                    {"string", "channel", "clr"},
                    {"writepath", "output_path"},
                },
                {},
                {},
                {"Landscape"},
            });

        ZENDEFNODE(
            ExtractVertexHeightField,
            {
                {
                    {"prim"},
                    {"string", "HeightChannel", "height"},
                    {"string", "NxUserDataChannel", "nx"},
                    {"string", "NyUserDataChannel", "ny"},
                },
                {{"LandscapeData"}},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            ExtractVertexAttributeToTexture,
            {
                {
                    {"prim"},
                    {"string", "channel", "clr"},
                    {"enum HeightField vec3f", "type", "vec3f"},
                },
                {{"pngTexture"}},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            SetLandscapeDataBaseColorTexture,
            {
                {
                    {"LandscapeData"},
                    {"string", "guidRef"},
                },
                {{"LandscapeData"}},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            CreateAssetBundle,
            {
                {
                    {"dict", "AssetData"},
                },
                {{"Bundle"}},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            PushAssetToBundle,
            {
                {
                    {"Bundle"},
                    {"Asset"},
                },
                {{"Bundle"},
                 {"string", "GUID"}},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            CreatePointSet,
            {
                {
                    {"Prim"},
                    {"enum Misc Tree Grass", "PointType", "Misc"},
                },
                {"PointSet"},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            SaveAssetBundle,
            {
                {
                    {"asset_bundle"},
                    {"writepath", "output_path"},
                },
                {},
                {},
                {"Unreal"},
            });

        ZENDEFNODE(
            ResampleToUnrealLandscape,
            {
                {
                    {"Prim"},
                    {"vec3f", "Scale"},
                    {"string", "HeightChannel", "height"},
                    {"string", "NxUserDataChannel", "nx"},
                    {"string", "NyUserDataChannel", "ny"},
                },
                {
                    {"Prim"},
                },
                {},
                {"Unreal"},
            });

    }// namespace

}// namespace zeno
