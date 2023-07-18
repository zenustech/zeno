#include "zeno/core/INode.h"
#include "zeno/core/defNode.h"
#include "zeno/utils/logger.h"
#include "zeno/unreal/UnrealTool.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/UserData.h"

// Note: Should we check libpng version in cmake file?
#include "libpng16/png.h"

#define CHECK_PTR(ARG)                                                                                                \
    if (!ARG) {                                                                                                        \
        zeno::log_error("null pointer: " #ARG);                                                                        \
        return;                                                                                                        \
    }

namespace zeno {

uint16_t MapHeightDataF32ToU16(const float Height) {
    constexpr uint16_t uint16Max = std::numeric_limits<uint16_t>::max();
    // LandscapeDataAccess.h:
    // static_cast<uint16>(FMath::RoundToInt(FMath::Clamp<float>(Height *
    // LANDSCAPE_INV_ZSCALE + MidValue, 0.f, MaxValue)))
    auto NewValue = static_cast<uint16_t>(
        std::round(zeno::clamp(Height * UE_LANDSCAPE_ZSCALE_INV + float(0x8000), 0.f, static_cast<float>(uint16Max))));
    return NewValue;
}

float MapHeightDataU16ToF32(const uint16_t Height, const float Scale) {
    // ((float)Height - MidValue) * LANDSCAPE_ZSCALE
    return ((float) Height - 0x8000) * UE_LANDSCAPE_ZSCALE * Scale;
}


void WriteGrayscalePNG(uint32_t Width, uint32_t Height, const void* Data, const char* Filename, uint8_t BitDepth = 16) {
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    CHECK_PTR(png_ptr);

    png_infop info_ptr = png_create_info_struct(png_ptr);
    CHECK_PTR(info_ptr);

    png_bytep* row_pointers = (png_bytep*) png_malloc(png_ptr, Height * sizeof(png_bytep));
    if (setjmp(png_jmpbuf(png_ptr)) != 0) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        png_free(png_ptr, row_pointers);
        return;
    }

    FILE* fd = fopen(Filename, "wb");
    CHECK_PTR(fd);

    // Not compression by default
    png_set_compression_level(png_ptr, 3);
    png_set_IHDR(png_ptr, info_ptr, Width, Height, BitDepth, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_write_fn(png_ptr, (void*) fd, [](png_structp png_ptr, png_bytep data, png_size_t length) {
        FILE* fp = (FILE*) png_get_io_ptr(png_ptr);
        fwrite(data, 1, length, fp);
    }, nullptr);

    const uint64_t PixelChannels = 1;
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

        const auto& heights = prim->verts.attr<float>(channel);

        std::vector<uint16_t> image_data;
        image_data.reserve(heights.size());
        for (const auto height : heights) {
            image_data.push_back(MapHeightDataF32ToU16(height));
        }

        WriteGrayscalePNG(nx, ny, image_data.data(), output_path.c_str(), 16);
    }
};

ZENDEFNODE(
    ExportHeightfieldToImage,
    {
        {
            {"prim"},
            {"string", "channel", "height"},
            {"writepath", "output_path"},
        },
        {
        },
        {},
        {"Landscape"},
    }
);

}
