//
// Created by zh on 2023/7/31.
//
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <stdexcept>
#include <filesystem>
#include <zeno/utils/log.h>
#include <zeno/utils/image_proc.h>
#include <zeno/utils/scope_exit.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <tinygltf/stb_image.h>
#include <zeno/types/HeatmapObject.h>
#include <png.h>
#include <cstdio>
#include <vector>

static std::shared_ptr<zeno::PrimitiveObject> read_png(const char* file_path) {
    auto img = std::make_shared<zeno::PrimitiveObject>();
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        zeno::log_error("Error: File not found: {}", file_path);
        return img;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(file);
        zeno::log_error("Error: png_create_read_struct failed.");
        return img;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        fclose(file);
        zeno::log_error("Error: png_create_info_struct failed.");
        return img;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(file);
        zeno::log_error("Error: Error during png_read_png.");
        return img;
    }

    png_init_io(png_ptr, file);
    png_set_sig_bytes(png_ptr, 0);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, nullptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);

    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int channels = png_get_channels(png_ptr, info_ptr);

    png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);
    std::vector<zeno::vec3f> image_data;
    image_data.reserve(width * height);
    std::vector<float> alpha_data;
    alpha_data.reserve(width * height);

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            zeno::vec3f color;
            float alpha = 1;
            if (bit_depth == 16) {
                if (channels == 1) {
                    uint16_t value = (row[x * 2] << 8) | row[x * 2 + 1];
                    float normalized_value = float(value) / 65535.0f;
                    color = zeno::vec3f(normalized_value);
                } else if (channels == 3) {
                    uint16_t r = (row[x * 6] << 8) | row[x * 6 + 1];
                    uint16_t g = (row[x * 6 + 2] << 8) | row[x * 6 + 3];
                    uint16_t b = (row[x * 6 + 4] << 8) | row[x * 6 + 5];

                    color[0] = float(r) / 65535.0f;
                    color[1] = float(g) / 65535.0f;
                    color[2] = float(b) / 65535.0f;
                } else if (channels == 4) {
                    uint16_t r = (row[x * 8] << 8) | row[x * 8 + 1];
                    uint16_t g = (row[x * 8 + 2] << 8) | row[x * 8 + 3];
                    uint16_t b = (row[x * 8 + 4] << 8) | row[x * 8 + 5];
                    uint16_t a = (row[x * 8 + 6] << 8) | row[x * 8 + 7];

                    color[0] = float(r) / 65535.0f;
                    color[1] = float(g) / 65535.0f;
                    color[2] = float(b) / 65535.0f;
                    alpha = float(a) / 65535.0f;
                }
            } else if (bit_depth == 8) {
            }

            image_data.push_back(color);
            alpha_data.push_back(alpha);
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(file);
    zeno::image_flip_vertical(image_data.data(), width, height);
    img->verts.values = image_data;
    zeno::image_flip_vertical(alpha_data.data(), width, height);
    img->add_attr<float>("alpha") = alpha_data;
    img->userData().set2("isImage", 1);
    img->userData().set2("w", width);
    img->userData().set2("h", height);
    img->userData().set2("bit_depth", bit_depth);
    img->userData().set2("channels", channels);
    return img;
}


namespace zeno {
struct ReadPNG16 : INode {//todo: select custom color space
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        path = std::filesystem::u8path(path).string();
        auto img = read_png(path.c_str());
        if(img->userData().get2<int>("bit_depth") == 8)
        {
            img = zeno::readImageFile(path); 
        }
        set_output("image", img);
    }
};
ZENDEFNODE(ReadPNG16, {
    {
        {"readpath", "path"},
    },
    {
        {"PrimitiveObject", "image"},
    },
    {},
    {"comp"},
});
}