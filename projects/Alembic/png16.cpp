//
// Created by zh on 2023/7/31.
//
#include <ImfMultiPartOutputFile.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfHeader.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <half.h>

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <stdexcept>
#include <filesystem>
#include <zeno/utils/log.h>
#include <zeno/utils/image_proc.h>
#include <zeno/utils/string.h>
#include "zeno/types/DictObject.h"
#include <zeno/utils/fileio.h>

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
                if (channels == 1) {
                    uint8_t value = row[x];
                    float normalized_value = float(value) / 255.0f;
                    color = zeno::vec3f(normalized_value);
                } else if (channels == 3) {
                    uint8_t r = row[x * 3];
                    uint8_t g = row[x * 3 + 1];
                    uint8_t b = row[x * 3 + 2];

                    color[0] = float(r) / 255.0f;
                    color[1] = float(g) / 255.0f;
                    color[2] = float(b) / 255.0f;
                } else if (channels == 4) {
                    uint8_t r = row[x * 4];
                    uint8_t g = row[x * 4 + 1];
                    uint8_t b = row[x * 4 + 2];
                    uint8_t a = row[x * 4 + 3];

                    color[0] = float(r) / 255.0f;
                    color[1] = float(g) / 255.0f;
                    color[2] = float(b) / 255.0f;
                    alpha = float(a) / 255.0f;
                }
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

static void SaveMultiLayerEXR(
        std::vector<float*> pixels
        , int width
        , int height
        , std::vector<std::string> channels
        , const char* exrFilePath
) {
    using namespace Imath;
    using namespace Imf;

    Header header(width, height);
    ChannelList channelList;

    const char *std_suffix = "RGB";
    for (auto channel: channels) {
        for (int i = 0; i < 3; i++) {
            std::string name = zeno::format("{}{}", channel, std_suffix[i]);
            channelList.insert(name, Channel(HALF));
        }
    }

    header.channels() = channelList;

    OutputFile file (exrFilePath, header);
    FrameBuffer frameBuffer;

    std::vector<std::vector<half>> data;
    for (float *rgb: pixels) {
        std::vector<half> r(width * height);
        std::vector<half> g(width * height);
        std::vector<half> b(width * height);
        for (auto i = 0; i < width * height; i++) {
            r[i] = rgb[3 * i + 0];
            g[i] = rgb[3 * i + 1];
            b[i] = rgb[3 * i + 2];
        }
        zeno::image_flip_vertical(r.data(), width, height);
        zeno::image_flip_vertical(g.data(), width, height);
        zeno::image_flip_vertical(b.data(), width, height);
        data.push_back(std::move(r));
        data.push_back(std::move(g));
        data.push_back(std::move(b));
    }

    for (auto i = 0; i < channels.size(); i++) {
        for (auto j = 0; j < 3; j++) {
            std::string name = zeno::format("{}{}", channels[i], std_suffix[j]);
            frameBuffer.insert (name, Slice ( HALF, (char*) data[i * 3 + j].data(), sizeof (half) * 1, sizeof (half) * width));
        }
    }

    file.setFrameBuffer (frameBuffer);
    file.writePixels (height);
}
struct WriteExr : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        create_directories_when_write_file(path);
        std::vector<float*> pixels;
        std::vector<std::string> channels;
        std::vector<std::pair<int, int>> width_heights;
        auto dict = get_input<DictObject>("channels");
        for (auto &[k, v]: dict->lut) {
            if (k.size()) {
                channels.push_back(k+'.');
            }
            else {
                channels.push_back(k);
            }
            auto img = std::dynamic_pointer_cast<PrimitiveObject>(v);
            if (!img || !img->userData().get2<int>("isImage", 0)) {
                throw zeno::makeError("input not image");
            }
            int w = img->userData().get2<int>("w");
            int h = img->userData().get2<int>("h");
            width_heights.emplace_back(w, h);
            pixels.push_back(reinterpret_cast<float *>(img->verts.data()));
        }
        for (auto i = 1; i < width_heights.size(); i++) {
            if (width_heights[i] != width_heights[0]) {
                throw zeno::makeError("input image as different size!");
            }
        }
        SaveMultiLayerEXR(pixels, width_heights[0].first, width_heights[0].second, channels, path.c_str());
    }
};
ZENDEFNODE(WriteExr, {
    {
        {"writepath", "path"},
        {"dict", "channels"},
    },
    {
    },
    {},
    {"comp"},
});
struct ReadExr : INode {
    std::pair<std::string, int> get_output_name(std::string name) {
        std::string output_name;
        int channel = 0;

        if (zeno::ends_with(name, ".R")) {
            output_name = name.substr(0, name.size() - 2);
            channel = 0;
        }
        else if (zeno::ends_with(name, ".G")) {
            output_name = name.substr(0, name.size() - 2);
            channel = 1;
        }
        else if (zeno::ends_with(name, ".B")) {
            output_name = name.substr(0, name.size() - 2);
            channel = 2;
        }
        else if (name == "R") {
            output_name = "";
            channel = 0;
        }
        else if (name == "G") {
            output_name = "";
            channel = 1;
        }
        else if (name == "B") {
            output_name = "";
            channel = 2;
        }
        else {
            output_name = name;
            channel = 0;
        }
        return {output_name, channel};

    }
    virtual void apply() override {
        using namespace Imf;
        using namespace Imath;
        auto path = get_input2<std::string>("path");
        // Open the EXR file
        InputFile exrFile(path.c_str());

        // Get the header information
        const Header& exrHeader = exrFile.header();

        // Get the data window size
        Box2i dataWindow = exrHeader.dataWindow();
        int width = dataWindow.max.x - dataWindow.min.x + 1;
        int height = dataWindow.max.y - dataWindow.min.y + 1;

        // Get the channel names
        std::vector<std::string> channelNames;
        const ChannelList& channels = exrHeader.channels();
        std::map<std::string, std::shared_ptr<PrimitiveObject>> lut;
        for (auto it = channels.begin(); it != channels.end(); it++) {
            channelNames.emplace_back(it.name());
            std::string name = it.name();
            auto [output_name, _] = get_output_name(name);
            if (lut.count(output_name) == 0) {
                auto img = std::make_shared<zeno::PrimitiveObject>();
                img->verts.resize(width * height);
                img->userData().set2("isImage", 1);
                img->userData().set2("w", width);
                img->userData().set2("h", height);

                lut[output_name] = img;
            }
        }

        // Read each channel and store the pixel data in a vector
        std::vector<std::vector<float>> pixelData(channelNames.size());
        FrameBuffer frameBuffer;
        for (size_t i = 0; i < channelNames.size(); ++i) {
            pixelData[i].resize(width * height);

            // Define a frame buffer for the current channel
            frameBuffer.insert(channelNames[i].c_str(),
                               Slice(FLOAT,               // Data type
                                     (char*)&pixelData[i][0], // Pointer to data
                                     sizeof(float) * 1,      // Stride (1 channel)
                                     sizeof(float) * width)); // xStride
        }
        // Read the pixel data
        exrFile.setFrameBuffer(frameBuffer);
        exrFile.readPixels(dataWindow.min.y, dataWindow.max.y);
        for (size_t k = 0; k < channelNames.size(); ++k) {
            auto name = channelNames[k];
            auto [output_name, c] = get_output_name(name);
            auto img = lut[output_name];
            for (auto j = 0; j < height; j++) {
                for (auto i = 0; i < width; i++) {
                    auto index = j * width + i;
                    img->verts[index][c] = pixelData[k][index];
                }
            }
        }

        auto dict = std::make_shared<zeno::DictObject>();
        for (auto [k, v]: lut) {
            dict->lut[k] = std::dynamic_pointer_cast<IObject>(v);
        }
        set_output("channels", std::move(dict));
    }
};
ZENDEFNODE(ReadExr, {
    {
        {"readpath", "path"},
    },
    {
        {"dict", "channels"},
    },
    {},
    {"comp"},
});
}
