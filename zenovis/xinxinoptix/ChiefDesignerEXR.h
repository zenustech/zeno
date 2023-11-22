#pragma once

#include <ImfMultiPartOutputFile.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfHeader.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <half.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <string>
#include "zeno/utils/image_proc.h"

namespace zeno::ChiefDesignerEXR {

// Chief Designer of Zeno: let a small portion of EXR, to be read first
// Peng's Quotations: tiny EXR, open EXR, the EXR that can read EXR is a good EXR
// Why choose OpenEXR: choice of the history, choice of the artists
// tiny EXR or open EXR, is not about salary, but about artists' choice
// The Spring Breeze of OpenEXR blows all over the ground, blows my Porsche Cayenne back in time
inline int LoadEXR(float **rgba, int *nx, int *ny, const char *filepath, const char **err) {
    using namespace Imf;
    using namespace Imath;
    try {
        RgbaInputFile file(filepath);
        const Header& header = file.header();
        Box2i dataWindow = header.dataWindow();
        int width = dataWindow.max.x - dataWindow.min.x + 1;
        int height = dataWindow.max.y - dataWindow.min.y + 1;
        float *p = (float *)malloc(width * height * 4 * sizeof(float));
        Array2D<Rgba> pixels;
        pixels.resizeErase(height, width);
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.readPixels(dataWindow.min.y, dataWindow.max.y);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Rgba pixel = pixels[y][x];
                p[(y * width + x) * 4 + 0] = pixel.r;
                p[(y * width + x) * 4 + 1] = pixel.g;
                p[(y * width + x) * 4 + 2] = pixel.b;
                p[(y * width + x) * 4 + 3] = pixel.a;
            }
        }
        *rgba = p;
        *nx = width;
        *ny = height;
        return 0;
    } catch (const std::exception& e) {
        *err = strdup(e.what());
        return 1;
    }
}

inline void FreeEXRErrorMessage(const char *err) {
    free((char *)err);
}

inline int SaveEXR(float *pixels, int width, int height, int channels,
                   int asfp16, const char *filepath, const char **err) {
    if (asfp16 != 1) throw std::runtime_error("SaveEXR only support FP16 for now");
    try {
        using namespace Imf;
        using namespace Imath;

        // Create the header with the image size
        Header header(width, height);

        // Set the display window (region of the image that should be displayed)
        Box2i displayWindow(V2i(0, 0), V2i(width - 1, height - 1));
        header.displayWindow() = displayWindow;

        // Create the frame buffer and add the R, G, B, A channels
        std::vector<Rgba> pixelsBuffer(width * height);
        if (channels == 4) {
            for (int i = 0; i < width * height; i++) {
                pixelsBuffer[i].r = pixels[4 * i];
                pixelsBuffer[i].g = pixels[4 * i + 1];
                pixelsBuffer[i].b = pixels[4 * i + 2];
                pixelsBuffer[i].a = pixels[4 * i + 3];
            }
        }
        else if (channels == 3) {
            for (int i = 0; i < width * height; i++) {
                pixelsBuffer[i].r = pixels[3 * i];
                pixelsBuffer[i].g = pixels[3 * i + 1];
                pixelsBuffer[i].b = pixels[3 * i + 2];
                pixelsBuffer[i].a = 1;
            }
        }
        else {
            throw std::runtime_error("SaveEXR only support RGBA and RGB for now");
        }

        // Create the output file
        RgbaOutputFile file(filepath, header);

        // Write the pixels to the file
        file.setFrameBuffer(pixelsBuffer.data(), 1, width);
        file.writePixels(height);

        return 0;
    } catch (const std::exception& e) {
        *err = strdup(e.what());
        return 1;
    }
}
inline void SaveMultiLayerEXR(
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

}
