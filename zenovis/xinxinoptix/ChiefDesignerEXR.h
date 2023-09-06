#pragma once

#include <ImfMultiPartOutputFile.h>
#include <ImfOutputPart.h>
#include <ImfChannelList.h>
#include <ImfHeader.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <half.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <string>

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
    std::vector<float*> pixels, int width, int height, std::vector<std::string> names,
    const char *filepath
 ) {
    int layer_count = names.size();
    using namespace Imf;
    using namespace Imath;
    std::vector<Header> headers(layer_count);
    for (auto l = 0; l < layer_count; l++) {

        // Create the header with the image size
        headers[l] = Header(width, height);
        headers[l].setName(names[l]);

        // Set the display window (region of the image that should be displayed)
        Box2i displayWindow(V2i(0, 0), V2i(width - 1, height - 1));
        headers[l].displayWindow() = displayWindow;
    }
    MultiPartOutputFile multiPartFile(filepath, headers.data(), layer_count);

    // Create the frame buffer and add the R, G, B, A channels
    for (auto l = 0; l < layer_count; l++) {
        OutputPart outputPart(multiPartFile, l);

        std::vector<Rgba> pixelsBuffer(width * height);
        for (int i = 0; i < width * height; i++) {
            pixelsBuffer[i].r = pixels[l][3 * i];
            pixelsBuffer[i].g = pixels[l][3 * i + 1];
            pixelsBuffer[i].b = pixels[l][3 * i + 2];
            pixelsBuffer[i].a = 1;
        }

        size_t xs = 1 * sizeof (Rgba);
        size_t ys = width * sizeof (Rgba);

        FrameBuffer fb;

        fb.insert ("R", Slice (HALF, (char*) &pixelsBuffer[0].r, xs, ys));
        fb.insert ("G", Slice (HALF, (char*) &pixelsBuffer[0].g, xs, ys));
        fb.insert ("B", Slice (HALF, (char*) &pixelsBuffer[0].b, xs, ys));
        fb.insert ("A", Slice (HALF, (char*) &pixelsBuffer[0].a, xs, ys));

        outputPart.setFrameBuffer(fb);
        outputPart.writePixels(height);
    }
}

}
