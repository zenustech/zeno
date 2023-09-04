#pragma once

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
    if (channels != 4) throw std::runtime_error("SaveEXR only support RGBA for now");
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
        Rgba* pixelsBuffer = new Rgba[width * height];
        for (int i = 0; i < width * height; i++) {
            pixelsBuffer[i].r = pixels[4 * i];
            pixelsBuffer[i].g = pixels[4 * i + 1];
            pixelsBuffer[i].b = pixels[4 * i + 2];
            pixelsBuffer[i].a = pixels[4 * i + 3];
        }

        // Create the output file
        RgbaOutputFile file(filepath, header);

        // Write the pixels to the file
        file.setFrameBuffer(pixelsBuffer, 1, width);
        file.writePixels(height);

        // Clean up
        delete[] pixelsBuffer;
        return 0;
    } catch (const std::exception& e) {
        *err = strdup(e.what());
        return 1;
    }
}

}
