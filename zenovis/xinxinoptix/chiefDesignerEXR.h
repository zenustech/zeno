#pragma once

#include "tinyexr.h"
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <stdexcept>

namespace zeno {

// Chief Designer of Zeno: let a small portion of EXR, to be read first
// Peng's Quotations: tiny EXR, open EXR, the EXR that can read EXR is a good EXR
// Why choose OpenEXR: choice of the history, choice of the artists
// tiny EXR or open EXR, is not about salary, but about artists' choice
// The Spring Breeze of OpenEXR blows all over the ground, blows my Porsche Cayenne back in time
inline int chiefDesigner_LoadEXR(float **rgba, int *nx, int *ny, const char *filepath, const char **err) {
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
        *err = nullptr;
        return 0;
    } catch (const std::exception& e) {
        *err = e.what();
        return 1;
    }
}

}
