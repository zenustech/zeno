//
// Created by zh on 2023/4/21.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <cstring>
#include <zeno/utils/log.h>
#include <tiffio.h>
#include <stdio.h>
#include "imgcv.h"

namespace zeno {
//std::shared_ptr<PrimitiveObject> readTiffFile(std::string const &path) {
//    TIFF* tiff = TIFFOpen(path.c_str(), "r");
//    if (!tiff) {
//        printf("Failed to open TIFF file!\n");
//    }
//    scope_exit delTiff = [=] { TIFFClose(tiff); };
//
//    uint32 width, height;
//    uint16 bitsPerSample, samplesPerPixel;
//    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
//    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
//    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
//    TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
//
//    uint32* raster = (uint32*)_TIFFmalloc(width * height * sizeof(uint32));
//    scope_exit delRaster = [=] { _TIFFfree(raster); };
//    if (!raster) {
//        printf("Failed to allocate memory for image!\n");
//    }
//    auto img = std::make_shared<PrimitiveObject>();
//    uint32* rasterPtr = raster;
//    for (uint32 row = 0; row < height; row++) {
//        TIFFReadScanline(tiff, rasterPtr, row, 0);
//        rasterPtr += width;
//    }
//
////    img->userData().set2("isImage", 1);
//    img->userData().set2("w", (int)width);
//    img->userData().set2("h", (int)height);
//    img->userData().set2("bitsPerSample", (int)bitsPerSample);
//    img->userData().set2("samplesPerPixel", (int)samplesPerPixel);
//
//    // Process image data here...
//    return img;
//}
std::shared_ptr<PrimitiveObject> readTiffFile(std::string const &path, int type = 0) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) {
        throw std::runtime_error("tiff read fail");
    }
    auto img = std::make_shared<PrimitiveObject>();

    uint32 width, height;
    uint16_t samplesPerPixel, bitsPerSample;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    img->userData().set2("isImage", 1);
    img->userData().set2("w", (int)width);
    img->userData().set2("h", (int)height);
    img->userData().set2("samplesPerPixel", (int)samplesPerPixel);
    img->userData().set2("bitsPerSample", (int)bitsPerSample);
    if (bitsPerSample != 32) {
        throw std::runtime_error("tiff read fail");
    }

    std::vector<uint8_t> data_;
    uint32_t rowSize = TIFFScanlineSize(tif);
    img->userData().set2("rowSize", (int)rowSize);
    data_.resize(rowSize * height);

    for (int32_t row = 0; row < height; ++row) {
        TIFFReadScanline(tif, &data_[row * rowSize], row);
    }
    TIFFClose(tif);

    img->resize(width * height);
    if (samplesPerPixel == 4) {
        vec4f *ptr = (vec4f*)data_.data();
        auto &alpha = img->verts.add_attr<float>("alpha");
        auto &uv = img->verts.add_attr<zeno::vec2f>("uv");
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                vec4f rgba = ptr[i * width + j];
                img->verts[i * width + j] =  { rgba[0], rgba[1], rgba[2] };
                alpha[i * width + j] =  rgba[3];
                uv[i * width + j] = vec2f(j,i);
            }
        }
    }
    else if (samplesPerPixel == 3) {
        vec3f *ptr = (vec3f*)data_.data();
        auto &uv = img->verts.add_attr<zeno::vec2f>("uv");
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                vec3f rgb = ptr[i * width + j];
                img->verts[i * width + j] = rgb;
                uv[i * width + j] = vec2f(j,i);
            }
        }
    }
    else if (samplesPerPixel == 1) {
        auto &uv = img->verts.add_attr<zeno::vec2f>("uv");
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                float r;
                if(type == 0)
                  r = ((int *)data_.data())[i*width + j];
                else
                  r = ((float *)data_.data())[i*width + j];
                img->verts[i * width + j] = {r, r, r};
                uv[i * width + j] = vec2f(j,i);
            }
        }
    }

    // Process image data here...
    return img;
}
struct ReadTiffFile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto type = get_input2<std::string>("dataType");
        set_output("image", readTiffFile(path, type=="int"?0:1));
    }
};
ZENDEFNODE(ReadTiffFile, {
{
    {"readpath", "path"},
    {"enum int float", "dataType", "float"},
},
{
    {"PrimitiveObject", "image"},
},
{},
    {"primitive"},
});

}


