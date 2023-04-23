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
static void flipVertically(PrimitiveObject* image) {
    auto& ud = image->userData();
    int width = ud.get2<int>("w");
    int height = ud.get2<int>("h");
    {
        std::vector<vec3f> temp(width);
        for (auto i = 0; i < height / 2; i++) {
            memcpy(temp.data(), &image->verts[i * width], width * sizeof(vec3f));
            memcpy(&image->verts[i * width], &image->verts[(height - 1 - i) * width], width * sizeof(vec3f));
            memcpy(&image->verts[(height - 1 - i) * width], temp.data(), width * sizeof(vec3f));
        }
    }
    if (image->verts.has_attr("alpha")) {
        std::vector<float> temp(width);
        auto &alpha = image->verts.attr<float>("alpha");
        for (auto i = 0; i < height / 2; i++) {
            memcpy(temp.data(), &alpha[i * width], width * sizeof(float));
            memcpy(&alpha[i * width], &alpha[(height - 1 - i) * width], width * sizeof(float));
            memcpy(&alpha[(height - 1 - i) * width], temp.data(), width * sizeof(float));
        }
    }
}
std::shared_ptr<PrimitiveObject> readTiffFile(std::string const &path) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) {
        throw std::runtime_error("tiff read fail");
    }
    auto img = std::make_shared<PrimitiveObject>();

    uint32 width, height;
    uint16_t samplesPerPixel, bitsPerSample;
    uint32 tile_width = 0;
    uint32 tile_height = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_height);
    img->userData().set2("isImage", 1);
    img->userData().set2("w", (int)width);
    img->userData().set2("h", (int)height);
    img->userData().set2("samplesPerPixel", (int)samplesPerPixel);
    img->userData().set2("bitsPerSample", (int)bitsPerSample);
    img->userData().set2("tileWidth", (int)tile_width);
    img->userData().set2("tileHeight", (int)tile_height);
    if (bitsPerSample != 32) {
        throw std::runtime_error("tiff read fail");
    }

    std::vector<uint8_t> data_;
    uint32_t rowSize = TIFFScanlineSize(tif);
    img->userData().set2("rowSize", (int)rowSize);
    data_.resize(rowSize * height);

    if (tile_height == 0 && tile_height == 0) {
        for (int32_t row = 0; row < height; ++row) {
            TIFFReadScanline(tif, &data_[row * rowSize], row);
        }
    }
    else {
        if (TIFFTileSize(tif) != (uint32)(tile_width * tile_height * sizeof(float))) {
            zeno::log_error("TIFFTileSize(tif) != tileWidth * tileHeight * sizeof(uint32)");
        }
        float* tile = (float*)_TIFFmalloc(tile_width * tile_height * sizeof(float));
        if (!tile) {
            zeno::log_error("Failed to allocate memory for tile!");
        }

        for (uint32_t y = 0; y < height; y += tile_height) {
            for (uint32_t x = 0; x < width; x += tile_width) {
                TIFFReadTile(tif, tile, x, y, 0, 0);

                uint32_t w = (x + tile_width > width) ? width - x : tile_width;
                uint32_t h = (y + tile_height > height) ? height - y : tile_height;

                for (auto j = 0; j < h; j++) {
                    auto tile_index = j * tile_width;
                    auto index = (y + j) * width + x;
                    auto ptr = (float*)data_.data();
                    memcpy(&ptr[index], &tile[tile_index], w * sizeof(float));
                }
            }
        }

        _TIFFfree(tile);
    }
    TIFFClose(tif);

    img->resize(width * height);
    if (samplesPerPixel == 4) {
        vec4f *ptr = (vec4f*)data_.data();
        auto &alpha = img->verts.add_attr<float>("alpha");
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                vec4f rgba = ptr[i * width + j];
                img->verts[i * width + j] =  { rgba[0], rgba[1], rgba[2] };
                alpha[i * width + j] =  rgba[3];
            }
        }
    }
    else if (samplesPerPixel == 3) {
        vec3f *ptr = (vec3f*)data_.data();
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                vec3f rgb = ptr[i * width + j];
                img->verts[i * width + j] = rgb;
            }
        }
    }
    else if (samplesPerPixel == 1) {
        float *ptr = (float *)data_.data();
        for (auto i = 0; i < height; i++) {
            for (auto j = 0; j < width; j++) {
                float r = ptr[i * width + j];
                img->verts[i * width + j] = {r, r, r};
            }
        }
    }

    // Process image data here...
    flipVertically(img.get());
    return img;
}
struct ReadTiffFile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        set_output("image", readTiffFile(path));
    }
};
ZENDEFNODE(ReadTiffFile, {
{
    {"readpath", "path"},
},
{
    {"PrimitiveObject", "image"},
},
{},
    {"primitive"},
});

struct WriteTiffFile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto image = get_input2<PrimitiveObject>("image");
        auto tiled = get_input2<bool>("tiled");
        auto& ud = image->userData();
        int width = ud.get2<int>("w");
        int height = ud.get2<int>("h");
        flipVertically(image.get());
        TIFF* tif = TIFFOpen(path.c_str(), "w");
        if (tif == nullptr) {
            zeno::log_error("Failed to open TIFF file: {}", path);
        }
        uint16_t bits_per_sample = 32;
        uint16_t samples_per_pixel = 1;
        uint16_t planar_config = PLANARCONFIG_CONTIG;
        uint16_t photometric = PHOTOMETRIC_MINISBLACK;
        uint32_t tile_width = 128;
        uint32_t tile_height = 128;

        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, planar_config);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, photometric);
        if (tiled) {
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_height);
        }

        if (tiled) {
            for (uint32_t y = 0; y < height; y += tile_height) {
                for (uint32_t x = 0; x < width; x += tile_width) {

                    uint32_t w = (x + tile_width > width) ? width - x : tile_width;
                    uint32_t h = (y + tile_height > height) ? height - y : tile_height;

                    std::vector<float> temp;
                    temp.reserve(tile_width * tile_height);
                    if (w == tile_width && h == tile_height) {
                        for (auto j = y; j < y + tile_height; j++) {
                            for (auto i = x; i < x + tile_width; i++) {
                                int index = j * width + i;
                                temp.push_back(image->verts[index][0]);
                            }
                        }
                    }
                    else {
                        for (auto j = y; j < y + tile_height; j++) {
                            for (auto i = x; i < x + tile_width; i++) {
                                if (j < y + h && i < x + w) {
                                    int index = j * width + i;
                                    temp.push_back(image->verts[index][0]);
                                }
                                else {
                                    temp.push_back(0);
                                }
                            }
                        }
                    }

                    TIFFWriteEncodedTile(tif, TIFFComputeTile(tif, x, y, 0, 0), temp.data(), tile_width * tile_height * sizeof(float));
                }
            }
        }
        else {
            std::vector<float> data(width * height);
            for (auto i = 0; i < width; i++) {
                for (auto j = 0; j < height; j++) {
                    int index = j * width + i;
                    data[index] = image->verts[index][0];
                }
            }
            for (uint32_t y = 0; y < height; y++) {
                TIFFWriteScanline(tif, &data[y * width], y);
            }
        }

        TIFFClose(tif);
        zeno::log_info("TIFF write done!");
    }
};
ZENDEFNODE(WriteTiffFile, {
{
    {"writepath", "path"},
    {"image"},
    {"bool", "tiled", "0"},
},
{},
{},
{"primitive"},
});
}


