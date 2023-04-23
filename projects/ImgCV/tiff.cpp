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
std::shared_ptr<PrimitiveObject> readTiffFile(std::string const &path) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) {
        throw std::runtime_error("tiff read fail");
    }
    auto img = std::make_shared<PrimitiveObject>();

    uint32 width, height;
    uint16_t samplesPerPixel, bitsPerSample;
    uint32 tileWidth = 0;
    uint32 tileHeight = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
    img->userData().set2("isImage", 1);
    img->userData().set2("w", (int)width);
    img->userData().set2("h", (int)height);
    img->userData().set2("samplesPerPixel", (int)samplesPerPixel);
    img->userData().set2("bitsPerSample", (int)bitsPerSample);
    img->userData().set2("tileWidth", (int)tileWidth);
    img->userData().set2("tileHeight", (int)tileHeight);
    if (bitsPerSample != 32) {
        throw std::runtime_error("tiff read fail");
    }

    std::vector<uint8_t> data_;
    uint32_t rowSize = TIFFScanlineSize(tif);
    img->userData().set2("rowSize", (int)rowSize);
    data_.resize(rowSize * height);

    if (tileHeight == 0 && tileWidth == 0) {
        for (int32_t row = 0; row < height; ++row) {
            TIFFReadScanline(tif, &data_[row * rowSize], row);
        }
    }
    else {
        if (TIFFTileSize(tif) != (uint32)(tileWidth * tileHeight * sizeof(uint32))) {
            zeno::log_error("TIFFTileSize(tif) != tileWidth * tileHeight * sizeof(uint32)");
        }
        uint32* tile = (uint32*)_TIFFmalloc(tileWidth * tileHeight * sizeof(uint32));
        if (!tile) {
            zeno::log_error("Failed to allocate memory for tile!");
        }

        for (uint32 tileY = 0; tileY < height; tileY += tileHeight) {
            zeno::log_info("tile: {} ", tileY);
            for (uint32 tileX = 0; tileX < width; tileX += tileWidth) {
                uint32 maxX = tileX + tileWidth < width ? tileX + tileWidth : width;
                uint32 maxY = tileY + tileHeight < height ? tileY + tileHeight : height;

                for (uint32 y = tileY; y < maxY; y++) {
                    uint32* dataPtr = (uint32*)((uint32*)data_.data() + y * width + tileX);
                    for (uint32 x = tileX; x < maxX; x += tileWidth) {
                        TIFFReadTile(tif, tile, x, y, 0, 0);
                        for (uint32 i = 0; i < tileWidth && x + i < width; i++) {
                            dataPtr[i] = tile[i];
                        }
                        dataPtr += tileWidth;
                    }
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

struct TestTiff : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        TIFF* tif = TIFFOpen(path.c_str(), "r");
        if (!tif) {
            printf("Failed to open TIFF file: %s\n", path.c_str());
        }

        // Determine if TIFF file is tiled
        uint32 tile_width, tile_length;
        if (TIFFIsTiled(tif)) {
            printf("TIFF file is tiled.\n");
            TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width);
            TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_length);
        } else {
            printf("TIFF file is not tiled.\n");
            TIFFClose(tif);
        }

        // Read a tile
        uint32 tile_x = 0;
        uint32 tile_y = 0;
        uint16 tile_index = 0;
        uint32 tile_size = TIFFTileSize(tif);
        void* tile_buf = _TIFFmalloc(tile_size);
        tsize_t result = TIFFReadTile(tif, tile_buf, tile_x, tile_y, 0, tile_index);
        if (result > 0) {
            printf("Successfully read tile %d at (%d, %d)\n", tile_index, tile_x, tile_y);
        } else {
            printf("Failed to read tile %d at (%d, %d)\n", tile_index, tile_x, tile_y);
        }

        _TIFFfree(tile_buf);
        TIFFClose(tif);
    }
};
ZENDEFNODE(TestTiff, {
{
    {"readpath", "path"},
},
{
},
{},
    {"primitive"},
});
}


