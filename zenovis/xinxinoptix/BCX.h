#pragma once

#include <cstring>
#include <stb_dxt.h>
#include <tbb/task.h>
#include <tbb/task_group.h>

#include <map>
#include <vector>
#include <vector_types.h>

template <char N>
inline void compress(unsigned char* packed, unsigned char* block) {

    if constexpr (N == 1)
        stb_compress_bc4_block(packed, block);
    if constexpr (N == 2)
        stb_compress_bc5_block(packed, block);
    if constexpr (N == 3)
        stb_compress_dxt_block(packed, block, 0, STB_DXT_HIGHQUAL);
    if constexpr (N == 4)
        stb_compress_dxt_block(packed, block, 1, STB_DXT_HIGHQUAL);
}

template <char channel, char byte_per_source_pixel=channel>
inline std::vector<unsigned char> compressBCx(unsigned char* img, uint32_t nx, uint32_t ny) {

    static const char sizes[] = { 0, 8, 16, 8, 16 };

    const auto size_per_packed = sizes[channel];

    auto count = size_per_packed * (nx/4) * (ny/4);
    std::vector<unsigned char> result(count);
    
    tbb::task_group bc_group;

    for (size_t i=0; i<ny; i+=4) { // row

        bc_group.run([&, i]{

            std::vector<unsigned char> block(16 * byte_per_source_pixel, 0);

            for (size_t j=0; j<nx; j+=4) { // col

                for (size_t k=0; k<16; k+=4) {
                    auto offset_i = k / 4;
                    //auto offset_j = k % 4;

                    auto index = nx * (i+offset_i) + (j);

                    auto dst_ptr = block.data() + k*byte_per_source_pixel;
                    auto src_ptr = img + index*byte_per_source_pixel;
                    memcpy(dst_ptr, src_ptr, byte_per_source_pixel * 4);
                }

                auto offset = size_per_packed * ((nx/4) * i/4 + j/4);
                auto packed = result.data()+offset;
                compress<channel>(packed, block.data());
            }

        }); // run
    }

    bc_group.wait();
    return result;
}

inline std::vector<unsigned char> compressBC4(unsigned char* one_byte_per_pixel, uint32_t nx, uint32_t ny) {
    return compressBCx<1>(one_byte_per_pixel, nx, ny);
}

inline std::vector<unsigned char> compressBC5(unsigned char* two_byte_per_pixel, uint32_t nx, uint32_t ny) {
    return compressBCx<2>(two_byte_per_pixel, nx, ny);
}

inline std::vector<unsigned char> compressBC1(unsigned char* three_byte_per_pixel, uint32_t nx, uint32_t ny) {

    auto raw = three_byte_per_pixel;

    auto count = nx * ny;    
    std::vector<uchar4> alt(count);

    for (size_t i=0; i<count; ++i) {
        alt[i] = { raw[i*3 + 0], raw[i*3 + 1], raw[i*3 + 2], 255u };
    }
    
    return compressBCx<3, 4>((unsigned char*)alt.data(), nx, ny);
}

inline std::vector<unsigned char> compressBC3(unsigned char* four_byte_per_pixel, uint32_t nx, uint32_t ny) {
    return compressBCx<4>(four_byte_per_pixel, nx, ny);
}