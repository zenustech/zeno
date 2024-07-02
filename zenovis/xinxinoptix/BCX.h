#pragma once

#include <stb_dxt.h>
#include <tbb/task.h>
#include <tbb/task_group.h>

#include <map>
#include <vector>
#include <vector_types.h>

template <char N>
inline void compress(std::vector<unsigned char> &packed, std::vector<unsigned char> &block) {

    if constexpr (N == 1)
        return stb_compress_bc4_block(packed.data(), (unsigned char*)block.data());
    if constexpr (N == 2)
        return stb_compress_bc5_block(packed.data(), (unsigned char*)block.data());
    if constexpr (N == 4)
        return stb_compress_dxt_block(packed.data(), (unsigned char*)block.data(), 1, STB_DXT_HIGHQUAL);
}

template <char N>
inline std::vector<unsigned char> compressBCx(unsigned char* img, uint32_t nx, uint32_t ny) {

    static std::map<char, uint32_t> sizes {
        { 1, 8 },
        { 2, 16},
        { 4, 16}
    };

    const auto size_per_packed = sizes[N];

    auto count = size_per_packed * (nx/4) * (ny/4);
    std::vector<unsigned char> result(count);
    
    tbb::task_group bc_group;

    for (size_t i=0; i<ny; i+=4) { // row

        bc_group.run([&, i]{

            std::vector<unsigned char> block(16 * N, 0);
            std::vector<unsigned char> packed(size_per_packed, 0);

            for (size_t j=0; j<nx; j+=4) { // col

                for (size_t k=0; k<16; k+=4) {
                    auto offset_i = k / 4;
                    //auto offset_j = k % 4;

                    auto index = nx * (i+offset_i) + (j);
                    //raw_block[k] = img[index];
                    auto dst_ptr = block.data() + k*N;
                    auto src_ptr = img + index*N ;
                    memcpy(dst_ptr, src_ptr, N * 4);
                }
                compress<N>(packed, block);

                auto offset = size_per_packed * ((nx/4) * i/4 + j/4);
                memcpy(result.data()+offset, packed.data(), packed.size());
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

inline std::vector<unsigned char> compressBC3(unsigned char* four_byte_per_pixel, uint32_t nx, uint32_t ny) {
    return compressBCx<4>(four_byte_per_pixel, nx, ny);
}