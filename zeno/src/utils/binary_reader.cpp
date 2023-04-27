//
// Created by zhouhang on 2023/4/28.
//
#include "zeno/utils/binary_reader.h"

namespace zeno {
    BinaryReader::BinaryReader(std::vector<uint8_t> data_) {
        data = data_;
    }
    uint32_t BinaryReader::read_u32_LE() {
        uint32_t v = *(uint32_t *)(data.data() + cur);
        cur += 4;
        return v;
    }
    float BinaryReader::read_f32_LE() {
        float v = *(float *)(data.data() + cur);
        cur += 4;
        return v;
    }
    vec4f BinaryReader::read_vec4f_LE() {
        auto x = read_f32_LE();
        auto y = read_f32_LE();
        auto z = read_f32_LE();
        auto w = read_f32_LE();
        return {x, y, z, w};
    }
}