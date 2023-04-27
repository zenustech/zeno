//
// Created by zhouhang on 2023/4/28.
//

#ifndef ZENO_BINARY_READER_H
#define ZENO_BINARY_READER_H
#include <vector>
#include <stdint.h>
#include "zeno/utils/vec.h"

namespace zeno {
class BinaryReader {
    size_t cur = 0;
    std::vector<uint8_t> data;
public:
    BinaryReader(std::vector<uint8_t> data_);
    uint32_t read_u32_LE();
    float read_f32_LE();
    vec4f read_vec4f_LE();
};

}

#endif //ZENO_BINARY_READER_H
