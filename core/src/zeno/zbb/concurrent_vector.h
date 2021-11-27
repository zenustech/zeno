// TODO: https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
#pragma once


#include <zeno/common.h>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T, class Alloc = std::allocator<T>>
struct concurrent_vector {
};


}
ZENO_NAMESPACE_END
