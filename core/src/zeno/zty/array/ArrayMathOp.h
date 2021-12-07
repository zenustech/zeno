#pragma once


#include <string>
#include <zeno/zty/array/Array.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


Array arrayMathOp(std::string const &type, Array const &arr1);
Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2);
Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2, Array const &arr3);


}
ZENO_NAMESPACE_END
