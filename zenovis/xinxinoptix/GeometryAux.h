#pragma once

#include <cuda/BufferView.h>
#include <sutil/vec_math.h>

#ifndef __CUDACC_RTC__
#include <cassert>
#else
#define assert(x) /*nop*/
#endif    
    
struct CurveGroupAux
{
    BufferView<float2> strand_u;     // strand_u at segment start per segment
    GenericBufferView  strand_i;     // strand index per segment
    BufferView<uint2>  strand_info;  // info.x = segment base
                                     // info.y = strand length (segments)
};
