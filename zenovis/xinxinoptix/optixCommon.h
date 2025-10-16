#pragma once

#include <optix.h>
#include <raiicuda.h>

#include "XAS.h"

#ifndef uint 
using uint = unsigned int;
#endif

#ifndef ushort 
using ushort = unsigned short;
#endif

struct SceneNode {
    xinxinoptix::raii<CUdeviceptr> buffer;
    OptixTraversableHandle handle;
};