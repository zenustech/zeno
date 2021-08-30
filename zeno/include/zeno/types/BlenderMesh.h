#pragma once

#ifndef _MSC_VER
#warning "<zeno/types/BlenderMesh.h> is deprecated, use <zeno/types/PolyMeshObject.h> instead"
#endif

#include <zeno/types/PolymeshObject.h>

namespace zeno {

using BlenderMesh = PolyMeshObject;
using BlenderAxis = TransformObject;

};
