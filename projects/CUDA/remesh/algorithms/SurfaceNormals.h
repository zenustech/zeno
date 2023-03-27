// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <zeno/utils/vec.h>
#include "../SurfaceMesh.h"

namespace zeno {
namespace pmp {

class SurfaceNormals
{
public:
    SurfaceNormals() = delete;
    SurfaceNormals(const SurfaceNormals&) = delete;

    static vec3f compute_vertex_normal(const SurfaceMesh* mesh, int v);
    static vec3f compute_face_normal(const SurfaceMesh* mesh, int f);
    static void compute_vertex_normals(SurfaceMesh* mesh);

};

} // namespace pmp
} // namespace zeno
