#pragma once

namespace zeno {
    // some supported cell types
    constexpr int VTK_VERTEEX = 1;
    constexpr int VTK_TRIANGLE = 5;
    constexpr int VTK_TRIANGLE_STRIP = 6;
    constexpr int VTK_POLYGON = 7;
    constexpr int VTK_PIXEL = 8;
    constexpr int VTK_QUAD = 9;
    constexpr int VTK_TETRA = 10;
    constexpr int VTK_VOXEL = 11;
    constexpr int VTK_HEXAHEDRON = 12;
    constexpr int VTK_WEDGE = 13;
    constexpr int PYRAMID = 14;
    constexpr int VTK_PENTAGONAL_PRISM = 15;
    constexpr int VTK_HEXAGONAL_PRISM = 16;
    // // some other vtk cell types

    constexpr int FILENAMESIZE = 1024;
    constexpr int INPUTLINESIZE = 2048;

}