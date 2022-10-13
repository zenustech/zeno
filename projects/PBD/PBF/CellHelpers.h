#pragma once
#include <zeno/zeno.h>
#include "PBF.h"

namespace zeno{

//helpers for neighborSearch
inline bool PBF::isInBound(const vec3i& cellXYZ)
{
    return cellXYZ[0] >= 0 && cellXYZ[0] < numCellXYZ[0] &&
           cellXYZ[1] >= 0 && cellXYZ[1] < numCellXYZ[1] &&
           cellXYZ[2] >= 0 && cellXYZ[2] < numCellXYZ[2];
}

inline int PBF::getCellID(const vec3f& p)
{
    vec3i xyz = p*dxInv;
    int numPerRow = numCellXYZ[0];
    int numPerFloor = numCellXYZ[0] * numCellXYZ[1];
    int res = numPerFloor * xyz[2] + numPerRow * xyz[1] + xyz[0];
    return res;
}

inline int PBF::cellXYZ2ID(const vec3i& xyz)
{
    int numPerRow = numCellXYZ[0];
    int numPerFloor = numCellXYZ[0] * numCellXYZ[1];
    int res = numPerFloor * xyz[2] + numPerRow * xyz[1] + xyz[0];
    return res;
}

inline vec3i PBF::cellID2XYZ(int i)
{
    //to calculate the x y z coord of cell
    int numPerRow = numCellXYZ[0];
    int numPerFloor = numCellXYZ[0] * numCellXYZ[1];

    int floor = (i / numPerFloor);
    int row = (i % numPerFloor) / numPerRow;
    int col = (i % numPerFloor) % numPerRow; 
    
    vec3i res{col,row,floor};
    return res;
}

inline vec3i PBF::getCellXYZ(const vec3f& p)
{
    vec3i res{p*dxInv};
    return res;
}

inline int PBF::getCellHash(int i, int j, int k)
{
    int res = ( (73856093 * i) ^ 
                (19349663 * j) ^ 
                (83492791 * k) ) 
                % (2 * numParticles);
    return res;
}

}//zeno