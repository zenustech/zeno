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

inline void PBF::initData()
{     
    //prepare cell data
    // cell.resize(numCell);
    for (size_t i = 0; i < numCell; i++)
    {
        // //to calculate the x y z coord of cell
        vec3i xyz = cellID2XYZ(i);
        int hash = getCellHash(xyz[0],xyz[1],xyz[2]);

        cell[hash].x = xyz[0];
        cell[hash].y = xyz[1];
        cell[hash].z = xyz[2];
        cell[hash].parInCell.reserve(10); //pre-allocate memory to speed up
    }
    
    //prepare neighbor list 
    neighborList.resize(numParticles);
    for (size_t i = 0; i < numParticles; i++)
        neighborList[i].reserve(10);
}

inline void PBF::initCellData()
{
    dxInv = 1.0/dx;
    int numX = int((bounds_max[0]-bounds_min[0]) / dx) + 1;
    int numY = int((bounds_max[1]-bounds_min[1]) / dx) + 1;
    int numZ = int((bounds_max[2]-bounds_min[2]) / dx) + 1;
    // numCellXYZ.resize(3);
    numCellXYZ[0] = numX;
    numCellXYZ[1] = numY;
    numCellXYZ[2] = numZ;
    numCell = numX * numY * numZ;
}




}//zeno