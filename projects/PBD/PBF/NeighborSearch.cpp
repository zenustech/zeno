#include <zeno/zeno.h>
#include "PBF.h"
#include "CellHelpers.h"
namespace zeno{

// This neighborSearch algorithm uses grid-based searching,
// which is simple but can be improved
void PBF::neighborSearch()
{
    auto &pos = prim->verts;
    //clear parInCell and neighborList
    for (size_t i = 0; i < numCell; i++)
        cell[i].parInCell.clear();
    for (size_t i = 0; i < numParticles; i++)
        neighborList[i].clear();
    
    //update the parInCell list
    vec3i cellXYZ;
    int cellID;
    for (size_t i = 0; i < numParticles; i++) // i is the particle ID
    {
        cellID = getCellID(pos[i]);
        // int hash = getCellHash(cellXYZ[0], cellXYZ[1], cellXYZ[2]);
        cell[cellID].parInCell.push_back(i);
    }

    //update the neighborList
    for (size_t i = 0; i < numParticles; i++)
    {
        cellXYZ = getCellXYZ(pos[i]);
        for (int off_x = -1; off_x < 2; off_x++)
            for (int off_y = -1; off_y < 2; off_y++)
                for (int off_z = -1; off_z < 2; off_z++)
                {
                    vec3i off{off_x, off_y, off_z};
                    vec3i toCheckXYZ = cellXYZ + off;
                    int toCheck = cellXYZ2ID(toCheckXYZ);
                    if (isInBound(toCheckXYZ))
                    {
                        Cell theCell = cell[toCheck];
                        std::vector<int> parInTheCell = theCell.parInCell;

                        for (int j = 0; j < parInTheCell.size(); j++)
                        {
                            int p = parInTheCell[j];
                            if(p!=i && length(pos[i] - pos[p]) < neighborSearchRadius)
                            {
                                neighborList[i].push_back(p);
                            }
                        }
                    }
                }  
    }
}

}//zeno
