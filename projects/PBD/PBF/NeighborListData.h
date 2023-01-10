#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
struct NeighborListData : zeno::IObjectClone<NeighborListData>
{
    std::vector<std::vector<int>> value;
};