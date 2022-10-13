#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
struct NeighborListData : zeno::IObject
{
    std::vector<std::vector<int>> value;
};