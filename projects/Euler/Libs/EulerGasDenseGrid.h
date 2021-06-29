#ifndef EULER_GAS_DENSE_GRID_H
#define EULER_GAS_DENSE_GRID_H
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/Logging.h>
#include <tbb/tbb.h>
#include <memory>

// using namespace SPGrid;
namespace Bow {
namespace EulerGas {

template <class StorageIndex>
class IDMap {
public:
    StorageIndex idx;
    StorageIndex iduf, idPf, idYf, idH;
    StorageIndex idus, idPs, idYs;
    IDMap()
    {
        idx = -1;
        iduf = -1;
        idPf = -1;
        idYf = -1;
        idH = -1;
        idus = -1;
        idPs = -1;
        idYs = -1;
    }
};

// normal dense container
template <int dim, class StorageIndex, bool XFastestSweep>
class GasDenseGrid {
public:
    Field<IDMap<StorageIndex>> grid;
    // domain bbox
    using IA = Array<int, dim, 1>;
    IA bbmin;
    IA bbmax;
    int ghost_layer;

    GasDenseGrid(const IA bbmin_, const IA bbmax_, int ghost_layer_)
        : bbmin(bbmin_), bbmax(bbmax_), ghost_layer(ghost_layer_)
    {
        BOW_ASSERT_INFO((bbmax > bbmin).all(), "illegal bounding box");
        BOW_ASSERT_INFO(ghost_layer >= 0, "illegal ghost layer");
        IA extend = bbmax - bbmin + 2 * ghost_layer;
        if constexpr (dim == 1) {
            StorageIndex size = extend(0);
            grid.resize(size, IDMap<StorageIndex>());
        }
        else if constexpr (dim == 2) {
            StorageIndex size = extend(0) * extend(1);
            grid.resize(size, IDMap<StorageIndex>());
        }
        else if constexpr (dim == 3) {
            StorageIndex size = extend(0) * extend(1) * extend(2);
            grid.resize(size, IDMap<StorageIndex>());
        }
        else
            BOW_NOT_IMPLEMENTED
        resetIDX();
    };
    ~GasDenseGrid(){};

    // spatial index -> linear index
    // the real idx can be modified in the IDMap by i.e. spatial hash to increase locality
    StorageIndex spatialToLinear(const Vector<int, dim>& I)
    {
        BOW_ASSERT_INFO((I.array() >= bbmin - ghost_layer).all() && (I.array() < bbmax + ghost_layer).all(), "access out of bound");
        IA extend = bbmax - bbmin + 2 * ghost_layer;
        IA I_to_min = I.array() - bbmin + ghost_layer;
        if constexpr (XFastestSweep) {
            // x-fastest, consistent with the VTK structure grid ordering
            if constexpr (dim == 1)
                return I_to_min(0);
            else if constexpr (dim == 2)
                return I_to_min(0) + I_to_min(1) * extend(0);
            else if constexpr (dim == 3)
                return I_to_min(0) + I_to_min(1) * extend(0) + I_to_min(2) * extend(0) * extend(1);
            else
                BOW_NOT_IMPLEMENTED
        }
        else {
            // z-fastest, consistent with ti version
            if constexpr (dim == 1)
                return I_to_min(0);
            else if constexpr (dim == 2)
                return I_to_min(0) * extend(1) + I_to_min(1);
            else if constexpr (dim == 3)
                return I_to_min(0) * extend(1) * extend(2) + I_to_min(1) * extend(2) + I_to_min(2);
            else
                BOW_NOT_IMPLEMENTED
        }
    };

    // []
    IDMap<StorageIndex>& operator[](const Vector<int, dim>& I)
    {
        return grid[spatialToLinear(I)];
    }
    const IDMap<StorageIndex>& operator[](const Vector<int, dim>& I) const
    {
        return grid[spatialToLinear(I)];
    }

    // judge if out of bbox
    bool in_bbox(const Vector<int, dim>& I, int extend = 0)
    {
        return (I.array() >= bbmin - extend).all() && (I.array() < bbmax + extend).all();
    }
    void resetGridState()
    {
        std::fill(grid.begin(), grid.end(), IDMap<StorageIndex>());
    };
    void resetIDX()
    {
        // defualt idx is the same as the linear id, if all cell active
        tbb::parallel_for<StorageIndex>(0, grid.size(), [&](StorageIndex i) {
            grid[i].idx = i;
        });
    };
    StorageIndex gridNum() { return grid.size(); };
};
}
} // namespace Bow::EulerGas

#endif
