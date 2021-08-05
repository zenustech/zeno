#ifndef EULER_GAS_DENSE_GRID_H
#define EULER_GAS_DENSE_GRID_H
#include "Types.h"
#include <tbb/tbb.h>
#include <memory>

// using namespace SPGrid;
namespace ZenEulerGas {

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
        assertm((bbmax > bbmin).all(), "illegal bounding box");
        assertm(ghost_layer >= 0, "illegal ghost layer");
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
        else {
            std::cout << "not implemented" << std::endl;
            exit(1);
        }
        resetIDX();
    };
    ~GasDenseGrid(){};

    // spatial index -> linear index
    // the real idx can be modified in the IDMap by i.e. spatial hash to increase locality
    StorageIndex spatialToLinear(const Vector<int, dim>& I)
    {
        assertm((I.array() >= bbmin - ghost_layer).all() && (I.array() < bbmax + ghost_layer).all(), "access out of bound");
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
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else {
            // z-fastest, consistent with ti version
            if constexpr (dim == 1)
                return I_to_min(0);
            else if constexpr (dim == 2)
                return I_to_min(0) * extend(1) + I_to_min(1);
            else if constexpr (dim == 3)
                return I_to_min(0) * extend(1) * extend(2) + I_to_min(1) * extend(2) + I_to_min(2);
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
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
} // namespace ZenEulerGas

#endif
