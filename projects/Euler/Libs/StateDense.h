#ifndef EULER_GAS_STATE_DENSE_H
#define EULER_GAS_STATE_DENSE_H
#include "EulerGasDenseGrid.h"
#include "IdealGas.h"
#include <fstream>
#include <iostream>
#include <tbb/tbb.h>

// using namespace SPGrid;
namespace ZenEulerGas {

enum CellType : int { NONE,
    GAS,
    BOUND,
    SOLID,
    FREE,
    INLET };

enum InterfaceType : int {
    NA,
    GAS_GAS,
    GAS_BOUND,
    GAS_SOLID,
    GAS_FREE,
    GAS_INLET,
    BOUND_GAS,
    BOUND_BOUND,
    BOUND_SOLID,
    BOUND_FREE,
    BOUND_INLET,
    SOLID_GAS,
    SOLID_BOUND,
    SOLID_SOLID,
    SOLID_FREE,
    SOLID_INLET,
    FREE_GAS,
    FREE_BOUND,
    FREE_SOLID,
    FREE_FREE,
    FREE_INLET,
    INLET_GAS,
    INLET_BOUND,
    INLET_SOLID,
    INLET_FREE,
    INLET_INLET
};

template <class T, int dim, class StorageIndex, bool XFastestSweep>
class FieldHelperDense {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using IA = Array<int, dim, 1>;
    using QArray = Array<T, dim + 2, 1>;
    using FArray = Array<T, dim + 2, dim>;

    GasDenseGrid<dim, StorageIndex, XFastestSweep> grid;
    // Background Fields:
    Field<QArray> q, q_backup; // center
    Field<FArray> flux; // mac
    Field<T> rhof; // center
    Field<T> Pf, Pf_backup; //vertex
    Field<TV> uf; //center
    // Solid Background Fields:
    Field<T> rhos; // center
    Field<T> Ps, Ps_backup; // vertex
    Field<TV> us; // center
    Field<T> las, ghost_volume, ghost_volume_center; // vertex/vertex/center
    // celltypes
    Field<int> cell_type, cell_type_backup, cell_type_origin; // center
    // for visualization
    Field<T> shawdow_graph;
    Field<TV> schlieren;
    float dx;
    QArray m_q_amb;
    bool initialized = false;

    // active dof numbers
    StorageIndex nuf = 0, nPf = 0, nYf = 0, nHf = 0;
    StorageIndex nus = 0, nPs = 0, nYs = 0;
    // passive interfaces <I, which face of the element, normal direction>
    Field<std::tuple<IV, int, T>> B_interfaces, H_interfaces, Bs_interfaces,
        Hs_interfaces;
    // const velocity boudnary <I, alpha, normal, velocity>
    Field<std::tuple<IV, int, int, T>> moving_Yf_interfaces_override;
    Field<std::tuple<IV, int, int, T>> moving_Ys_interfaces_override;

    // source term, added to the qs after advection+projection directly
    Field<QArray> source;
    void set_ambient(const Array<T, dim + 2, 1> q_amb_)
    {
        std::fill(q.begin(), q.end(), q_amb_);
        std::fill(q_backup.begin(), q_backup.end(),
            q_amb_);
    }
    FieldHelperDense(const QArray& q_amb, const IA bbmin_, const IA bbmax_, float _dx)
        : grid(bbmin_, bbmax_, 2), dx(_dx)
    {
        // Resize the background fields
        // TODO decouple fields to outside this class
        m_q_amb = q_amb;
        StorageIndex gridNum = grid.gridNum();
        q.resize(gridNum, q_amb);
        q_backup.resize(gridNum, q_amb);

        flux.resize(gridNum, FArray::Zero());
        rhof.resize(gridNum, 0);
        Pf.resize(gridNum, 0);
        Pf_backup.resize(gridNum, 0);
        uf.resize(gridNum, TV::Zero());
        // Resize solid the background fields
        rhos.resize(gridNum, 0);
        Ps.resize(gridNum, 0);
        Ps_backup.resize(gridNum, 0);
        us.resize(gridNum, TV::Zero());
        las.resize(gridNum, 0);
        ghost_volume.resize(gridNum, 0);
        ghost_volume_center.resize(gridNum, 0);
        // Resize cell types
        cell_type.resize(gridNum, CellType::GAS);
        cell_type_backup.resize(gridNum, CellType::GAS);
        cell_type_origin.resize(gridNum, CellType::GAS);
        // visualization only
        shawdow_graph.resize(gridNum, 0);
        schlieren.resize(gridNum, TV::Zero());
        // source term
        source.resize(gridNum, QArray::Zero());
        if (!initialized) {
            set_ambient(q_amb);
            initialized = true;
        }
        std::cout << "constructed field helper" << std::endl;
    };
    ~FieldHelperDense(){};

    // Helper iterate functions:
    template <typename OP>
    void iterateGridSerial(const OP& operation, int extend = 0)
    {
        if constexpr (XFastestSweep) {
            // x - fastest, consistent with VTK
            if constexpr (dim == 1)
                for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend; i++)
                    operation(IV{ i });
            else if constexpr (dim == 2)
                for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend; j++)
                    for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend; i++)
                        operation(IV{ i, j });
            else if constexpr (dim == 3)
                for (int k = grid.bbmin(2) - extend; k < grid.bbmax(2) + extend; k++)
                    for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend; j++)
                        for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend;
                             i++)
                            operation(IV{ i, j, k });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else {
            // z-fastest, consistent with ti
            if constexpr (dim == 1)
                for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend; i++)
                    operation(IV{ i });
            else if constexpr (dim == 2)
                for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend; i++)
                    for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend; j++)
                        operation(IV{ i, j });
            else if constexpr (dim == 3)
                for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend; i++)
                    for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend; j++)
                        for (int k = grid.bbmin(2) - extend; k < grid.bbmax(2) + extend;
                             k++)
                            operation(IV{ i, j, k });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    template <typename OP>
    void iterateGridParallel(const OP& operation, int extend = 0)
    {
        if constexpr (XFastestSweep) {
            // x - fastest, consistent with VTK
            if constexpr (dim == 1)
                tbb::parallel_for<StorageIndex>(
                    grid.bbmin(0) - extend, grid.bbmax(0) + extend,
                    [&](StorageIndex i) { operation(IV{ i }); });
            else if constexpr (dim == 2)
                tbb::parallel_for<StorageIndex>(grid.bbmin(1) - extend,
                    grid.bbmax(1) + extend,
                    [&](StorageIndex j) {
                        for (int i = grid.bbmin(0) - extend;
                             i < grid.bbmax(0) + extend; i++)
                            operation(IV{ i, j });
                    });
            else if constexpr (dim == 3)
                tbb::parallel_for<StorageIndex>(
                    grid.bbmin(2) - extend, grid.bbmax(2) + extend,
                    [&](StorageIndex k) {
                        for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend;
                             j++)
                            for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend;
                                 i++)
                                operation(IV{ i, j, (int)k });
                    });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else {
            // z-fastest, consistent with ti
            if constexpr (dim == 1)
                tbb::parallel_for<StorageIndex>(
                    grid.bbmin(0) - extend, grid.bbmax(0) + extend,
                    [&](StorageIndex i) { operation(IV{ i }); });
            else if constexpr (dim == 2)
                tbb::parallel_for<StorageIndex>(grid.bbmin(0) - extend,
                    grid.bbmax(0) + extend,
                    [&](StorageIndex i) {
                        for (int j = grid.bbmin(1) - extend;
                             j < grid.bbmax(1) + extend; j++)
                            operation(IV{ i, j });
                    });
            else if constexpr (dim == 3)
                tbb::parallel_for<StorageIndex>(
                    grid.bbmin(0) - extend, grid.bbmax(0) + extend,
                    [&](StorageIndex i) {
                        for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend;
                             j++)
                            for (int k = grid.bbmin(2) - extend; k < grid.bbmax(2) + extend;
                                 k++)
                                operation(IV{ i, j, k });
                    });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    template <typename OP>
    void iterateGridColoredParallel(const OP& operation, int nColor = 1,
        int extend = 0)
    {
        if constexpr (XFastestSweep) {
            // x - fastest, consistent with VTK
            if constexpr (dim == 1)
                for (int iColor = 0; iColor < nColor; iColor++)
                    tbb::parallel_for<StorageIndex>(
                        grid.bbmin(0) - extend + iColor, grid.bbmax(0) + extend, nColor,
                        [&](StorageIndex i) { operation(IV{ i }); });
            else if constexpr (dim == 2)
                for (int iColor = 0; iColor < nColor; iColor++)
                    tbb::parallel_for<StorageIndex>(
                        grid.bbmin(1) - extend + iColor, grid.bbmax(1) + extend, nColor,
                        [&](StorageIndex j) {
                            for (int i = grid.bbmin(0) - extend; i < grid.bbmax(0) + extend;
                                 i++)
                                operation(IV{ i, j });
                        });
            else if constexpr (dim == 3)
                for (int iColor = 0; iColor < nColor; iColor++)
                    tbb::parallel_for<StorageIndex>(
                        grid.bbmin(2) - extend + iColor, grid.bbmax(2) + extend, nColor,
                        [&](StorageIndex k) {
                            for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend;
                                 j++)
                                for (int i = grid.bbmin(0) - extend;
                                     i < grid.bbmax(0) + extend; i++)
                                    operation(IV{ i, j, (int)k });
                        });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else {
            // z-fastest, consistent with ti
            if constexpr (dim == 1)
                for (int iColor = 0; iColor < nColor; iColor++)
                    tbb::parallel_for<StorageIndex>(
                        grid.bbmin(0) - extend + iColor, grid.bbmax(0) + extend, nColor,
                        [&](StorageIndex i) { operation(IV{ i }); });
            else if constexpr (dim == 2)
                for (int iColor = 0; iColor < nColor; iColor++)
                    tbb::parallel_for<StorageIndex>(
                        grid.bbmin(0) - extend + iColor, grid.bbmax(0) + extend, nColor,
                        [&](StorageIndex i) {
                            for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend;
                                 j++)
                                operation(IV{ i, j });
                        });
            else if constexpr (dim == 3)
                for (int iColor = 0; iColor < nColor; iColor++)
                    tbb::parallel_for<StorageIndex>(
                        grid.bbmin(0) - extend + iColor, grid.bbmax(0) + extend, nColor,
                        [&](StorageIndex i) {
                            for (int j = grid.bbmin(1) - extend; j < grid.bbmax(1) + extend;
                                 j++)
                                for (int k = grid.bbmin(2) - extend;
                                     k < grid.bbmax(2) + extend; k++)
                                    operation(IV{ i, j, k });
                        });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    // iterate kernel and double kernel(used in element-by-element assembling)
    template <typename OP>
    void iterateKernel(const OP& operation, const IV& I, const int& offset_low,
        const int& offset_high)
    {
        if constexpr (XFastestSweep) {
            if constexpr (dim == 1)
                for (int di = offset_low; di < offset_high; di++)
                    operation(I, IV{ I(0) + di });
            else if constexpr (dim == 2)
                for (int dj = offset_low; dj < offset_high; dj++)
                    for (int di = offset_low; di < offset_high; di++)
                        operation(I, IV{ I(0) + di, I(1) + dj });
            else if constexpr (dim == 3)
                for (int dk = offset_low; dk < offset_high; dk++)
                    for (int dj = offset_low; dj < offset_high; dj++)
                        for (int di = offset_low; di < offset_high; di++)
                            operation(I, IV{ I(0) + di, I(1) + dj, I(2) + dk });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else {
            if constexpr (dim == 1)
                for (int di = offset_low; di < offset_high; di++)
                    operation(I, IV{ I(0) + di });
            else if constexpr (dim == 2)
                for (int di = offset_low; di < offset_high; di++)
                    for (int dj = offset_low; dj < offset_high; dj++)
                        operation(I, IV{ I(0) + di, I(1) + dj });
            else if constexpr (dim == 3)
                for (int di = offset_low; di < offset_high; di++)
                    for (int dj = offset_low; dj < offset_high; dj++)
                        for (int dk = offset_low; dk < offset_high; dk++)
                            operation(I, IV{ I(0) + di, I(1) + dj, I(2) + dk });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    template <typename OP>
    void iterateDoubleKernel(const OP& operation, const IV& I,
        const int& test_low, const int& test_high,
        const int& trial_low, const int& trial_high)
    {
        if constexpr (XFastestSweep) {
            if constexpr (dim == 1)
                for (int di = test_low; di < test_high; di++)
                    for (int di_ = trial_low; di_ < trial_high; di_++)
                        operation(I, IV{ I(0) + di }, IV{ I(0) + di_ });
            else if constexpr (dim == 2)
                for (int dj = test_low; dj < test_high; dj++)
                    for (int di = test_low; di < test_high; di++)
                        for (int dj_ = trial_low; dj_ < trial_high; dj_++)
                            for (int di_ = trial_low; di_ < trial_high; di_++)
                                operation(I, IV{ I(0) + di, I(1) + dj },
                                    IV{ I(0) + di_, I(1) + dj_ });
            else if constexpr (dim == 3)
                for (int dk = test_low; dk < test_high; dk++)
                    for (int dj = test_low; dj < test_high; dj++)
                        for (int di = test_low; di < test_high; di++)
                            for (int dk_ = trial_low; dk_ < trial_high; dk_++)
                                for (int dj_ = trial_low; dj_ < trial_high; dj_++)
                                    for (int di_ = trial_low; di_ < trial_high; di_++)
                                        operation(I, IV{ I(0) + di, I(1) + dj, I(2) + dk },
                                            IV{ I(0) + di_, I(1) + dj_, I(2) + dk_ });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
        else {
            if constexpr (dim == 1)
                for (int di = test_low; di < test_high; di++)
                    for (int di_ = trial_low; di_ < trial_high; di_++)
                        operation(I, IV{ I(0) + di }, IV{ I(0) + di_ });
            else if constexpr (dim == 2)
                for (int di = test_low; di < test_high; di++)
                    for (int dj = test_low; dj < test_high; dj++)
                        for (int di_ = trial_low; di_ < trial_high; di_++)
                            for (int dj_ = trial_low; dj_ < trial_high; dj_++)
                                operation(I, IV{ I(0) + di, I(1) + dj },
                                    IV{ I(0) + di_, I(1) + dj_ });
            else if constexpr (dim == 3)
                for (int di = test_low; di < test_high; di++)
                    for (int dj = test_low; dj < test_high; dj++)
                        for (int dk = test_low; dk < test_high; dk++)
                            for (int di_ = trial_low; di_ < trial_high; di_++)
                                for (int dj_ = trial_low; dj_ < trial_high; dj_++)
                                    for (int dk_ = trial_low; dk_ < trial_high; dk_++)
                                        operation(I, IV{ I(0) + di, I(1) + dj, I(2) + dk },
                                            IV{ I(0) + di_, I(1) + dj_, I(2) + dk_ });
            else {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    // TODO add bi/tri/linear interpolation
    // Cell face values
    std::pair<TV, TV> cellFacePressure(const IV& I)
    {
        if constexpr (dim == 1) {
            auto& g0 = grid[I];
            auto& g1 = grid[IV{ I[0] + 1 }];
            T Pfl = Pf[g0.idx];
            T Pfr = Pf[g1.idx];
            return std::pair<TV, TV>{ TV{ Pfl }, TV{ Pfr } };
        }
        else if constexpr (dim == 2) {
            auto& g0 = grid[I];
            auto& g1 = grid[IV{ I[0] + 1, I[1] }];
            auto& g2 = grid[IV{ I[0] + 1, I[1] + 1 }];
            auto& g3 = grid[IV{ I[0], I[1] + 1 }];
            T Pfl = (Pf[g3.idx] + Pf[g0.idx]) / 2;
            T Pfr = (Pf[g1.idx] + Pf[g2.idx]) / 2;
            T Pfd = (Pf[g0.idx] + Pf[g1.idx]) / 2;
            T Pfu = (Pf[g2.idx] + Pf[g3.idx]) / 2;
            return std::pair<TV, TV>{ TV{ Pfl, Pfd }, TV{ Pfr, Pfu } };
        }
        else if constexpr (dim == 3) {
            auto& g0 = grid[I];
            auto& g1 = grid[IV{ I[0] + 1, I[1], I[2] }];
            auto& g2 = grid[IV{ I[0] + 1, I[1] + 1, I[2] }];
            auto& g3 = grid[IV{ I[0], I[1] + 1, I[2] }];
            auto& g4 = grid[IV{ I[0], I[1], I[2] + 1 }];
            auto& g5 = grid[IV{ I[0] + 1, I[1], I[2] + 1 }];
            auto& g6 = grid[IV{ I[0] + 1, I[1] + 1, I[2] + 1 }];
            auto& g7 = grid[IV{ I[0], I[1] + 1, I[2] + 1 }];
            T Pfl = (Pf[g3.idx] + Pf[g0.idx] + Pf[g7.idx] + Pf[g4.idx]) / 4;
            T Pfr = (Pf[g1.idx] + Pf[g2.idx] + Pf[g5.idx] + Pf[g6.idx]) / 4;
            T Pfd = (Pf[g0.idx] + Pf[g1.idx] + Pf[g4.idx] + Pf[g5.idx]) / 4;
            T Pfu = (Pf[g2.idx] + Pf[g3.idx] + Pf[g6.idx] + Pf[g7.idx]) / 4;
            T Pfb = (Pf[g0.idx] + Pf[g1.idx] + Pf[g2.idx] + Pf[g3.idx]) / 4;
            T Pff = (Pf[g4.idx] + Pf[g5.idx] + Pf[g6.idx] + Pf[g7.idx]) / 4;
            return std::pair<TV, TV>{ TV{ Pfl, Pfd, Pfb }, TV{ Pfr, Pfu, Pff } };
        }
        else {
            {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    std::pair<TV, TV> cellFaceVelocity(const IV& I)
    {
        if constexpr (dim == 1) {
            auto& g = grid[I];
            auto& gl = grid[IV{ I[0] - 1 }];
            auto& gr = grid[IV{ I[0] + 1 }];
            T ufl = (uf[g.idx](0) + uf[gl.idx](0)) / 2;
            T ufr = (uf[g.idx](0) + uf[gr.idx](0)) / 2;
            return std::pair<TV, TV>{ TV{ ufl }, TV{ ufr } };
        }
        else if constexpr (dim == 2) {
            auto& g = grid[I];
            auto& gl = grid[IV{ I[0] - 1, I[1] }];
            auto& gr = grid[IV{ I[0] + 1, I[1] }];
            auto& gd = grid[IV{ I[0], I[1] - 1 }];
            auto& gu = grid[IV{ I[0], I[1] + 1 }];
            T ufl = (uf[g.idx](0) + uf[gl.idx](0)) / 2;
            T ufr = (uf[g.idx](0) + uf[gr.idx](0)) / 2;
            T ufd = (uf[g.idx](1) + uf[gd.idx](1)) / 2;
            T ufu = (uf[g.idx](1) + uf[gu.idx](1)) / 2;
            return std::pair<TV, TV>{ TV{ ufl, ufd }, TV{ ufr, ufu } };
        }
        else if constexpr (dim == 3) {
            auto& g = grid[I];
            auto& gl = grid[IV{ I[0] - 1, I[1], I[2] }];
            auto& gr = grid[IV{ I[0] + 1, I[1], I[2] }];
            auto& gd = grid[IV{ I[0], I[1] - 1, I[2] }];
            auto& gu = grid[IV{ I[0], I[1] + 1, I[2] }];
            auto& gb = grid[IV{ I[0], I[1], I[2] - 1 }];
            auto& gf = grid[IV{ I[0], I[1], I[2] + 1 }];
            T ufl = (uf[g.idx](0) + uf[gl.idx](0)) / 2;
            T ufr = (uf[g.idx](0) + uf[gr.idx](0)) / 2;
            T ufd = (uf[g.idx](1) + uf[gd.idx](1)) / 2;
            T ufu = (uf[g.idx](1) + uf[gu.idx](1)) / 2;
            T ufb = (uf[g.idx](2) + uf[gb.idx](2)) / 2;
            T uff = (uf[g.idx](2) + uf[gf.idx](2)) / 2;
            return std::pair<TV, TV>{ TV{ ufl, ufd, ufb }, TV{ ufr, ufu, uff } };
        }
        else {
            {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    TV cellPressureGrad(const IV& I, T inv_dx)
    {
        auto face_P = cellFacePressure(I);
        TV grad_P(inv_dx * (face_P.second - face_P.first));
        return grad_P;
    };
    TV cellPressureXVelocityGrad(const IV& I, T inv_dx)
    {
        auto face_P = cellFacePressure(I);
        auto face_U = cellFaceVelocity(I);
        TV grad_P(inv_dx * (face_P.second.cwiseProduct(face_U.second) - face_P.first.cwiseProduct(face_U.first)));
        return grad_P;
    };
    IV interface_type(const IV& I)
    {
        auto type = [&](const int& left_cell_type, const int& right_cell_type) {
            if ((left_cell_type == CellType::GAS) && (right_cell_type == CellType::GAS))
                return InterfaceType::GAS_GAS;
            else if ((left_cell_type == CellType::GAS) && (right_cell_type == CellType::BOUND))
                return InterfaceType::GAS_BOUND;
            else if ((left_cell_type == CellType::GAS) && (right_cell_type == CellType::SOLID))
                return InterfaceType::GAS_SOLID;
            else if ((left_cell_type == CellType::GAS) && (right_cell_type == CellType::FREE))
                return InterfaceType::GAS_FREE;
            else if ((left_cell_type == CellType::GAS) && (right_cell_type == CellType::INLET))
                return InterfaceType::GAS_INLET;
            else if ((left_cell_type == CellType::BOUND) && (right_cell_type == CellType::GAS))
                return InterfaceType::BOUND_GAS;
            else if ((left_cell_type == CellType::BOUND) && (right_cell_type == CellType::BOUND))
                return InterfaceType::BOUND_BOUND;
            else if ((left_cell_type == CellType::BOUND) && (right_cell_type == CellType::SOLID))
                return InterfaceType::BOUND_SOLID;
            else if ((left_cell_type == CellType::BOUND) && (right_cell_type == CellType::FREE))
                return InterfaceType::BOUND_FREE;
            else if ((left_cell_type == CellType::BOUND) && (right_cell_type == CellType::INLET))
                return InterfaceType::BOUND_INLET;
            else if ((left_cell_type == CellType::SOLID) && (right_cell_type == CellType::GAS))
                return InterfaceType::SOLID_GAS;
            else if ((left_cell_type == CellType::SOLID) && (right_cell_type == CellType::BOUND))
                return InterfaceType::SOLID_BOUND;
            else if ((left_cell_type == CellType::SOLID) && (right_cell_type == CellType::SOLID))
                return InterfaceType::SOLID_SOLID;
            else if ((left_cell_type == CellType::SOLID) && (right_cell_type == CellType::FREE))
                return InterfaceType::SOLID_FREE;
            else if ((left_cell_type == CellType::SOLID) && (right_cell_type == CellType::INLET))
                return InterfaceType::SOLID_INLET;
            else if ((left_cell_type == CellType::FREE) && (right_cell_type == CellType::GAS))
                return InterfaceType::FREE_GAS;
            else if ((left_cell_type == CellType::FREE) && (right_cell_type == CellType::BOUND))
                return InterfaceType::FREE_BOUND;
            else if ((left_cell_type == CellType::FREE) && (right_cell_type == CellType::SOLID))
                return InterfaceType::FREE_SOLID;
            else if ((left_cell_type == CellType::FREE) && (right_cell_type == CellType::FREE))
                return InterfaceType::FREE_FREE;
            else if ((left_cell_type == CellType::FREE) && (right_cell_type == CellType::INLET))
                return InterfaceType::FREE_INLET;
            else if ((left_cell_type == CellType::INLET) && (right_cell_type == CellType::GAS))
                return InterfaceType::INLET_GAS;
            else if ((left_cell_type == CellType::INLET) && (right_cell_type == CellType::BOUND))
                return InterfaceType::INLET_BOUND;
            else if ((left_cell_type == CellType::INLET) && (right_cell_type == CellType::SOLID))
                return InterfaceType::INLET_SOLID;
            else if ((left_cell_type == CellType::INLET) && (right_cell_type == CellType::FREE))
                return InterfaceType::INLET_FREE;
            else if ((left_cell_type == CellType::INLET) && (right_cell_type == CellType::INLET))
                return InterfaceType::INLET_INLET;
            else
                return InterfaceType::NA;
        };
        if constexpr (dim == 1) {
            auto& g = grid[I];
            auto& gl = grid[IV{ I(0) - 1 }];
            return IV{ type(cell_type[gl.idx], cell_type[g.idx]) };
        }
        else if constexpr (dim == 2) {
            auto& g = grid[I];
            auto& gl = grid[IV{ I(0) - 1, I(1) }];
            auto& gd = grid[IV{ I(0), I(1) - 1 }];
            return IV{ type(cell_type[gl.idx], cell_type[g.idx]),
                type(cell_type[gd.idx], cell_type[g.idx]) };
        }
        else if constexpr (dim == 3) {
            auto& g = grid[I];
            auto& gl = grid[IV{ I(0) - 1, I(1), I(2) }];
            auto& gd = grid[IV{ I(0), I(1) - 1, I(2) }];
            auto& gb = grid[IV{ I(0), I(1), I(2) - 1 }];
            return IV{ type(cell_type[gl.idx], cell_type[g.idx]),
                type(cell_type[gd.idx], cell_type[g.idx]),
                type(cell_type[gb.idx], cell_type[g.idx]) };
        }
        else {
            {
                std::cout << "not implemented" << std::endl;
                exit(1);
            }
        }
    };
    // IO
    void save_ply(std::string filename, T dx, T gamma, bool binary = true)
    {
        return;
    };
};

typedef FieldHelperDense<double, 3, long long, true>* FieldHelperDenseDouble3Ptr;
typedef FieldHelperDense<double, 3, long long, true> FieldHelperDenseDouble3;
} // namespace ZenEulerGas

#endif
