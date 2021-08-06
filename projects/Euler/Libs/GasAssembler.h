#pragma once

#include "Assembler.h"

namespace ZenEulerGas {
namespace LinearProjection {
template <class T, int dim, class StorageIndex, bool XFastestSweep, int BSplineDegree_ = BSplineDegree::B1B0>
class GasAssembler : virtual public AssemblerBase<T, dim, StorageIndex, XFastestSweep, BSplineDegree_> {
public:
    using IJK = Eigen::Triplet<T>;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;

    using Base = AssemblerBase<T, dim, StorageIndex, XFastestSweep, BSplineDegree_>;
    using FieldHelper = typename Base::FieldHelper;
    using Base::global_operators;
    using Base::sys_name;

    T gamma;

    GasAssembler()
    {
        sys_name = "_gas_";
    }

    // overrides
    // spatial I -> linear system dof
    virtual bool active_int_cell(FieldHelper& field_helper, const IV& I) override { return field_helper.cell_type[field_helper.grid[I].idx] == ZenEulerGas::CellType::GAS; };
    virtual StorageIndex get_idx(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idx; };
    virtual StorageIndex get_idu(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].iduf; };
    virtual StorageIndex get_idP(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idPf; };
    virtual StorageIndex get_idY(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idYf; };
    virtual StorageIndex get_idH(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idH; };
    // copy from spatial field <-> linear vec
    virtual void copy_from_spatial_field(FieldHelper& field_helper) override
    {
        Base::copy_from(field_helper, field_helper.uf, field_helper.Pf, field_helper.nuf, field_helper.nPf, field_helper.nYf, field_helper.nHf);
    };
    virtual void copy_to_spatial_field(FieldHelper& field_helper) override
    {
        Base::copy_to(field_helper, field_helper.uf, field_helper.Pf);
    };
    // for diagonal mass/stiffness matrix
    virtual T get_mass(FieldHelper& field_helper, const IV& I) override { return field_helper.rhof[field_helper.grid[I].idx]; };
    virtual T get_stiffness(FieldHelper& field_helper, const IV& I) override { return gamma * field_helper.Pf_backup[field_helper.grid[I].idx]; };
};
}
} // namespace ZenEulerGas::LinearProjection
