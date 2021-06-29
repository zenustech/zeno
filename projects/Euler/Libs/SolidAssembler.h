#pragma once

#include "Assembler.h"

namespace Bow {
namespace LinearProjection {
template <class T, int dim, class StorageIndex, bool XFastestSweep, int BSplineDegree_ = BSplineDegree::B2B1>
class SolidAssembler : virtual public AssemblerBase<T, dim, StorageIndex, XFastestSweep, BSplineDegree_> {
public:
    using IJK = Eigen::Triplet<T>;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;

    using Base = AssemblerBase<T, dim, StorageIndex, XFastestSweep, BSplineDegree_>;
    using FieldHelper = typename Base::FieldHelper;
    using Base::global_operators;
    using Base::sys_name;

    T P_amb;

    SolidAssembler()
    {
        sys_name = "_solid_";
    }

    // overrides
    // spatial I -> linear system dof
    virtual bool active_int_cell(FieldHelper& field_helper, const IV& I) override { return field_helper.cell_type[field_helper.grid[I].idx] == Bow::EulerGas::CellType::SOLID; };
    virtual StorageIndex get_idx(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idx; };
    virtual StorageIndex get_idu(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idus; };
    virtual StorageIndex get_idP(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idPs; };
    virtual StorageIndex get_idY(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idYs; };
    virtual StorageIndex get_idH(FieldHelper& field_helper, const IV& I) override { return field_helper.grid[I].idH; };
    // copy from spatial field <-> linear vec
    virtual void copy_from_spatial_field(FieldHelper& field_helper) override
    {
        Base::copy_from(field_helper, field_helper.us, field_helper.Ps, field_helper.nus, field_helper.nPs, field_helper.nYs, field_helper.nHf);
    };
    virtual void copy_to_spatial_field(FieldHelper& field_helper) override
    {
        Base::copy_to(field_helper, field_helper.us, field_helper.Ps);
    };
    // for diagonal mass/stiffness matrix
    virtual T get_mass(FieldHelper& field_helper, const IV& I) override { return field_helper.rhos[field_helper.grid[I].idx]; };
    virtual T get_stiffness(FieldHelper& field_helper, const IV& I) override
    {
        StorageIndex idx = get_idx(field_helper, I);
        return field_helper.las[idx] - field_helper.Ps_backup[idx] + P_amb;
    };

    void fix_velocity_by_projection(T dt_by_dx)
    {
        global_operators.Linear_U -= dt_by_dx * global_operators.invM * (global_operators.G * global_operators.Linear_P + global_operators.Y * global_operators.Linear_Y + global_operators.H * global_operators.Linear_H);
    }
};
}
} // namespace Bow::LinearProjection
