#pragma once

#include "AdvectionOp.h"
#include "BasicOp.h"
#include "Builder.h"
#include "GasAssembler.h"
#include "ProjectionOp.h"

namespace ZenEulerGas {

template <class T, int dim>
struct solverControl {
    // ambient data
    Array<T, dim + 2, 1> q_amb;
    T P_amb;
    T gamma;
    // clamp data
    T clamp_ratio = 1e-6;
    T lowest_rho = 1.25e-6;
    T lowest_int_e_by_rho = 2e-6;
    // control datas
    bool use_RK = true;
    bool high_order_Bspline = false;
    bool have_solid = false;
    bool have_gravity_gas = false;
    bool add_source_term = false;
    T cg_converge_cretiria = 1e-7;
    int cg_it_limit = 500;
    bool output_vtk = false;
    // calculating dt
    T dt_min = 5e-10;
    T dt_max = 5e-2;
    T CFL = 0.5;
    T v_ref = 0;
    T a_ref = 0;
    bool initialized = false;
};
typedef solverControl<double, 3> solverControld;
typedef solverControl<float, 3> solverControlf;
template <class T, int dim, class StorageIndex, bool XFastestSweep = false>
class GasSimulator {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using QArray = Array<T, dim + 2, 1>;
    // base
    // using Base = PhysicallyBasedSimulator<T, dim>;
    int end_frame;
    float frame_dt;
    std::string output_directory;
    float suggested_dt;

    T dx;

    // ambient data
    Array<T, dim + 2, 1> q_amb;
    T P_amb;
    T gamma;
    // clamp data
    T clamp_ratio = 1e-3;
    T lowest_rho = 1.25e-3;
    T lowest_int_e_by_rho = 2e-3;
    // control datas
    bool use_RK = true;
    bool high_order_Bspline = false;
    bool have_solid = false;
    bool have_gravity_gas = false;
    bool add_source_term = false;
    T cg_converge_cretiria = 1e-7;
    int cg_it_limit = 500;
    bool output_vtk = false;
    // calculating dt
    T dt_min = 5e-10;
    T dt_max = 5e-2;
    T CFL = 0.5;
    T v_ref = 0;
    T a_ref = 0;
    bool initialized = false;

    // fields wrapper
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    // linear projection sys
    // assembler and builder
    ZenEulerGas::LinearProjection::GasAssembler<
        T, dim, StorageIndex, XFastestSweep,
        ZenEulerGas::LinearProjection::BSplineDegree::B1B0>
        gas_assembler;
    ZenEulerGas::LinearProjection::Builder<T, dim, StorageIndex> sys_builder;

    // set ambient value
    void set_ambient(const Array<T, dim + 2, 1> q_amb_, const T gamma_)
    {
        assertm(q_amb_(0) > 0, "defined negative density for ambient");
        assertm(q_amb_(1) > 0, "defined negative total energy for ambient");
        T int_e = q_amb_(1) - 0.5 * (q_amb_.template tail<dim>() * q_amb_.template tail<dim>()).sum() / q_amb_(0);
        assertm(int_e > 0, "defined negative internal energy for ambient");
        q_amb = q_amb_;
        P_amb = ZenEulerGas::ConstitutiveModel::IdealGas::get_pressure_from_int_energy(
            gamma_, int_e);
        gamma = gamma_;
        gas_assembler.gamma = gamma;
        // threshold for clamping
        lowest_rho = clamp_ratio * q_amb(0);
        lowest_int_e_by_rho = clamp_ratio * int_e / q_amb(0);
    }

    // set dx
    void set_dx(const T dx_)
    {
        dx = dx_;
        sys_builder.dx = dx_;
    }

    solverControl<T, dim> getSolverControl()
    {
        solverControl<T, dim> _sc;
        _sc.q_amb = q_amb;
        _sc.P_amb = P_amb;
        _sc.gamma = gamma;
        _sc.clamp_ratio = clamp_ratio;
        _sc.lowest_rho = lowest_rho;
        _sc.lowest_int_e_by_rho = lowest_int_e_by_rho;
        _sc.use_RK = use_RK;
        _sc.high_order_Bspline = high_order_Bspline;
        _sc.have_solid = have_solid;
        _sc.have_gravity_gas = have_gravity_gas;
        _sc.add_source_term = add_source_term;
        _sc.cg_converge_cretiria = cg_converge_cretiria;
        _sc.cg_it_limit = cg_it_limit;
        _sc.output_vtk = output_vtk;
        _sc.dt_min = dt_min;
        _sc.dt_max = dt_max;
        _sc.CFL = CFL;
        _sc.v_ref = v_ref;
        _sc.a_ref = a_ref;
        _sc.initialized = initialized;
        return _sc;
    }

    void setSolverControl(solverControl<T, dim>& _sc)
    {
        q_amb = _sc.q_amb;
        P_amb = _sc.P_amb;
        gamma = _sc.gamma;
        clamp_ratio = _sc.clamp_ratio;
        lowest_rho = _sc.lowest_rho;
        lowest_int_e_by_rho = _sc.lowest_int_e_by_rho;
        use_RK = _sc.use_RK;
        high_order_Bspline = _sc.high_order_Bspline;
        have_solid = _sc.have_solid;
        have_gravity_gas = _sc.have_gravity_gas;
        add_source_term = _sc.add_source_term;
        cg_converge_cretiria = _sc.cg_converge_cretiria;
        cg_it_limit = _sc.cg_it_limit;
        output_vtk = _sc.output_vtk;
        dt_min = _sc.dt_min;
        dt_max = _sc.dt_max;
        CFL = _sc.CFL;
        v_ref = _sc.v_ref;
        a_ref = _sc.a_ref;
        initialized = _sc.initialized;
    }

    // constructor
    GasSimulator(const T dx_, const Array<int, dim, 1> bbmin_,
        const Array<int, dim, 1> bbmax_,
        const Array<T, dim + 2, 1> q_amb_, FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& _field_helper,
        const T gamma_ = 1.4,
        const T CFL_ = 0.5)
        : dx(dx_), field_helper(_field_helper), sys_builder(dx_), gamma(gamma_), CFL(CFL_)
    {
        set_ambient(q_amb_, gamma_);
    }

    T calculate_dt()
    {
        // get max velocity and acceleration
        TV max_abs_v = std::abs(v_ref) * TV::Ones();
        TV max_abs_a = std::abs(a_ref) * TV::Ones();
        T inv_dx = (T)1 / dx;
        // loop gas
        auto find_max_v_a = [&](const IV& I) {
            if (field_helper.cell_type[field_helper.grid[I].idx] == CellType::GAS) {
                auto& g = field_helper.grid[I];
                auto grad_P = field_helper.cellPressureGrad(I, inv_dx);
                max_abs_v = max_abs_v.cwiseMax((field_helper.uf[g.idx]).cwiseAbs());
                max_abs_a = max_abs_a.cwiseMax((grad_P / field_helper.q[g.idx](0)).cwiseAbs());
            }
        };
        field_helper.iterateGridSerial(find_max_v_a);
        // loop solid
        auto find_max_v_a_solid = [&](const IV& I) {
            if (field_helper.cell_type[field_helper.grid[I].idx] == CellType::SOLID) {
                auto& g = field_helper.grid[I];
                max_abs_v = max_abs_v.cwiseMax((field_helper.us[g.idx]).cwiseAbs());
            }
        };
        field_helper.iterateGridSerial(find_max_v_a_solid);
        // apply modified cfl condition
        T sum_abs_v_by_dx = max_abs_v.sum() * inv_dx;
        T sum_abs_a_by_dx = max_abs_a.sum() * inv_dx;
        T dt = CFL * (T)2 / (sum_abs_v_by_dx + std::sqrt(sum_abs_v_by_dx * sum_abs_v_by_dx + (T)4 * sum_abs_a_by_dx));
        // Logging::debug("sum_abs_a_by_dx ", sum_abs_a_by_dx);
        // Logging::debug("sum_abs_v_by_dx ", sum_abs_v_by_dx);
        // Logging::info("dt_no_clamp ", dt);
        if (dt < dt_min) {
            // Logging::error("too little dt ", dt);
            std::cout << "too little dt " << std::endl;
            exit(1);
        }
        dt = std::min(std::max(dt_min, dt), dt_max);
        // Logging::info("calculated dt for compressible flow ", dt);
        return dt;
    }

    void convert_q_to_primitives()
    {
        std::fill(field_helper.rhof.begin(), field_helper.rhof.end(), 0);
        std::fill(field_helper.uf.begin(), field_helper.uf.end(), TV::Zero());
        std::fill(field_helper.Pf.begin(), field_helper.Pf.end(), 0);

        ExtrapolateOp<T, dim, StorageIndex, XFastestSweep> extrapolate_into_ghost{
            {}, field_helper, q_amb
        };
        ConvertMomentumToU<T, dim, StorageIndex, XFastestSweep> convert_mu_to_u{
            {}, field_helper, gamma
        };
        EOS<T, dim, StorageIndex, XFastestSweep> apply_EOS{
            {}, field_helper, gamma, P_amb
        };

        extrapolate_into_ghost();
        convert_mu_to_u();
        apply_EOS();
    }

    void mark_dof()
    {
        MarkOp<T, dim, StorageIndex, XFastestSweep> mark{ {}, field_helper };
        mark();
    }

    void backup()
    {
        field_helper.q_backup = field_helper.q;
        field_helper.Pf_backup = field_helper.Pf;
        field_helper.cell_type_backup = field_helper.cell_type;
    }

    void advection(T dt, int substep)
    {
        AdvectionOp<T, dim, StorageIndex, XFastestSweep> flux_based_adv{
            {}, field_helper, (T)1 / dx, lowest_rho, lowest_int_e_by_rho
        };

        flux_based_adv(dt, substep);
        convert_q_to_primitives();
    }

    void projection(T dt, int substep)
    {
        PostFixMuOp<T, dim, StorageIndex, XFastestSweep> post_fix_mu{
            {}, field_helper, (T)1 / dx
        };
        ExtrapolateOp<T, dim, StorageIndex, XFastestSweep> extrapolate_into_ghost{
            {}, field_helper, q_amb
        };
        ConvertMomentumToU<T, dim, StorageIndex, XFastestSweep> convert_mu_to_u{
            {}, field_helper, gamma
        };
        PostFixEOp<T, dim, StorageIndex, XFastestSweep> post_fix_E{
            {}, field_helper, (T)1 / dx, lowest_int_e_by_rho
        };

        {
            // get the velocity at the outlets and inlets
            field_helper.moving_Yf_interfaces_override.clear();
            // currently we can just loop all the Yf
            for (const auto& it_mark : field_helper.B_interfaces) {
                auto [I, d, normal] = it_mark;
                int IT = field_helper.interface_type(I)(d);
                if (IT == InterfaceType::GAS_FREE || IT == InterfaceType::INLET_GAS) {
                    auto I_gas = I;
                    I_gas(d) -= 1;
                    T vel = field_helper.uf[field_helper.grid[I_gas].idx](d);
                    field_helper.moving_Yf_interfaces_override.push_back(
                        std::make_tuple(I, d, normal, vel));
                }
                else if (IT == InterfaceType::FREE_GAS || IT == InterfaceType::GAS_INLET) {
                    auto I_gas = I;
                    T vel = field_helper.uf[field_helper.grid[I_gas].idx](d);
                    field_helper.moving_Yf_interfaces_override.push_back(
                        std::make_tuple(I, d, normal, vel));
                }
                else if (IT == InterfaceType::SOLID_GAS) {
                    auto I_solid = I;
                    I_solid(d) -= 1;
                    T vel = field_helper.us[field_helper.grid[I_solid].idx](d);
                    field_helper.moving_Yf_interfaces_override.push_back(
                        std::make_tuple(I, d, normal, vel));
                }
                else if (IT == InterfaceType::GAS_SOLID) {
                    auto I_solid = I;
                    T vel = field_helper.us[field_helper.grid[I_solid].idx](d);
                    field_helper.moving_Yf_interfaces_override.push_back(
                        std::make_tuple(I, d, normal, vel));
                }
            }
        }

        gas_assembler.copy_from_spatial_field(field_helper);
        gas_assembler.assemble(field_helper, field_helper.B_interfaces, field_helper.H_interfaces, field_helper.moving_Yf_interfaces_override, (substep > 0));
        sys_builder.build_and_solve(dt, gas_assembler.global_operators, (substep > 0), cg_it_limit, cg_converge_cretiria);
        gas_assembler.copy_to_spatial_field(field_helper);

        // fix gas qs by conservation form
        post_fix_mu(dt, substep);
        extrapolate_into_ghost();
        convert_mu_to_u();
        post_fix_E(dt, substep);

        // adding the source term
        if (add_source_term) {
            AddSourceTermOp<T, dim, StorageIndex, XFastestSweep> add_source{
                {}, field_helper
            };

            add_source(dt);
        }
    }

    void initialize()
    {
        if (!initialized) {
            // close the simd domain
            field_helper.iterateGridSerial(
                [&](const IV& I) {
                    StorageIndex idx = field_helper.grid[I].idx;
                    if (field_helper.grid.in_bbox(I))
                        field_helper.cell_type[idx] = CellType::GAS;
                    else
                        field_helper.cell_type[idx] = CellType::FREE;
                },
                2);
            // initial backup
            field_helper.cell_type_origin = field_helper.cell_type;
            field_helper.cell_type_backup = field_helper.cell_type;

            initialized = true;
        }

        // reset fluid cell map to the origin
        field_helper.cell_type = field_helper.cell_type_origin;

        mark_dof();
        convert_q_to_primitives();
        backup();
    }

    void restart_prepare()
    {
        field_helper.q_backup = field_helper.q;
        field_helper.cell_type_backup = field_helper.cell_type;
        mark_dof();
        convert_q_to_primitives();
        backup();
    }

    void advance(T dt)
    {
        int RK_lim = (use_RK ? 3 : 1);
        for (int substep = 0; substep < RK_lim; substep++) {
            // Logging::info("RK substep ", substep);
            advection(dt, substep);
            projection(dt, substep);
        }
        mark_dof();
        convert_q_to_primitives();
        backup();
    }

    void dump_output(int frame_num)
    {
    }
};

typedef GasSimulator<double, 3, long long, true> zenCompressSim;
} // namespace ZenEulerGas