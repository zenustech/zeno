#pragma once

#include "AdvectionOp.h"
#include "BasicOp.h"
#include "Builder.h"
#include "GasAssembler.h"
#include "ProjectionOp.h"
#include "SolidAssembler.h"
//#include <Bow/IO/partio.h>
#include <Bow/Simulation/PhysicallyBasedSimulator.h>
#include <Bow/Utils/Serialization.h>

namespace Bow {
namespace EulerGas {

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
class GasSimulator : virtual public PhysicallyBasedSimulator<T, dim> {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using QArray = Array<T, dim + 2, 1>;
    // base
    using Base = PhysicallyBasedSimulator<T, dim>;
    using Base::end_frame;
    using Base::frame_dt;
    using Base::output_directory;
    using Base::suggested_dt;

    T dx;

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

    // fields wrapper
    FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& field_helper;
    // linear projection sys
    // assembler and builder
    Bow::LinearProjection::GasAssembler<
        T, dim, StorageIndex, XFastestSweep,
        Bow::LinearProjection::BSplineDegree::B1B0>
        gas_assembler;
    Bow::LinearProjection::SolidAssembler<
        T, dim, StorageIndex, XFastestSweep,
        Bow::LinearProjection::BSplineDegree::B2B1>
        solid_assembler;
    Bow::LinearProjection::Builder<T, dim, StorageIndex> sys_builder;
    Bow::LinearProjection::CoupledBuilder<T, dim, StorageIndex>
        coupling_sys_builder;

    // set ambient value
    void set_ambient(const Array<T, dim + 2, 1> q_amb_, const T gamma_)
    {
        BOW_ASSERT_INFO(q_amb_(0) > 0, "defined negative density for ambient");
        BOW_ASSERT_INFO(q_amb_(1) > 0, "defined negative total energy for ambient");
        T int_e = q_amb_(1) - 0.5 * (q_amb_.template tail<dim>() * q_amb_.template tail<dim>()).sum() / q_amb_(0);
        BOW_ASSERT_INFO(int_e > 0, "defined negative internal energy for ambient");
        q_amb = q_amb_;
        P_amb = Bow::ConstitutiveModel::IdealGas::get_pressure_from_int_energy(
            gamma_, int_e);
        gamma = gamma_;
        gas_assembler.gamma = gamma;
        solid_assembler.P_amb = P_amb;
        //std::fill(field_helper.q.begin(), field_helper.q.end(), q_amb);
        //std::fill(field_helper.q_backup.begin(), field_helper.q_backup.end(),
        //q_amb);
        // threshold for clamping
        lowest_rho = clamp_ratio * q_amb(0);
        lowest_int_e_by_rho = clamp_ratio * int_e / q_amb(0);
    }

    // set dx
    virtual void set_dx(const T dx_)
    {
        dx = dx_;
        sys_builder.dx = dx_;
        coupling_sys_builder.dx = dx_;
    }

    void prepare_final_dir(std::string raw_dir)
    {
        // append any information to the output dir
        // append resolution
        Array<int, dim, 1> extend = field_helper.grid.bbmax - field_helper.grid.bbmin;
        output_directory = raw_dir;
        for (int d = 0; d < dim; d++)
            output_directory += "_" + std::to_string(extend(d));
        output_directory += "/";

        Bow::FileSystem::create_path(output_directory);
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
    GasSimulator(const T dx_, const Array<int, dim, 1> bbmin_,
        const Array<int, dim, 1> bbmax_,
        const Array<T, dim + 2, 1> q_amb_, FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& _field_helper,
        const T gamma_ = 1.4,
        const T CFL_ = 0.5)
        : dx(dx_), field_helper(_field_helper), sys_builder(dx_), coupling_sys_builder(dx_), gamma(gamma_), CFL(CFL_)
    {
        set_ambient(q_amb_, gamma_);
        // close the simd domain
        field_helper.iterateGridSerial(
            [&](const IV& I) {
                StorageIndex idx = field_helper.grid[I].idx;
                if ((I.array() - bbmin_ < 0).any() || (I.array() - bbmax_ >= 0).any())
                    field_helper.cell_type[idx] = CellType::FREE;
                else
                    field_helper.cell_type[idx] = CellType::GAS;
            },
            2);
        // initial backup
        field_helper.cell_type_origin = field_helper.cell_type;
        field_helper.cell_type_backup = field_helper.cell_type;
    }
    // constructor
    GasSimulator(const T dx_, const Array<int, dim, 1> bbmin_,
        const Array<int, dim, 1> bbmax_,
        FieldHelperDense<T, dim, StorageIndex, XFastestSweep>& _field_helper,
        const Array<T, dim + 2, 1> q_amb_, const T gamma_ = 1.4,
        const T CFL_ = 0.5)
        : dx(dx_), field_helper(_field_helper), sys_builder(dx_), coupling_sys_builder(dx_), gamma(gamma_), CFL(CFL_)
    {
        set_ambient(q_amb_, gamma_);
        // close the simd domain
        field_helper.iterateGridSerial(
            [&](const IV& I) {
                StorageIndex idx = field_helper.grid[I].idx;
                if ((I.array() - bbmin_ < 0).any() || (I.array() - bbmax_ >= 0).any())
                    field_helper.cell_type[idx] = CellType::FREE;
                else
                    field_helper.cell_type[idx] = CellType::GAS;
            },
            2);
        // initial backup
        field_helper.cell_type_origin = field_helper.cell_type;
        field_helper.cell_type_backup = field_helper.cell_type;
    }

    virtual T calculate_dt() override
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
        Logging::debug("sum_abs_a_by_dx ", sum_abs_a_by_dx);
        Logging::debug("sum_abs_v_by_dx ", sum_abs_v_by_dx);
        Logging::info("dt_no_clamp ", dt);
        if (dt < dt_min) {
            Logging::error("too little dt ", dt);
            exit(1); // I have no idea why it doesnot quit here
        }
        dt = std::min(std::max(dt_min, dt), dt_max);
        Logging::info("calculated dt for compressible flow ", dt);
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
            }
        }
        {
            // get the velocity at the outlets and inlets
            field_helper.moving_Ys_interfaces_override.clear();
            // currently we can just loop all the Yf
            for (const auto& it_mark : field_helper.Bs_interfaces) {
                auto [I, d, normal] = it_mark;
                int IT = field_helper.interface_type(I)(d);
                if (IT == InterfaceType::SOLID_FREE || IT == InterfaceType::SOLID_INLET) {
                    auto I_solid = I;
                    I_solid(d) -= 1;
                    T vel = field_helper.us[field_helper.grid[I_solid].idx](d);
                    field_helper.moving_Ys_interfaces_override.push_back(
                        std::make_tuple(I, d, normal, vel));
                }
                else if (IT == InterfaceType::FREE_SOLID || IT == InterfaceType::INLET_SOLID) {
                    auto I_solid = I;
                    T vel = field_helper.us[field_helper.grid[I_solid].idx](d);
                    field_helper.moving_Ys_interfaces_override.push_back(
                        std::make_tuple(I, d, normal, vel));
                }
            }
        }

        gas_assembler.copy_from_spatial_field(field_helper);
        gas_assembler.assemble(
            field_helper, field_helper.B_interfaces, field_helper.H_interfaces,
            field_helper.moving_Yf_interfaces_override, (substep > 0));
        if (have_solid) {
            solid_assembler.copy_from_spatial_field(field_helper);
            solid_assembler.assemble(
                field_helper, field_helper.Bs_interfaces, field_helper.Hs_interfaces,
                field_helper.moving_Ys_interfaces_override, (substep > 0));
            coupling_sys_builder.coupled_build_and_solve(
                dt, gas_assembler.global_operators, solid_assembler.global_operators,
                (substep > 0), cg_it_limit, cg_converge_cretiria);
            // fix solid velocity by weakform solution
            solid_assembler.fix_velocity_by_projection(dt / dx);
            // the gas velocity is not fixed here
            // gas field's velocity will be overwrtten later, so the copy of velocity
            // here won't cause any error however we need the gas pressure field later
            gas_assembler.copy_to_spatial_field(field_helper);
            solid_assembler.copy_to_spatial_field(field_helper);
        }
        else {
            sys_builder.build_and_solve(dt, gas_assembler.global_operators,
                (substep > 0), cg_it_limit,
                cg_converge_cretiria);
            gas_assembler.copy_to_spatial_field(field_helper);
        }

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

    void initialize() override
    {
        if (!initialized) {

            // test once
            Bow::Vector<int, dim> center_I = ((field_helper.grid.bbmin + field_helper.grid.bbmax) / 2).matrix();
            float r = 20;
            int extend = 0;
            for (int i = field_helper.grid.bbmin(0) - extend; i < field_helper.grid.bbmax(0) + extend; i++)
                for (int j = field_helper.grid.bbmin(1) - extend; j < field_helper.grid.bbmax(1) + extend; j++)
                    for (int k = field_helper.grid.bbmin(2) - extend; k < field_helper.grid.bbmax(2) + extend;
                         k++) {
                        Bow::Vector<int, dim> I(i, j, k);
                        if ((I - center_I).template cast<T>().norm() < r) {
                            int idx = field_helper.grid[I].idx;
                            field_helper.q[idx] *= 10;
                        }
                    }
            // field_helper.iterateGridSerial([&](const Bow::Vector<int, dim>& I) {
            //     if ((I - center_I).template cast<T>().norm() < r) {
            //         int idx = field_helper.grid[I].idx;
            //         field_helper.q[idx] *= 10;
            //     }
            // },0);
            initialized = true;
        }

        mark_dof();
        convert_q_to_primitives();
        int idx = field_helper.grid[Bow::Vector<int, dim>::Zero()].idx;
        std::cout << "center q " << field_helper.q[idx] << std::endl;
        backup();
    }

    void restart_prepare() override
    {
        field_helper.q_backup = field_helper.q;
        field_helper.cell_type_backup = field_helper.cell_type;
        mark_dof();
        convert_q_to_primitives();
        backup();
    }

    void advance(T dt) override
    {
        int RK_lim = (use_RK ? 3 : 1);
        for (int substep = 0; substep < RK_lim; substep++) {
            Logging::info("RK substep ", substep);
            advection(dt, substep);
            projection(dt, substep);
        }
        mark_dof();
        convert_q_to_primitives();
        backup();
    }

    void get_advanced_visual_attr()
    {
        // get schlieren
        // TODO change the interpolation to WENO (consistent with simd)?
        {
            std::fill(field_helper.schlieren.begin(), field_helper.schlieren.end(),
                TV::Zero());
            field_helper.iterateGridSerial(
                [&](const Vector<int, dim> I) {
                    int idx = field_helper.grid[I].idx;
                    TV grad_rho = TV::Zero();
                    if (field_helper.cell_type[idx] == CellType::GAS) {
                        T rho_c = field_helper.rhof[idx];
                        for (int d = 0; d < dim; d++) {
                            // look front
                            Vector<int, dim> I_f = I;
                            I_f(d) += 1;
                            int idx_f = field_helper.grid[I_f].idx;
                            T rho_f = (field_helper.cell_type[idx_f] == CellType::GAS
                                    ? field_helper.rhof[idx_f]
                                    : rho_c);
                            // look back
                            Vector<int, dim> I_b = I;
                            I_b(d) -= 1;
                            int idx_b = field_helper.grid[I_b].idx;
                            T rho_b = (field_helper.cell_type[idx_b] == CellType::GAS
                                    ? field_helper.rhof[idx_b]
                                    : rho_c);
                            // grad
                            grad_rho(d) = (rho_f - rho_b) / 2 / dx;
                        }
                        field_helper.schlieren[idx] = grad_rho;
                    }
                },
                0);
        }
        // get shawdow_graph
        {
            std::fill(field_helper.shawdow_graph.begin(),
                field_helper.shawdow_graph.end(), 0);
            field_helper.iterateGridSerial(
                [&](const Vector<int, dim> I) {
                    int idx = field_helper.grid[I].idx;
                    if (field_helper.cell_type[idx] == CellType::GAS) {
                        T rho_c = field_helper.rhof[idx];
                        T laplacian_rho = -rho_c * (T)dim * 2;
                        for (int d = 0; d < dim; d++) {
                            // look front
                            Vector<int, dim> I_f = I;
                            I_f(d) += 1;
                            int idx_f = field_helper.grid[I_f].idx;
                            laplacian_rho += (field_helper.cell_type[idx_f] == CellType::GAS
                                    ? field_helper.rhof[idx_f]
                                    : rho_c);
                            // look back
                            Vector<int, dim> I_b = I;
                            I_b(d) -= 1;
                            int idx_b = field_helper.grid[I_b].idx;
                            laplacian_rho += (field_helper.cell_type[idx_b] == CellType::GAS
                                    ? field_helper.rhof[idx_b]
                                    : rho_c);
                        }
                        laplacian_rho /= (dx * dx);
                        field_helper.shawdow_graph[idx] = laplacian_rho;
                    }
                },
                0);
        }
    }

    virtual void dump_vtk(int frame_num)
    {
        return;
    }

    void dump_output(int frame_num) override
    {
        get_advanced_visual_attr();
        BOW_TIMER_FLAG("IO");
        std::string output_file = output_directory + "gas_" + std::to_string(frame_num) + ".ply";
        field_helper.save_ply(output_file, dx, gamma);
        // now just output vtk along with ply, later seperate them as options TODO
        if (output_vtk)
            dump_vtk(frame_num);
    }
};
typedef GasSimulator<double, 3, long long, true> zenCompressSim;
}
} // namespace Bow::EulerGas