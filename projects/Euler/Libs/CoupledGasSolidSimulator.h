#pragma once

#include "AdvectionOp.h"
#include "GasSimulator.h"
#include <Bow/Simulation/MPM/MPMSimulator.h>

namespace Bow {
namespace EulerGas {
template <class T, int dim, class StorageIndex, bool XFastestSweep = false>
class CoupledSimulator
    : public GasSimulator<T, dim, StorageIndex, XFastestSweep>,
      public Bow::MPM::MPMSimulator<T, dim> {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using QArray = Array<T, dim + 2, 1>;
    // base
    using GasBase = GasSimulator<T, dim, StorageIndex, XFastestSweep>;
    using SolidBase = Bow::MPM::MPMSimulator<T, dim>;

    using GasBase::dx;
    using GasBase::output_directory;
    using GasBase::output_vtk;
    using GasBase::P_amb;
    using GasBase::q_amb;

    using GasBase::field_helper;
    using GasBase::gas_assembler;
    using GasBase::solid_assembler;

    using GasBase::advection;
    using GasBase::projection;
    // solid(MPM) data
    using SolidBase::m_C;
    using SolidBase::m_mass;
    using SolidBase::m_V;
    using SolidBase::m_X;
    // ghost matrix data
    Field<T> m_vol;
    Field<T> m_J_ghost;
    Field<T> m_la_ghost;
    SERIALIZATION_REGISTER(m_vol)
    SERIALIZATION_REGISTER(m_J_ghost)
    SERIALIZATION_REGISTER(m_la_ghost)

    using SolidBase::grid;
    using SolidBase::newton_tol;
    using SolidBase::ppc;
    using SolidBase::stress;

    using SolidBase::elasticity_models;
    using SolidBase::plasticity_models;

    using SolidBase::add_boundary_condition;

    bool update_J_ghost_use_ghost_rule = true;
    bool direct_solver = false;

    // if turned on the simulation falls back to MPM sim
    bool stop_gas = false;

    // set dx
    virtual void set_dx(const T dx_) override
    {
        GasBase::set_dx(dx_);
        SolidBase::dx = GasBase::dx;
    }

    // (call it only before running the simulation/init)
    void close_domain_for_MPM()
    {
        // should 2, and 10 (extra layer for mpm particles) be inputs?
        Array<int, dim, 1> bbmin = field_helper.grid.bbmin * 2 - 10,
                           bbmax = field_helper.grid.bbmax * 2 + 10;
        bbmin = bbmin.max(-grid.half_spgrid_size);
        bbmax = bbmax.min(grid.half_spgrid_size);
        TV min_corner = bbmin.matrix().template cast<T>() * dx;
        TV max_corner = bbmax.matrix().template cast<T>() * dx;
        // sync adding bc for solid
        if constexpr (dim == 1) {
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(min_corner), TV(1)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(max_corner), TV(-1)));
        }
        else if constexpr (dim == 2) {
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(min_corner), TV(1, 0)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(min_corner), TV(0, 1)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(max_corner), TV(-1, 0)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(max_corner), TV(0, -1)));
        }
        else if constexpr (dim == 3) {
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(min_corner), TV(1, 0, 0)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(min_corner), TV(0, 1, 0)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(min_corner), TV(0, 0, 1)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(max_corner), TV(-1, 0, 0)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(max_corner), TV(0, -1, 0)));
            add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(
                Geometry::STICKY, gas_to_mpm(max_corner), TV(0, 0, -1)));
        }
        else
            BOW_NOT_IMPLEMENTED
    }

    // constructor
    CoupledSimulator(const T dx_, const Array<int, dim, 1> bbmin_,
        const Array<int, dim, 1> bbmax_,
        const Array<T, dim + 2, 1> q_amb_, const T gamma_ = 1.4,
        const T CFL_ = 0.5)
        : GasBase(dx_, bbmin_, bbmax_, q_amb_, gamma_, CFL_)
    {
        // solid info
        newton_tol = 1e-3;
        SolidBase::dx = GasBase::dx;
        SolidBase::cfl = GasBase::CFL;
        SolidBase::apic = false;
        SolidBase::backward_euler = true;
        GasBase::have_solid = true;
    }

    // define coordinate transform between two grids
    inline TV mpm_to_gas(const TV& pos) { return pos + TV::Ones() * 0.5 * dx; }
    inline TV gas_to_mpm(const TV& pos) { return pos - TV::Ones() * 0.5 * dx; }

    void
    add_particles_by_bbox(std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>> model,
        const TV& min_corner, const TV& max_corner, T density,
        T lambda_ghost, const TV& velocity = TV::Zero(),
        bool random = false, T sparsity = 1)
    {
        T vol = dim == 2 ? dx * dx / 4 : dx * dx * dx / 8;
        T interval = dx / std::pow((T)ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval)
                                      .array()
                                      .round()
                                      .template cast<int>();
        Vector<T, dim> delta_extend = region.template cast<T>() * interval - (max_corner - min_corner);
        Vector<T, dim> region_begin = min_corner - delta_extend * 0.5;
        int start = m_X.size();
        int cnt = 0;
        Bow::MPM::iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = region_begin + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval;
            if (random)
                position += TV::Random() * 0.5 * interval;
            cnt++;
            // generate a random number between 0~1
            // if this is smaller than the sparse
            // then push back this particle
            T rand = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX);
            if (rand < sparsity) {
                m_X.push_back(gas_to_mpm(position));
                m_V.push_back(velocity);
                m_C.push_back(TM::Zero());
                m_mass.push_back(density * vol);
                stress.push_back(TM::Zero());
                // ghost data
                m_vol.push_back(vol);
                m_J_ghost.push_back((T)1);
                m_la_ghost.push_back(lambda_ghost);
            }
        });
        int end = m_X.size();
        // because of the sparsity the volume is differnt
        // corrent the volume
        vol *= (T)cnt / (T)(end - start);
        // fix volume for partivles
        for (int i = start; i < end; i++)
            m_vol[i] = vol;
        model->append(start, end, vol);
    }

    template <typename FT>
    void add_particles_by_bbox_filter(
        std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>> model, const FT& filter,
        const TV& min_corner, const TV& max_corner, T density, T lambda_ghost,
        const TV& velocity = TV::Zero(), bool random = false, T sparsity = 1)
    {
        T vol = dim == 2 ? dx * dx / 4 : dx * dx * dx / 8;
        T interval = dx / std::pow((T)ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval)
                                      .array()
                                      .round()
                                      .template cast<int>();
        Vector<T, dim> delta_extend = region.template cast<T>() * interval - (max_corner - min_corner);
        Vector<T, dim> region_begin = min_corner - delta_extend * 0.5;
        int start = m_X.size();
        int cnt = 0;
        Bow::MPM::iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = region_begin + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval;
            if (random)
                position += TV::Random() * 0.5 * interval;
            if (filter(position)) {
                cnt++;
                // generate a random number between 0~1
                // if this is smaller than the sparse
                // then push back this particle
                T rand = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX);
                if (rand < sparsity) {
                    m_X.push_back(gas_to_mpm(position));
                    m_V.push_back(velocity);
                    m_C.push_back(TM::Zero());
                    m_mass.push_back(density * vol);
                    stress.push_back(TM::Zero());
                    // ghost data
                    m_vol.push_back(vol);
                    m_J_ghost.push_back((T)1);
                    m_la_ghost.push_back(lambda_ghost);
                }
            }
        });
        int end = m_X.size();
        // because of the sparsity the volume is differnt
        // corrent the volume
        vol *= (T)cnt / (T)(end - start);
        // fix volume for partivles
        for (int i = start; i < end; i++)
            m_vol[i] = vol;
        model->append(start, end, vol);
    }

    void add_particles_by_pos_vector(
        std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>> model,
        const Bow::Field<TV>& pos, T vol, T density, T lambda_ghost,
        const TV& velocity = TV::Zero())
    {
        int start = m_X.size();
        for (int i = 0; i < pos.size(); i++) {
            TV position = gas_to_mpm(pos[i]);
            m_X.push_back(position);
            m_V.push_back(velocity);
            m_C.push_back(TM::Zero());
            m_mass.push_back(density * vol);
            stress.push_back(TM::Zero());
            // ghost data
            m_vol.push_back(vol);
            m_J_ghost.push_back((T)1);
            m_la_ghost.push_back(lambda_ghost);
        }
        int end = m_X.size();
        model->append(start, end, vol);
    }

    // void add_particles_by_tet_mesh(
    //     std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>> model,
    //     const std::string mesh_path, T density, T lambda_ghost,
    //     const TV& velocity = TV::Zero())
    // {
    //     BOW_ASSERT_INFO(dim == 3,
    //         "sample by tet mesh is only available for 3D cases");
    //     Field<Vector<T, 3>> X;
    //     Field<Vector<int, 4>> cells;
    //     Bow::IO::read_tet_mesh(mesh_path, X, cells);
    //     std::string vtk_path = output_directory + "tet.vtk";
    //     Bow::IO::write_vtk(vtk_path, X, cells, false);
    //     T total_volume = 0;
    //     for (size_t i = 0; i < cells.size(); i++) {
    //         TV p0 = X[cells[i](0)], p1 = X[cells[i](1)], p2 = X[cells[i](2)],
    //            p3 = X[cells[i](3)];
    //         Matrix<T, 4, 4> A;
    //         A << 1, p0(0), p0(1), p0(2), 1, p1(0), p1(1), p1(2), 1, p2(0), p2(1),
    //             p2(2), 1, p3(0), p3(1), p3(2);
    //         T temp = A.determinant() / (T)6;
    //         total_volume += (temp > 0 ? temp : (-temp));
    //     }
    //     T vol = total_volume / (T)X.size();
    //     std::cout << "Tetmesh: Total volume:" << total_volume
    //               << " Approx num:" << 8 * total_volume / std::pow(dx, dim)
    //               << " Real num:" << X.size() << std::endl;
    //     add_particles_by_pos_vector(model, X, vol, density, lambda_ghost, velocity);
    // }

    virtual T calculate_dt() override
    {
        if (stop_gas) {
            T dt = SolidBase::calculate_dt();
            dt = std::min(std::max(GasBase::dt_min, dt), GasBase::dt_max);
            return dt;
        }
        else {
            T dt = min(GasBase::calculate_dt(), SolidBase::calculate_dt());
            dt = std::min(std::max(GasBase::dt_min, dt), GasBase::dt_max);
            return dt;
        }
    }

    void MPM_to_solid_grid_in_coupling_sys()
    {
        // a virtual p2g to update the grid's types
        grid.sortParticles(m_X, dx);
        Bow::MPM::ParticlesToGridOp<T, dim, false> p2g{
            {}, m_X, m_V, m_mass, m_C, stress, grid, dx, 0
        };
        p2g();
        // MPM to solid grid
        T inv_dx = (T)1 / dx;
        // reset solid grid data
        std::fill(field_helper.rhos.begin(), field_helper.rhos.end(), 0);
        std::fill(field_helper.us.begin(), field_helper.us.end(), TV::Zero());
        std::fill(field_helper.Ps.begin(), field_helper.Ps.end(), 0);
        std::fill(field_helper.las.begin(), field_helper.las.end(), 0);
        std::fill(field_helper.ghost_volume.begin(),
            field_helper.ghost_volume.end(), 0);
        std::fill(field_helper.ghost_volume_center.begin(),
            field_helper.ghost_volume_center.end(), 0);
        // reset fluid cell map to the origin
        field_helper.cell_type = field_helper.cell_type_origin;
        // copy solid's rho and v to grid
        grid.iterateWholeGridSerial(
            [&](Vector<int, dim> I, Bow::MPM::GridState<T, dim>& g) {
                // dont fall outof the bbox
                if (!(grid[I].v_and_m(dim) > 0) || grid[I].idx < 0 || !field_helper.grid.in_bbox(I, 1))
                    return;
                if (field_helper.grid[I].idx < 0)
                    return;
                int idx = field_helper.grid[I].idx;
                Vector<T, dim + 1> v_and_m = grid[I].v_and_m;
                field_helper.rhos[idx] = v_and_m(dim) * std::pow(inv_dx, dim);
                field_helper.us[idx] = v_and_m.template head<dim>();
                if (field_helper.cell_type_origin[idx] == CellType::GAS)
                    field_helper.cell_type[idx] = CellType::SOLID;
            });
        // customed P2G for ghost Pressure
        grid.colored_for([&](int p) {
            // here +0.5 is the offet from solid to gas field
            TV Xp_in_gas_to_corner(m_X[p] + 0.5 * dx * TV::Ones());
            TV Xp_in_gas_to_center(m_X[p]);
            BSplineWeights<T, dim, 1> spline_to_corner(Xp_in_gas_to_corner, dx);
            BSplineWeights<T, dim, 2> spline_to_center(Xp_in_gas_to_center, dx);
            T V0 = m_vol[p];
            T Jp_ghost = m_J_ghost[p];
            T lambda_ghost = m_la_ghost[p];
            T P_ghost = -lambda_ghost * (Jp_ghost - 1.0) + P_amb;
            grid.iterateKernel(spline_to_corner, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, Bow::MPM::GridState<T, dim>& g) {
                // dont fall outof the bbox
                if (!field_helper.grid.in_bbox(node, 1))
                    return;
                if (g.idx < 0)
                    return;
                if (field_helper.grid[node].idx < 0)
                    return;
                int idx = field_helper.grid[node].idx;
                field_helper.Ps[idx] += V0 * Jp_ghost * P_ghost * w;
                field_helper.las[idx] += V0 * Jp_ghost * lambda_ghost * w;
                field_helper.ghost_volume[idx] += V0 * Jp_ghost * w;
            });
            grid.iterateKernel(spline_to_center, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, Bow::MPM::GridState<T, dim>& g) {
                // dont fall outof the bbox
                if (!field_helper.grid.in_bbox(node, 1))
                    return;
                if (g.idx < 0)
                    return;
                if (field_helper.grid[node].idx < 0)
                    return;
                int idx = field_helper.grid[node].idx;
                field_helper.ghost_volume_center[idx] += V0 * Jp_ghost * w;
            });
        });
        field_helper.iterateGridSerial(
            [&](const Vector<int, dim>& I) {
                if (!(field_helper.ghost_volume[field_helper.grid[I].idx] > 0) || field_helper.grid[I].idx < 0)
                    return;
                int idx = field_helper.grid[I].idx;
                T ghost_volume = field_helper.ghost_volume[idx];
                field_helper.las[idx] /= ghost_volume;
                field_helper.Ps[idx] /= ghost_volume;
            },
            1);
        // backup Ps for assembling
        field_helper.Ps_backup = field_helper.Ps;
    }

    void mark_fluid_cell_with_solid_info()
    {
        MPM_to_solid_grid_in_coupling_sys();
        {
            // get rid of the partial solid cells whose pressure node quad is not
            // complete (more precise version: cut cell)
            field_helper.iterateGridSerial(
                [&](const Vector<int, dim>& I) {
                    auto& g = field_helper.grid[I];
                    if (field_helper.cell_type[g.idx] == CellType::SOLID) {
                        // new shrink
                        bool have_full_pressure_quad = true;
                        field_helper.iterateKernel(
                            [&](const Vector<int, dim>& I,
                                const Vector<int, dim>& adj_I) {
                                int adj_idx = field_helper.grid[adj_I].idx;
                                if (field_helper.Ps[adj_idx] == 0)
                                    have_full_pressure_quad = false;
                            },
                            I, 0, 2);
                        if (!have_full_pressure_quad)
                            field_helper.cell_type[g.idx] = field_helper.cell_type_origin[g.idx];
                    }
                },
                1);
        }
        {
            // fix_newly_released_gas_cell_qs
            field_helper.iterateGridParallel(
                [&](const Vector<int, dim>& I) {
                    int idx = field_helper.grid[I].idx;
                    if ((field_helper.cell_type[idx] == CellType::GAS) && (field_helper.cell_type_backup[idx] == CellType::SOLID)) {
                        QArray sum_Qs = QArray::Zero();
                        int sum_Qs_n = 0;
                        field_helper.iterateKernel(
                            [&](const Vector<int, dim>& I,
                                const Vector<int, dim>& adj_I) {
                                // dont fall outof the bbox
                                if (!field_helper.grid.in_bbox(adj_I, 0))
                                    return;
                                if (field_helper.grid[adj_I].idx < 0)
                                    return;
                                int adj_idx = field_helper.grid[adj_I].idx;
                                if (field_helper.cell_type_backup[adj_idx] == CellType::GAS) {
                                    sum_Qs += field_helper.q_backup[adj_idx];
                                    sum_Qs_n++;
                                }
                            },
                            I, -1, 2);
                        if (sum_Qs_n > 0) {
                            sum_Qs /= (T)sum_Qs_n;
                            field_helper.q[idx] = sum_Qs;
                        }
                        else {
                            Logging::warn(
                                "a gas cell released from solid has no gas neighbor");
                            field_helper.q[idx] = q_amb;
                        }
                    }
                },
                1);
        }
        GasBase::mark_dof();
        GasBase::convert_q_to_primitives();
        GasBase::backup();
    }

    void project_solid_J_ghost_to_MPM_particles()
    {
        BOW_TIMER_FLAG("update MPM ghost J");
        if (update_J_ghost_use_ghost_rule) {
            // using ghost pressure
            // before particle advection
            grid.parallel_for([&](int p) {
                TV Xp_in_gas_to_corner(m_X[p] + 0.5 * dx * TV::Ones());
                BSplineWeights<T, dim, 1> spline(Xp_in_gas_to_corner, dx);
                T P_ghost = 0;
                T tot_w = 0;
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, Bow::MPM::GridState<T, dim>& g) {
                    if (g.idx < 0 || !field_helper.grid.in_bbox(node, 1))
                        return;
                    int idx = field_helper.grid[node].idx;
                    T dP = field_helper.Ps[idx];
                    if (dP != 0) {
                        P_ghost += dP * w;
                        tot_w += w;
                    }
                });
                if (tot_w > 0)
                    P_ghost /= tot_w;
                else
                    P_ghost = P_amb;
                m_J_ghost[p] = std::max(0.3, std::min(1 - (P_ghost - P_amb) / m_la_ghost[p], 3.0));
            });
        }
        else {
            // using new deformation
            // after particle advection
            Field<TM> m_Fs;
            m_Fs.resize(m_X.size(), TM::Identity());
            for (auto& model : elasticity_models)
                model->collect_strain(m_Fs);
            tbb::parallel_for(0, (int)m_X.size(), [&](int p) {
                m_J_ghost[p] = std::max(0.3, std::min(m_Fs[p].determinant(), 3.0));
            });
        }
    }

    void initialize() override
    {
        close_domain_for_MPM();
        mark_fluid_cell_with_solid_info();
    }

    void restart_prepare() override
    {
        GasBase::backup();
        mark_fluid_cell_with_solid_info();
    }

    void advance(T dt) override
    {
        if (stop_gas)
            SolidBase::advance(dt);
        else {
            GasBase::have_solid = m_X.size() > 0;
            if (GasBase::have_solid) {
                {
                    // update in linear coupling system
                    // 3rd order TVD-RK, only used in advection when there involves MPM
                    int RK_lim = (GasBase::use_RK ? 3 : 1);
                    for (int substep = 0; substep < RK_lim; substep++) {
                        Logging::info("RK substep ", substep);
                        advection(dt, substep);
                    }
                    projection(dt, 0);
                    GasBase::backup();
                }
                {
                    // update in Newton step
                    // get the MPM grid velocity after coupling
                    field_helper.iterateGridParallel(
                        [&](Vector<int, dim> I) {
                            if (field_helper.grid[I].idus >= 0)
                                grid[I].v_and_m.template head<dim>() = field_helper.us[field_helper.grid[I].idx];
                        },
                        1);
                    if (update_J_ghost_use_ghost_rule)
                        project_solid_J_ghost_to_MPM_particles();
                    // standard MPM grid update, g2p, particle advection and material
                    // evolve
                    SolidBase::grid_update(dt, direct_solver);
                    SolidBase::g2p(dt);
                    if (!update_J_ghost_use_ghost_rule)
                        project_solid_J_ghost_to_MPM_particles();
                }
                mark_fluid_cell_with_solid_info();
            }
            else
                GasBase::advance(dt);
        }
    }

    virtual void dump_vtk(int frame_num) override
    {
        return;
    }

    void dump_output(int frame_num) override
    {
        return;
    }
};
}
} // namespace Bow::EulerGas