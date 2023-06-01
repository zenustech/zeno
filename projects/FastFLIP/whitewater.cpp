#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <zeno/VDBGrid.h>
#include <zeno/ZenoInc.h>
#include <zeno/zeno.h>

namespace zeno {
struct WhitewaterSource : INode {
    float eps = 10 * std::numeric_limits<float>::epsilon();

    bool checkAngle(const openvdb::Vec3f vel, const openvdb::Vec3f norm, float max_angle) {
        float dotProduct = vel.dot(norm);
        float cosine = dotProduct / std::max(vel.length() * norm.length(), eps);
        float angle = std::acos(cosine) * 180.f / M_PI;
        return angle < max_angle;
    }

    float clamp_map(float val, vec2f range) {
        return (std::min(val, range[1]) - std::min(val, range[0])) / std::max(range[1] - range[0], eps);
    }

    void apply() override {
        auto pars = std::make_shared<PrimitiveObject>();
        if (has_input("Primitive")) {
            pars = get_input<PrimitiveObject>("Primitive");
        }
        auto dt = get_input2<float>("dt");
        auto Lifespan = get_input2<float>("Lifespan");
        auto &Liquid_sdf = get_input<VDBFloatGrid>("LiquidSDF")->m_grid;
        auto &Solid_sdf = get_input<VDBFloatGrid>("SolidSDF")->m_grid;
        auto &Velocity = get_input<VDBFloat3Grid>("Velocity")->m_grid;
        auto &Pre_vel = get_input<VDBFloat3Grid>("PreVelocity")->m_grid;

        auto &par_pos = pars->verts.values;
        auto &par_vel = pars->add_attr<vec3f>("vel");
        auto &par_life = pars->add_attr<float>("life");
        // pars->verts.values.push_back(vec3f{});
        // vel.push_back();

        auto limit_depth = get_input2<float>("LimitDepth");
        auto speed_range = get_input2<vec2f>("SpeedRange");
        auto curv_emit = get_input2<float>("EmitFromCurvature");
        auto max_angle = get_input2<float>("MaxVelocityAngle");
        auto curv_range = get_input2<vec2f>("CurvatureRange");
        auto acc_emit = get_input2<float>("EmitFromAcceleration");
        auto acc_range = get_input2<vec2f>("AccelerationRange");
        auto vor_emit = get_input2<float>("EmitFromVorticity");
        auto vor_range = get_input2<vec2f>("VorticityRange");

        float dx = static_cast<float>(Velocity->voxelSize()[0]);

        openvdb::Vec3fGrid::Ptr Normal, Vorticity;
        openvdb::FloatGrid::Ptr Curvature;
        if (curv_emit > 0) {
            Normal = openvdb::tools::gradient(*Liquid_sdf);
            Curvature = openvdb::tools::meanCurvature(*Liquid_sdf);
        }
        if (vor_emit > 0) {
            Vorticity = openvdb::tools::curl(*Velocity);
        }
        auto liquid_sdf_axr = Liquid_sdf->getConstUnsafeAccessor();
        auto solid_sdf_axr = Solid_sdf->getConstUnsafeAccessor();
        auto vel_axr = Velocity->getConstUnsafeAccessor();
        auto pre_vel_axr = Pre_vel->getConstUnsafeAccessor();
        auto norm_axr = Normal->getConstUnsafeAccessor();
        auto curv_axr = Curvature->getConstUnsafeAccessor();
        auto vor_axr = Vorticity->getConstUnsafeAccessor();

        std::random_device rd;
        std::mt19937 gen(rd());

        for (auto iter = Liquid_sdf->cbeginValueOn(); iter.test(); ++iter) {
            auto icoord = iter.getCoord();
            auto wcoord = Liquid_sdf->indexToWorld(icoord);

            float m_sdf = *iter;
            float m_solid_sdf = openvdb::tools::BoxSampler::sample(solid_sdf_axr, Solid_sdf->worldToIndex(wcoord));
            if (m_sdf < limit_depth || m_solid_sdf < 0)
                continue;

            float generates = 0;

            openvdb::Vec3f m_vel = openvdb::tools::StaggeredBoxSampler::sample(vel_axr, Velocity->worldToIndex(wcoord));
            if (curv_emit > 0) {
                float m_curv = openvdb::tools::BoxSampler::sample(curv_axr, Curvature->worldToIndex(wcoord));
                openvdb::Vec3f m_norm = openvdb::tools::BoxSampler::sample(norm_axr, Normal->worldToIndex(wcoord));
                if (!checkAngle(m_vel, m_norm, max_angle))
                    m_curv = 0;
                generates += curv_emit * clamp_map(m_curv, curv_range);
            }
            if (acc_emit > 0) {
                openvdb::Vec3f m_pre_vel =
                    openvdb::tools::StaggeredBoxSampler::sample(pre_vel_axr, Pre_vel->worldToIndex(wcoord));
                auto m_acc_vec = (m_vel - m_pre_vel) / dt;
                float m_acc = m_acc_vec.length();
                generates += acc_emit * clamp_map(m_acc, acc_range);
            }
            if (vor_emit > 0) {
                openvdb::Vec3f m_vor_vec = openvdb::tools::BoxSampler::sample(vor_axr, Vorticity->worldToIndex(wcoord));
                float m_vor = m_vor_vec.length();
                generates += vor_emit * clamp_map(m_vor, vor_range);
            }

            float m_speed = m_vel.length();
            generates *= clamp_map(m_speed, speed_range);

            int m_new_pars = std::round(generates);

            std::uniform_real_distribution<float> disX(wcoord[0] - 0.5f * dx, wcoord[0] + 0.5f * dx);
            std::uniform_real_distribution<float> disY(wcoord[1] - 0.5f * dx, wcoord[1] + 0.5f * dx);
            std::uniform_real_distribution<float> disZ(wcoord[2] - 0.5f * dx, wcoord[2] + 0.5f * dx);

            for (int n = 0; n < m_new_pars; ++n) {
                openvdb::Vec3f m_par_pos{disX(gen), disY(gen), disZ(gen)};
                openvdb::Vec3f m_par_vel =
                    openvdb::tools::StaggeredBoxSampler::sample(vel_axr, Velocity->worldToIndex(m_par_pos));
                par_pos.push_back(other_to_vec<3>(m_par_pos));
                par_vel.push_back(other_to_vec<3>(m_par_vel));
                par_life.push_back(Lifespan);
            }
        }

        set_output("Primitive", pars);
    }
};

ZENDEFNODE(WhitewaterSource, {/* inputs: */
                              {"Primitive",
                               {"float", "dt", "0.04"},
                               {"float", "Lifespan", "0.8"},
                               "LiquidSDF",
                               "SolidSDF",
                               "Velocity",
                               "PreVelocity",
                               {"float", "LimitDepth", "-1"},
                               {"vec2f", "SpeedRange", "0, 1"},
                               {"float", "EmitFromCurvature", "0"},
                               {"float", "MaxVelocityAngle", "45"},
                               {"vec2f", "CurvatureRange", "0, 1"},
                               {"float", "EmitFromAcceleration", "0"},
                               {"vec2f", "AccelerationRange", "0, 1"},
                               {"float", "EmitFromVorticity", "0"},
                               {"vec2f", "VorticityRange", "0, 1"}},
                              /* outputs: */
                              {"Primitive"},
                              /* params: */
                              {},
                              /* category: */
                              {"FLIPSolver"}});

struct WhitewaterSolver : INode {
    void apply() override {
        auto pars = get_input<PrimitiveObject>("Primitive");
        auto dt = get_input2<float>("dt");
        auto &Liquid_sdf = get_input<VDBFloatGrid>("LiquidSDF")->m_grid;
        auto &Solid_sdf = get_input<VDBFloatGrid>("SolidSDF")->m_grid;
        auto &Velocity = get_input<VDBFloat3Grid>("Velocity")->m_grid;

        auto gravity = vec_to_other<openvdb::Vec3f>(get_input2<vec3f>("Gravity"));
        auto air_drag = get_input2<float>("AirDrag");
        auto buoyancy = get_input2<float>("Buoyancy");

        float dx = static_cast<float>(Velocity->voxelSize()[0]);

        auto &par_pos = pars->verts.values;
        auto &par_vel = pars->attr<vec3f>("vel");

        auto Normal = openvdb::tools::gradient(*Solid_sdf);

        auto liquid_sdf_axr = Liquid_sdf->getConstUnsafeAccessor();
        auto solid_sdf_axr = Solid_sdf->getConstUnsafeAccessor();
        auto vel_axr = Velocity->getConstUnsafeAccessor();
        auto norm_axr = Normal->getConstUnsafeAccessor();

#pragma omp parallel for
        for (size_t idx = 0; idx < pars->size(); ++idx) {
            auto m_vel = vec_to_other<openvdb::Vec3f>(par_vel[idx]);
            auto wcoord = vec_to_other<openvdb::Vec3f>(par_pos[idx]);
            float m_liquid_sdf = openvdb::tools::BoxSampler::sample(liquid_sdf_axr, Liquid_sdf->worldToIndex(wcoord));
            if (m_liquid_sdf > dx) {
                // spray
                m_vel += gravity * dt - air_drag * m_vel;
            } else if (m_liquid_sdf < -dx) {
                // bubble
                auto m_liquid_vel =
                    openvdb::tools::StaggeredBoxSampler::sample(vel_axr, Velocity->worldToIndex(wcoord));
                m_vel += -buoyancy * gravity * dt + air_drag * (m_liquid_vel - m_vel);
            } else {
                // foam
                m_vel = openvdb::tools::StaggeredBoxSampler::sample(vel_axr, Velocity->worldToIndex(wcoord));
            }
            auto wcoord_new = wcoord + dt * m_vel;
            float m_solid_sdf = openvdb::tools::BoxSampler::sample(solid_sdf_axr, Solid_sdf->worldToIndex(wcoord));
            if (m_solid_sdf < 0) {
                auto m_norm = openvdb::tools::BoxSampler::sample(norm_axr, Normal->worldToIndex(wcoord));
                m_norm.normalize();
                m_vel = -m_solid_sdf * m_norm / dt;
                wcoord_new += m_vel * dt;
            }

            par_pos[idx] = other_to_vec<3>(wcoord_new);
            par_vel[idx] = other_to_vec<3>(m_vel);
        }
    }
};

ZENDEFNODE(WhitewaterSolver, {/* inputs: */
                              {"Primitive",
                               {"float", "dt", "0.04"},
                               "LiquidSDF",
                               "SolidSDF",
                               "Velocity",
                               {"vec3f", "Gravity", "0, -9.8, 0"},
                               {"float", "AirDrag", "0.05"},
                               {"float", "Buoyancy", "100"}},
                              /* outputs: */
                              {"Primitive"},
                              /* params: */
                              {},
                              /* category: */
                              {"FLIPSolver"}});

} // namespace zeno