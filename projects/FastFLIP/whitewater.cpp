#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <thread>
#include <zeno/VDBGrid.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>

namespace zeno {
struct WhitewaterSource : INode {
    static constexpr float eps = 10 * std::numeric_limits<float>::epsilon();

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

        auto &par_pos = pars->verts.values;
        auto &par_vel = pars->add_attr<vec3f>("vel");
        auto &par_life = pars->add_attr<float>("life");
        // pars->verts.values.push_back(vec3f{});
        // vel.push_back();

        auto limit_depth = get_input2<vec2f>("LimitDepth");
        auto speed_range = get_input2<vec2f>("SpeedRange");
        auto curv_emit = get_input2<float>("EmitFromCurvature");
        auto max_angle = get_input2<float>("MaxVelocityAngle");
        auto curv_range = get_input2<vec2f>("CurvatureRange");
        auto acc_emit = get_input2<float>("EmitFromAcceleration");
        auto acc_range = get_input2<vec2f>("AccelerationRange");
        auto vor_emit = get_input2<float>("EmitFromVorticity");
        auto vor_range = get_input2<vec2f>("VorticityRange");

        float dx = static_cast<float>(Velocity->voxelSize()[0]);

        openvdb::Vec3fGrid::Ptr Normal, Vorticity, Pre_vel;
        openvdb::FloatGrid::Ptr Curvature;
        if (curv_emit > eps) {
            Normal = openvdb::tools::gradient(*Liquid_sdf);
            Curvature = openvdb::tools::meanCurvature(*Liquid_sdf);
        }
        if (acc_emit > eps) {
            Pre_vel = get_input<VDBFloat3Grid>("PreVelocity")->m_grid;
        }
        if (vor_emit > eps) {
            Vorticity = openvdb::tools::curl(*Velocity);
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        using MapT = std::map<std::thread::id, std::vector<vec3f>>;
        MapT new_pars_pos, new_pars_vel;
        std::mutex mutex;

        auto particle_emitter = [&](openvdb::FloatTree::LeafNodeType &leaf, openvdb::Index leafpos) {
            auto liquid_sdf_axr = Liquid_sdf->getConstUnsafeAccessor();
            auto solid_sdf_axr = Solid_sdf->getConstUnsafeAccessor();
            auto vel_axr = Velocity->getConstUnsafeAccessor();

            typename MapT::iterator posIter, velIter;
            {
                std::lock_guard<std::mutex> lk(mutex);
                bool tag;
                std::tie(posIter, tag) =
                    new_pars_pos.insert(std::make_pair(std::this_thread::get_id(), std::vector<vec3f>{}));
                std::tie(velIter, tag) =
                    new_pars_vel.insert(std::make_pair(std::this_thread::get_id(), std::vector<vec3f>{}));
            }
            auto &new_pos = posIter->second;
            auto &new_vel = velIter->second;

            for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
                float m_sdf = *iter;
                if (m_sdf < limit_depth[0] || m_sdf > limit_depth[1])
                    continue;

                auto icoord = iter.getCoord();
                auto wcoord = Liquid_sdf->indexToWorld(icoord);
                float m_solid_sdf = openvdb::tools::BoxSampler::sample(solid_sdf_axr, Solid_sdf->worldToIndex(wcoord));
                if (m_solid_sdf < 0)
                    continue;

                float generates = 0;

                openvdb::Vec3f m_vel =
                    openvdb::tools::StaggeredBoxSampler::sample(vel_axr, Velocity->worldToIndex(wcoord));
                if (curv_emit > eps) {
                    auto norm_axr = Normal->getConstUnsafeAccessor();
                    auto curv_axr = Curvature->getConstUnsafeAccessor();

                    float m_curv = openvdb::tools::BoxSampler::sample(curv_axr, Curvature->worldToIndex(wcoord));
                    m_curv = std::abs(m_curv);
                    openvdb::Vec3f m_norm = openvdb::tools::BoxSampler::sample(norm_axr, Normal->worldToIndex(wcoord));
                    if (!checkAngle(m_vel, m_norm, max_angle))
                        m_curv = 0;
                    generates += curv_emit * clamp_map(m_curv, curv_range);
                }
                if (acc_emit > eps) {
                    auto pre_vel_axr = Pre_vel->getConstUnsafeAccessor();
                    openvdb::Vec3f m_pre_vel =
                        openvdb::tools::StaggeredBoxSampler::sample(pre_vel_axr, Pre_vel->worldToIndex(wcoord));
                    auto m_acc_vec = (m_vel - m_pre_vel) / dt;
                    float m_acc = m_acc_vec.length();
                    generates += acc_emit * clamp_map(m_acc, acc_range);
                }
                if (vor_emit > eps) {
                    auto vor_axr = Vorticity->getConstUnsafeAccessor();

                    openvdb::Vec3f m_vor_vec =
                        openvdb::tools::BoxSampler::sample(vor_axr, Vorticity->worldToIndex(wcoord));
                    float m_vor = m_vor_vec.length();
                    generates += vor_emit * clamp_map(m_vor, vor_range);
                }

                float m_speed = m_vel.length();
                generates *= clamp_map(m_speed, speed_range);

                int m_new_pars = std::round(generates);

                std::uniform_real_distribution<float> disX(wcoord[0] - 0.5f * dx, wcoord[0] + 0.5f * dx);
                std::uniform_real_distribution<float> disY(wcoord[1] - 0.5f * dx, wcoord[1] + 0.5f * dx);
                std::uniform_real_distribution<float> disZ(wcoord[2] - 0.5f * dx, wcoord[2] + 0.5f * dx);

                new_pos.reserve(new_pos.size() + m_new_pars);
                new_vel.reserve(new_vel.size() + m_new_pars);

                for (int n = 0; n < m_new_pars; ++n) {
                    openvdb::Vec3f m_par_pos{disX(gen), disY(gen), disZ(gen)};
                    openvdb::Vec3f m_par_vel =
                        openvdb::tools::StaggeredBoxSampler::sample(vel_axr, Velocity->worldToIndex(m_par_pos));

                    new_pos.push_back(other_to_vec<3>(m_par_pos));
                    new_vel.push_back(other_to_vec<3>(m_par_vel));
                }
            }
        };

        auto leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(Liquid_sdf->tree());
        leafman.foreach (particle_emitter);

        size_t new_size = 0;
        for (const auto &p : new_pars_pos) {
            new_size += p.second.size();
        }

        par_pos.reserve(pars->verts.size() + new_size);
        par_vel.reserve(pars->verts.size() + new_size);
        par_life.reserve(pars->verts.size() + new_size);
        for (const auto &p : new_pars_pos) {
            const auto &pos = p.second;
            const auto &vel = new_pars_vel[p.first];

            par_pos.insert(par_pos.end(), pos.begin(), pos.end());
            par_vel.insert(par_vel.end(), vel.begin(), vel.end());
        }
        par_life.insert(par_life.end(), new_size, Lifespan);

        pars->verts.update();

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
                               {"vec2f", "LimitDepth", "-1, 0.5"},
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
        auto TargetVelAttr = get_input2<std::string>("TargetVelAttr");

        auto gravity = vec_to_other<openvdb::Vec3f>(get_input2<vec3f>("Gravity"));
        auto dragModel = get_input2<std::string>("DragModel");
        auto air_drag = get_input2<float>("AirDrag");
        auto foam_drag = get_input2<float>("FoamDrag");
        auto bubble_drag = get_input2<float>("BubbleDrag");
        auto buoyancy = get_input2<float>("Buoyancy");

        float dx = static_cast<float>(Liquid_sdf->voxelSize()[0]);

        auto &par_pos = pars->verts.values;
        auto &par_vel = pars->attr<vec3f>("vel");
        auto &par_life = pars->attr<float>("life");
        auto &par_tarVel = pars->attr<vec3f>(TargetVelAttr);

        auto Normal = openvdb::tools::gradient(*Solid_sdf);

#pragma omp parallel for
        for (size_t idx = 0; idx < pars->size(); ++idx) {
            auto liquid_sdf_axr = Liquid_sdf->getConstUnsafeAccessor();
            auto solid_sdf_axr = Solid_sdf->getConstUnsafeAccessor();
            auto norm_axr = Normal->getConstUnsafeAccessor();

            auto m_vel = vec_to_other<openvdb::Vec3f>(par_vel[idx]);
            auto wcoord = vec_to_other<openvdb::Vec3f>(par_pos[idx]);
            float m_liquid_sdf = openvdb::tools::BoxSampler::sample(liquid_sdf_axr, Liquid_sdf->worldToIndex(wcoord));
            float drag_coef = 0;
            if (m_liquid_sdf > dx) {
                // spray
                drag_coef = air_drag;
                m_vel += gravity * dt;

                par_life[idx] -= 0.5f * dt;
            } else if (m_liquid_sdf < -0.3f * dx) {
                // bubble
                drag_coef = bubble_drag;
                m_vel += -buoyancy * gravity * dt;

                par_life[idx] -= 0.5f * dt;
            } else {
                // foam
                drag_coef = foam_drag;

                par_life[idx] -= dt;
            }
            auto m_tarVel = vec_to_other<openvdb::Vec3f>(par_tarVel[idx]);

            if (dragModel == "square") {
                //second step, semi-implicit integrate drag
                // (v_np1 - v_n) / dt = c * length(v_tar - v_n) * (v_tar - v_np1)
                vec3f v_target = vec3f(m_tarVel[0], m_tarVel[1], m_tarVel[2]);
                vec3f m_vel2 = vec3f(m_vel.x(), m_vel.y(), m_vel.z());
                float v_diff = zeno::distance(v_target, m_vel2);
                float denom = 1 + dt * air_drag * v_diff;
                m_vel = vec_to_other<openvdb::Vec3f>((dt * air_drag * v_diff * v_target + m_vel2) / denom);
            } else {
                // simple drag
                m_vel += drag_coef * (m_tarVel - m_vel);
            }

            auto wcoord_new = wcoord + dt * m_vel;
            float m_solid_sdf = openvdb::tools::BoxSampler::sample(solid_sdf_axr, Solid_sdf->worldToIndex(wcoord_new));
            if (m_solid_sdf < 0) {
                auto m_norm = openvdb::tools::BoxSampler::sample(norm_axr, Normal->worldToIndex(wcoord_new));
                m_norm.normalize();
                m_vel = -m_solid_sdf * m_norm / dt;
                wcoord_new += m_vel * dt;
            }

            par_pos[idx] = other_to_vec<3>(wcoord_new);
            par_vel[idx] = other_to_vec<3>(m_vel);
        }

        set_output("Primitive", pars);
    }
};

ZENDEFNODE(WhitewaterSolver, {/* inputs: */
                              {"Primitive",
                               {"float", "dt", "0.04"},
                               "LiquidSDF",
                               "SolidSDF",
                               {"string", "TargetVelAttr", "tv"},
                               {"vec3f", "Gravity", "0, -9.8, 0"},
                               {"enum linear square", "DragModel", "linear"},
                               {"float", "AirDrag", "0.05"},
                               {"float", "FoamDrag", "0.9"},
                               {"float", "BubbleDrag", "0.1"},
                               {"float", "Buoyancy", "5"}},
                              /* outputs: */
                              {"Primitive"},
                              /* params: */
                              {},
                              /* category: */
                              {"FLIPSolver"}});

} // namespace zeno