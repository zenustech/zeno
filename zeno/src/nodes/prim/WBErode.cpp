//
// WangBo 2022/11/28.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <random>
#include <vector>

#include <glm/gtx/quaternion.hpp>
#include "reflect/reflection.generated.hpp"


namespace zeno
{
namespace
{

///////////////////////////////////////////////////////////////////////////////
// 2022.10.10 Erode
///////////////////////////////////////////////////////////////////////////////

// erode ################################################
// ######################################################
// ######################################################
int Pos2Idx(const int x, const int z, const int nx) {
    return z * nx + x;
}

static unsigned int erode_random(float seed, int idx) {
    unsigned int s = *(unsigned int*)(&seed);
    s ^= idx << 3;
    s *= 179424691; // a magic prime number
    s ^= s << 13 | s >> (32 - 13);
    s ^= s >> 17 | s << (32 - 17);
    s ^= s << 23;
    s *= 179424691;
    return s;
}

// rain                                             用于子图：Erode_Precipitation
struct erode_value2cond : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto value = get_input<NumericObject>("value")->get<float>();
        auto seed  = get_input<NumericObject>("seed")->get<float>();

        // 初始化网格属性
        if (!terrain->verts.has_attr("cond")) {
            auto &_cond = terrain->verts.add_attr<float>("cond");
            std::fill(_cond.begin(), _cond.end(), 0.0);
        }
        auto &attr_cond = terrain->verts.attr<float>("cond");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int z = 0; z < nz; z++)
        {
            for (int x = 0; x < nx; x++)
            {
                int idx = Pos2Idx(x, z, nx);

                if (value >= 1.0f)
                {
                    attr_cond[idx] = 1;
                }
                else
                {
                    value = clamp(value, 0, 1);
                    unsigned int cutoff = (unsigned int)(value * 4294967295.0);
                    unsigned int randval = erode_random(seed, idx + nx * nz);
                    attr_cond[idx] = randval < cutoff;
                }
            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_value2cond,
           {/* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},
                   {"float", "value", "1.0"}, // 0.0 ~ 1.0
                   {"float", "seed", "0.0"},
               },
               /* outputs: */
               {
                   "prim_2DGrid",
               },
               /* params: */
               {

               },
               /* category: */
               {
                   "erode",
               }});

struct erode_rand_color : INode {
    void apply() override {
        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter       = get_input<NumericObject>("iter")->get<int>();

        int perm[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        for (int i = 0; i < 8; i++)
        {
            vec2f vec;
            std::mt19937 mt(iterations * iter * 8 * i + i);
            vec[0] = distr(mt);
            vec[1] = distr(mt);

            int idx1 = floor(vec[0] * 8);
            int idx2 = floor(vec[1] * 8);
            idx1 = idx1 == 8 ? 7 : idx1;
            idx2 = idx2 == 8 ? 7 : idx2;

            int temp = perm[idx1];
            perm[idx1] = perm[idx2];
            perm[idx2] = temp;
        }

        auto list = std::make_shared<zeno::ListObject>();
        for (int i = 0; i < 8; i++)
        {
            auto num = std::make_shared<zeno::NumericObject>();
            num->set<int>(perm[i]);
            list->push_back(num);
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(erode_rand_color,
           {/* inputs: */ {
                   {"int", "iterations", "0"},
                   {"int", "iter", "0"},
               },
               /* outputs: */
               {
                   "list",
               },
               /* params: */
               {

               },
               /* category: */
               {
                   "erode",
               }});

struct erode_rand_dir : INode {
    void apply() override {

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter       = get_input<NumericObject>("iter")->get<int>();

        int dirs[] = { -1, -1 };
        for (int i = 0; i < 2; i++)
        {
            std::mt19937 mt(iterations * iter * 2 * i + i);
            float rand_val = distr(mt);
            if (rand_val > 0.5)
            {
                dirs[i] = 1;
            }
            else
            {
                dirs[i] = -1;
            }
        }

        auto list = std::make_shared<zeno::ListObject>();
        for (int i = 0; i < 2; i++)
        {
            auto num = std::make_shared<zeno::NumericObject>();
            num->set<int>(dirs[i]);
            list->push_back(num);
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(erode_rand_dir,
           {/* inputs: */ {
                   {"int", "iterations", "0"},
                   {"int", "iter", "0"},
               },
               /* outputs: */
               {
                   "list",
               },
               /* params: */ {}, /* category: */
               {
                   "erode",
               }});

// thermal erosion NOT slump                        用于子图：Erode_Thermal                  thermal_erosion
struct erode_tumble_material_erosion : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();
        auto cut_angle = get_input<NumericObject>("cutangle")->get<float>();
        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>();
        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>();
        auto erodability = get_input<NumericObject>("erodability")->get<float>();
        auto removalrate = get_input<NumericObject>("removalrate")->get<float>();
        auto maxdepth = get_input<NumericObject>("maxdepth")->get<float>();

        std::uniform_real_distribution<float> distr(0.0, 1.0); // 设置随机分布
        auto seed = get_input<NumericObject>("seed")->get<float>();

        auto iterations = get_input<NumericObject>("iterations")->get<int>(); // 外部迭代总次数      10
        auto iter = get_input<NumericObject>("iter")->get<int>();             // 外部迭代当前次数    1~10
        auto i = get_input<NumericObject>("i")->get<int>();                   // 内部迭代当前次数    0~7
        auto openborder = get_input<NumericObject>("openborder")->get<int>(); // 获取边界标记

        auto perm = get_input<ListObject>("perm")->get2<int>();
        auto p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        // 初始化网格属性
        auto erodabilitymask_name = get_input2<std::string>("erodability_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(erodabilitymask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(erodabilitymask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_erodabilitymask = terrain->verts.attr<float>(erodabilitymask_name);

        auto removalratemask_name = get_input2<std::string>("removalrate_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(removalratemask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(removalratemask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_removalratemask = terrain->verts.attr<float>(removalratemask_name);

        auto cutanglemask_name = get_input2<std::string>("cutangle_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(cutanglemask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(cutanglemask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_cutanglemask = terrain->verts.attr<float>(cutanglemask_name);

        auto gridbiasmask_name = get_input2<std::string>("gridbias_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(gridbiasmask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(gridbiasmask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_gridbiasmask = terrain->verts.attr<float>(gridbiasmask_name);

        // 存放地质特征的属性
        if (!terrain->verts.has_attr("_height") || !terrain->verts.has_attr("_debris") ||
            !terrain->verts.has_attr("_temp_height") || !terrain->verts.has_attr("_temp_debris")) {
            zeno::log_error("Node [erode_tumble_material_v0], no such data layer named '{}' or '{}' or '{}' or '{}'.",
                            "_height", "_debris", "_temp_height", "_temp_debris");
        }
        auto &_height = terrain->verts.attr<float>("_height"); // 计算用的临时属性
        auto &_debris = terrain->verts.attr<float>("_debris");
        auto &_temp_height = terrain->verts.attr<float>("_temp_height"); // 备份用的临时属性
        auto &_temp_debris = terrain->verts.attr<float>("_temp_debris");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];

                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                int dxs[] = { 0, p_dirs[0], 0, p_dirs[0], x_dirs[0], x_dirs[1], x_dirs[0], x_dirs[1] };
                int dzs[] = { p_dirs[1], 0, p_dirs[1], 0, x_dirs[0],-x_dirs[1], x_dirs[0],-x_dirs[1] };

                if (is_red || is_green || is_blue || is_yellow || is_x_turn_x || is_x_turn_y)
                {
                    int idx = Pos2Idx(id_x, id_z, nx);
                    int dx = dxs[color - 1];
                    int dz = dzs[color - 1];
                    int bound_x = nx;
                    int bound_z = nz;
                    int clamp_x = bound_x - 1;
                    int clamp_z = bound_z - 1;

                    float i_debris = _temp_debris[idx];
                    float i_height = _temp_height[idx];

                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);
                    if (validsource)
                    {
                        validsource = validsource || !openborder;
                        int j_idx = Pos2Idx(samplex, samplez, nx);
                        float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;
                        float j_height = _temp_height[j_idx];

                        int cidx = 0;
                        int cidz = 0;

                        float c_height = 0.0f;
                        float c_debris = 0.0f;
                        float n_debris = 0.0f;

                        int c_idx = 0;
                        int n_idx = 0;

                        int dx_check = 0;
                        int dz_check = 0;

                        float h_diff = 0.0f;

                        if ((j_height - i_height) > 0.0f)
                        {
                            cidx = samplex;
                            cidz = samplez;

                            c_height = j_height;
                            c_debris = j_debris;
                            n_debris = i_debris;

                            c_idx = j_idx;
                            n_idx = idx;

                            dx_check = -dx;
                            dz_check = -dz;

                            h_diff = j_height - i_height;
                        }
                        else
                        {
                            cidx = id_x;
                            cidz = id_z;

                            c_height = i_height;
                            c_debris = i_debris;
                            n_debris = j_debris;

                            c_idx = idx;
                            n_idx = j_idx;

                            dx_check = dx;
                            dz_check = dz;

                            h_diff = i_height - j_height;
                        }

                        float max_diff = 0.0f;
                        float dir_prob = 0.0f;
                        float c_gridbiasmask = _gridbiasmask[c_idx];
                        for (int tmp_dz = -1; tmp_dz <= 1; tmp_dz++)
                        {
                            for (int tmp_dx = -1; tmp_dx <= 1; tmp_dx++)
                            {
                                if (!tmp_dx && !tmp_dz)
                                    continue;

                                int tmp_samplex = clamp(cidx + tmp_dx, 0, clamp_x);
                                int tmp_samplez = clamp(cidz + tmp_dz, 0, clamp_z);
                                int tmp_validsource = (tmp_samplex == (cidx + tmp_dx)) && (tmp_samplez == (cidz + tmp_dz));
                                tmp_validsource = tmp_validsource || !openborder;
                                int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                float n_height = _temp_height[tmp_j_idx];

                                float tmp_diff = n_height - (c_height);

                                //float _gridbias = clamp(gridbias, -1.0f, 1.0f);
                                float _gridbias = clamp(gridbias * c_gridbiasmask, -1.0f, 1.0f);

                                if (tmp_dx && tmp_dz)
                                    tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                else
                                    tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                if (tmp_diff <= 0.0f)
                                {
                                    if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                        dir_prob = tmp_diff;
                                    if (tmp_diff < max_diff)
                                        max_diff = tmp_diff;
                                }
                            }
                        }
                        if (max_diff > 0.001f || max_diff < -0.001f)
                            dir_prob = dir_prob / max_diff;

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else
                        {
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        if (cond)
                        {
                            float abs_h_diff = h_diff < 0.0f ? -h_diff : h_diff;
                            //float _cut_angle = clamp(cut_angle, 0.0f, 90.0f);
                            float _cut_angle = clamp(cut_angle * _cutanglemask[n_idx], 0.0f, 90.0f);
                            float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f);
                            float height_removed = _cut_angle < 90.0f ? tan(_cut_angle * M_PI / 180) * delta_x : 1e10f;
                            float height_diff = abs_h_diff - height_removed;
                            if (height_diff < 0.0f)
                                height_diff = 0.0f;
                            float prob = ((n_debris + c_debris) != 0.0f) ? clamp((height_diff / (n_debris + c_debris)), 0.0f, 1.0f) : 1.0f;
                            unsigned int cutoff = (unsigned int)(prob * 4294967295.0);
                            unsigned int randval = erode_random(seed * 3.14, (idx + nx * nz) * 8 + color + iterseed);
                            int do_erode = randval < cutoff;

                            float height_removal_amt = do_erode * clamp(global_erosionrate * erosionrate * erodability * _erodabilitymask[c_idx], 0.0f, height_diff);

                            _height[c_idx] -= height_removal_amt;

                            //float bedrock_density = 1.0f - (removalrate);
                            float bedrock_density = 1.0f - (removalrate * _removalratemask[c_idx]);
                            if (bedrock_density > 0.0f)
                            {
                                float newdebris = bedrock_density * height_removal_amt;
                                if (n_debris + newdebris > maxdepth)
                                {
                                    float rollback = n_debris + newdebris - maxdepth;
                                    rollback = min(rollback, newdebris);
                                    _height[c_idx] += rollback / bedrock_density;
                                    newdebris -= rollback;
                                }
                                _debris[c_idx] += newdebris;
                            }
                        }
                    }
                }
            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_erosion,
           {/* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},

                   {"ListObject", "perm"},
                   {"ListObject", "p_dirs"},
                   {"ListObject", "x_dirs"},

                   {"float", "seed", "9676.79"},
                   {"int", "iterations", "0"},
                   {"int", "iter", "0"},
                   {"int", "i", "0"},

                   {"int", "openborder", "0"},
                   {"float", "maxdepth", "5.0"},
                   {"float", "global_erosionrate", "1.0"},
                   {"float", "erosionrate", "0.03"},

                   {"float", "cutangle", "35"},
                   {"string", "cutangle_mask_layer", "cutangle_mask"},

                   {"float", "erodability", "0.4"},
                   {"string", "erodability_mask_layer", "erodability_mask"},

                   {"float", "removalrate", "0.7"},
                   {"string", "removalrate_mask_layer", "removalrate_mask"},

                   {"float", "gridbias", "0.0"},
                   {"string", "gridbias_mask_layer", "gridbias_mask"},
               },
               /* outputs: */
               {
                   "prim_2DGrid",
               },
               /* params: */
               {

               },
               /* category: */
               {
                   "erode",
               }});

// smooth slump                                     实现有误，如需要使用，在 v1 基础上修改即可     smooth
struct erode_tumble_material_v0 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();
        auto cut_angle = get_input<NumericObject>("cutangle")->get<float>();
        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>();
        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>();
        auto erodability = get_input<NumericObject>("erodability")->get<float>();
        auto removalrate = get_input<NumericObject>("removalrate")->get<float>();
        auto maxdepth = get_input<NumericObject>("maxdepth")->get<float>();

        std::uniform_real_distribution<float> distr(0.0, 1.0); // 设置随机分布
        auto seed = get_input<NumericObject>("seed")->get<float>();

        auto iterations = get_input<NumericObject>("iterations")->get<int>(); // 外部迭代总次数      10
        auto iter = get_input<NumericObject>("iter")->get<int>();             // 外部迭代当前次数    1~10
        auto i = get_input<NumericObject>("i")->get<int>();                   // 内部迭代当前次数    0~7
        auto openborder = get_input<NumericObject>("openborder")->get<int>(); // 获取边界标记

        auto perm = get_input<ListObject>("perm")->get2<int>();
        auto p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        // 初始化网格属性
        auto erodabilitymask_name = get_input2<std::string>("erodability_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(erodabilitymask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(erodabilitymask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_erodabilitymask = terrain->verts.attr<float>(erodabilitymask_name);

        auto removalratemask_name = get_input2<std::string>("removalrate_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(removalratemask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(removalratemask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_removalratemask = terrain->verts.attr<float>(removalratemask_name);

        auto cutanglemask_name = get_input2<std::string>("cutangle_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(cutanglemask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(cutanglemask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_cutanglemask = terrain->verts.attr<float>(cutanglemask_name);

        auto gridbiasmask_name = get_input2<std::string>("gridbias_mask_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 1.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(gridbiasmask_name))
        {
            auto &_temp = terrain->verts.add_attr<float>(gridbiasmask_name);
            std::fill(_temp.begin(), _temp.end(), 1.0);
        }
        auto &_gridbiasmask = terrain->verts.attr<float>(gridbiasmask_name);

        // 存放地质特征的属性
        if (!terrain->verts.has_attr("_height") || !terrain->verts.has_attr("_debris") ||
            !terrain->verts.has_attr("_temp_height") || !terrain->verts.has_attr("_temp_debris")) {
            zeno::log_error("Node [erode_tumble_material_v0], no such data layer named '{}' or '{}' or '{}' or '{}'.",
                            "_height", "_debris", "_temp_height", "_temp_debris");
        }
        auto &_height = terrain->verts.attr<float>("_height"); // 计算用的临时属性
        auto &_debris = terrain->verts.attr<float>("_debris");
        auto &_temp_height = terrain->verts.attr<float>("_temp_height"); // 备份用的临时属性
        auto &_temp_debris = terrain->verts.attr<float>("_temp_debris");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];

                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                int dxs[] = { 0, p_dirs[0], 0, p_dirs[0], x_dirs[0], x_dirs[1], x_dirs[0], x_dirs[1] };
                int dzs[] = { p_dirs[1], 0, p_dirs[1], 0, x_dirs[0],-x_dirs[1], x_dirs[0],-x_dirs[1] };

                if (is_red || is_green || is_blue || is_yellow || is_x_turn_x || is_x_turn_y)
                {
                    int idx = Pos2Idx(id_x, id_z, nx);
                    int dx = dxs[color - 1];
                    int dz = dzs[color - 1];
                    int bound_x = nx;
                    int bound_z = nz;
                    int clamp_x = bound_x - 1;
                    int clamp_z = bound_z - 1;

                    float i_debris = _temp_debris[idx];
                    float i_height = _temp_height[idx];

                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);
                    if (validsource)
                    {
                        validsource = validsource || !openborder;
                        int j_idx = Pos2Idx(samplex, samplez, nx);
                        float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;
                        float j_height = _temp_height[j_idx];

                        int cidx = 0;
                        int cidz = 0;

                        float c_height = 0.0f;
                        float c_debris = 0.0f;
                        float n_debris = 0.0f;

                        int c_idx = 0;
                        int n_idx = 0;

                        int dx_check = 0;
                        int dz_check = 0;

                        float h_diff = 0.0f;

                        if ((j_height - i_height) > 0.0f)
                        {
                            cidx = samplex;
                            cidz = samplez;

                            c_height = j_height;
                            c_debris = j_debris;
                            n_debris = i_debris;

                            c_idx = j_idx;
                            n_idx = idx;

                            dx_check = -dx;
                            dz_check = -dz;

                            h_diff = j_height - i_height;
                        }
                        else
                        {
                            cidx = id_x;
                            cidz = id_z;

                            c_height = i_height;
                            c_debris = i_debris;
                            n_debris = j_debris;

                            c_idx = idx;
                            n_idx = j_idx;

                            dx_check = dx;
                            dz_check = dz;

                            h_diff = i_height - j_height;
                        }

                        float max_diff = 0.0f;
                        float dir_prob = 0.0f;
                        float c_gridbiasmask = _gridbiasmask[c_idx];
                        for (int tmp_dz = -1; tmp_dz <= 1; tmp_dz++)
                        {
                            for (int tmp_dx = -1; tmp_dx <= 1; tmp_dx++)
                            {
                                if (!tmp_dx && !tmp_dz)
                                    continue;

                                int tmp_samplex = clamp(cidx + tmp_dx, 0, clamp_x);
                                int tmp_samplez = clamp(cidz + tmp_dz, 0, clamp_z);
                                int tmp_validsource = (tmp_samplex == (cidx + tmp_dx)) && (tmp_samplez == (cidz + tmp_dz));
                                tmp_validsource = tmp_validsource || !openborder;
                                int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                float n_height = _temp_height[tmp_j_idx];

                                float tmp_diff = n_height - (c_height);

                                //float _gridbias = clamp(gridbias, -1.0f, 1.0f);
                                float _gridbias = clamp(gridbias * c_gridbiasmask, -1.0f, 1.0f);

                                if (tmp_dx && tmp_dz)
                                    tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                else
                                    tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                if (tmp_diff <= 0.0f)
                                {
                                    if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                        dir_prob = tmp_diff;
                                    if (tmp_diff < max_diff)
                                        max_diff = tmp_diff;
                                }
                            }
                        }
                        if (max_diff > 0.001f || max_diff < -0.001f)
                            dir_prob = dir_prob / max_diff;

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else
                        {
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        if (cond)
                        {
                            float abs_h_diff = h_diff < 0.0f ? -h_diff : h_diff;
                            //float _cut_angle = clamp(cut_angle, 0.0f, 90.0f);
                            float _cut_angle = clamp(cut_angle * _cutanglemask[n_idx], 0.0f, 90.0f);
                            float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f);
                            float height_removed = _cut_angle < 90.0f ? tan(_cut_angle * M_PI / 180) * delta_x : 1e10f;
                            float height_diff = abs_h_diff - height_removed;
                            if (height_diff < 0.0f)
                                height_diff = 0.0f;
                            float prob = ((n_debris + c_debris) != 0.0f) ? clamp((height_diff / (n_debris + c_debris)), 0.0f, 1.0f) : 1.0f;
                            unsigned int cutoff = (unsigned int)(prob * 4294967295.0);
                            unsigned int randval = erode_random(seed * 3.14, (idx + nx * nz) * 8 + color + iterseed);
                            int do_erode = randval < cutoff;

                            float height_removal_amt = do_erode * clamp(global_erosionrate * erosionrate * erodability * _erodabilitymask[c_idx], 0.0f, height_diff);

                            _height[c_idx] -= height_removal_amt;

                            //float bedrock_density = 1.0f - (removalrate);
                            float bedrock_density = 1.0f - (removalrate * _removalratemask[c_idx]);
                            if (bedrock_density > 0.0f)
                            {
                                float newdebris = bedrock_density * height_removal_amt;
                                if (n_debris + newdebris > maxdepth)
                                {
                                    float rollback = n_debris + newdebris - maxdepth;
                                    rollback = min(rollback, newdebris);
                                    _height[c_idx] += rollback / bedrock_density;
                                    newdebris -= rollback;
                                }
                                _debris[c_idx] += newdebris;
                            }
                        }
                    }
                }
            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v0,
           {/* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},

                   {"ListObject", "perm"},
                   {"ListObject", "p_dirs"},
                   {"ListObject", "x_dirs"},

                   {"float", "seed", "9676.79"},
                   {"int", "iterations", "0"},
                   {"int", "iter", "0"},
                   {"int", "i", "0"},

                   {"int", "openborder", "0"},
                   {"float", "maxdepth", "5.0"},
                   {"float", "global_erosionrate", "1.0"},
                   {"float", "erosionrate", "0.03"},

                   {"float", "cutangle", "35"},
                   {"string", "cutangle_mask_layer", "cutangle_mask"},

                   {"float", "erodability", "0.4"},
                   {"string", "erodability_mask_layer", "erodability_mask"},

                   {"float", "removalrate", "0.7"},
                   {"string", "removalrate_mask_layer", "removalrate_mask"},

                   {"float", "gridbias", "0.0"},
                   {"string", "gridbias_mask_layer", "gridbias_mask"},
               },
               /* outputs: */
               {
                   "prim_2DGrid",
               },
               /* params: */
               {

               },
               /* category: */
               {
                   "erode",
               }});

// smooth slump + flow                              用于子图：Erode_Smooth_Slump_Flow        smooth + flow
struct erode_tumble_material_v1 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("HeightField");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto openborder = get_input<NumericObject>("openborder")->get<int>();
        auto repose_angle = get_input<NumericObject>("repose_angle")->get<float>();
        auto flow_rate = get_input<NumericObject>("flow_rate")->get<float>();
        auto height_factor = get_input<NumericObject>("height_factor")->get<float>();
        auto entrainmentrate = get_input<NumericObject>("entrainmentrate")->get<float>();

        // 初始化网格属性
        auto write_back_material_layer = get_input2<std::string>("write_back_material_layer");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 0.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(write_back_material_layer))
        {
            auto &_sta = terrain->verts.add_attr<float>(write_back_material_layer);
            std::fill(_sta.begin(), _sta.end(), 0.0);
        }
        auto &write_back_material = terrain->verts.attr<float>(write_back_material_layer);

        // 存放地质特征的属性
        if (!terrain->verts.has_attr("height") ||
            !terrain->verts.has_attr("_material") ||
            !terrain->verts.has_attr("flowdir")) {
            zeno::log_error("no such data layer named '{}' or '{}' or '{}'.",
                            "height", "_material", "flowdir");
        }
        auto &height                = terrain->verts.attr<float>("height");
        auto &_material             = terrain->verts.attr<float>("_material");
        auto &flowdir               = terrain->verts.attr<vec3f>("flowdir");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                int bound_x = nx;
                int bound_z = nz;
                int clamp_x = bound_x - 1;
                int clamp_z = bound_z - 1;

                // Validate parameters
                flow_rate = clamp(flow_rate, 0.0f, 1.0f);
                repose_angle = clamp(repose_angle, 0.0f, 90.0f);
                height_factor = clamp(height_factor, 0.0f, 1.0f);

                // The maximum slope at which we stop slumping
                float static_diff = repose_angle < 90.0f ? tan(repose_angle * M_PI / 180.0) * cellSize : 1e10f;

                // Initialize accumulation of flow
                float net_diff = 0.0f;
                float net_entrained = 0.0f;

                float net_diff_x = 0.0f;
                float net_diff_z = 0.0f;

                // Get the current height level
                float i_material = _material[idx];
                float i_entrained = 0;
                float i_height = height_factor * height[idx] + i_material + i_entrained;

                bool moved = false;
                // For each of the 8 neighbours, we get the difference in total
                // height and add to our flow values.
                for (int dz = -1; dz <= 1; dz++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (!dx && !dz)
                            continue;

                        int samplex = clamp(id_x + dx, 0, clamp_x);
                        int samplez = clamp(id_z + dz, 0, clamp_z);
                        int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);
                        // If we have closed borders, pretend a valid source to create
                        // a streak condition
                        validsource = validsource || !openborder;
                        int j_idx = samplex + samplez * nx;
                        float j_material = validsource ? _material[j_idx] : 0.0f;
                        float j_entrained = 0;

                        float j_height = height_factor * height[j_idx] + j_material + j_entrained;

                        float diff = j_height - i_height;

                        // Calculate the distance to this neighbour
                        float distance = (dx && dz) ? 1.4142136f : 1.0f;
                        // Cutoff at the repose angle
                        float static_cutoff = distance * static_diff;
                        diff = diff > 0.0f ? max(diff - static_cutoff, 0.0f) : min(diff + static_cutoff, 0.0f);

                        // Weight the difference by the inverted distance
                        diff = distance > 0.0f ? diff / distance : 0.0f;

                        // Clamp within the material levels of the voxels
                        diff = clamp(diff, -i_material, j_material);

                        // Some percentage of the material flow will drag
                        // the entrained material instead.
                        float entrained_diff = diff * entrainmentrate;

                        // Clamp entrained diff by the entrained levels.
                        entrained_diff = clamp(entrained_diff, -i_entrained, j_entrained);

                        // Flow uses total diff, including entrained material
                        net_diff_x += (float) dx * diff;
                        net_diff_z += (float) dz * diff;

                        // And reduce the material diff by the amount of entrained substance
                        // moved so total height updates as expected.
                        diff -= entrained_diff;

                        // Accumulate the diff
                        net_diff += diff;
                        net_entrained += entrained_diff;
                    }
                }

                // 0.17 is to keep us in the circle of stability
                float weight = flow_rate * 0.17;
                net_diff *= weight;
                net_entrained *= weight;

                // Negate the directional flow so that they are positive in their axis direction
                net_diff_x *= -weight;
                net_diff_z *= -weight;

                // Ensure diff cannot bring the material level negative
                net_diff = max(net_diff, -i_material);
                net_entrained = max(net_entrained, -i_entrained);

                // Update the material level
                write_back_material[idx] = i_material + net_diff;

                // Update the flow
                flowdir[idx][0] += net_diff_x;
                flowdir[idx][2] += net_diff_z;
            }
        }

        set_output("HeightField", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v1,
           {/* inputs: */ {
                   {"", "HeightField", "", zeno::Socket_ReadOnly},
                   {"string", "write_back_material_layer", "write_back_material"},
                   {"int", "openborder", "0"},
                   {"float", "repose_angle", "15.0"},
                   {"float", "flow_rate", "1.0"},
                   {"float", "height_factor", "1.0"},
                   {"float", "entrainmentrate", "0.0"},
               },
               /* outputs: */
               {
                   "HeightField",
               },
               /* params: */
               {
               },
               /* category: */
               {
                   "erode",
               }});

// granular slump                                   用于子图：Erode_Slump_Debris             granular
struct erode_tumble_material_v2 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("HeightField");
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz"))) zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto& pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();
        auto repose_angle = get_input<NumericObject>("repose_angle")->get<float>();
        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>();
        auto flow_rate = get_input<NumericObject>("flow_rate")->get<float>();

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto seed = get_input<NumericObject>("seed")->get<float>();

        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter = get_input<NumericObject>("iter")->get<int>();
        auto i = get_input<NumericObject>("i")->get<int>();
        auto openborder = get_input<NumericObject>("openborder")->get<int>();

        auto perm = get_input<ListObject>("perm")->get2<int>();
        auto p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        // 初始化网格属性
        auto stablilityMaskName = get_input2<std::string>("stabilitymask");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 0.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(stablilityMaskName)) {
            auto &_sta = terrain->verts.add_attr<float>(stablilityMaskName);
            std::fill(_sta.begin(), _sta.end(), 0.0);
        }
        auto &stabilitymask = terrain->verts.attr<float>(stablilityMaskName);

        if (!terrain->verts.has_attr("_height") ||
            !terrain->verts.has_attr("_material") ||
            !terrain->verts.has_attr("_temp_material")) {
            zeno::log_error("Node [erode_tumble_material_v2], no such data layer named '{}' or '{}' or '{}'.",
                            "_height", "_material", "_temp_material");
        }
        auto &_height           = terrain->verts.attr<float>("_height");
        auto &_material         = terrain->verts.attr<float>("_material");
        auto &_temp_material    = terrain->verts.attr<float>("_temp_material");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];

                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                int dxs[] = { 0, p_dirs[0], 0, p_dirs[0], x_dirs[0], x_dirs[1], x_dirs[0], x_dirs[1] };
                int dzs[] = { p_dirs[1], 0, p_dirs[1], 0, x_dirs[0],-x_dirs[1], x_dirs[0],-x_dirs[1] };

                if (is_red || is_green || is_blue || is_yellow || is_x_turn_x || is_x_turn_y)
                {
                    int idx = Pos2Idx(id_x, id_z, nx);
                    int dx = dxs[color - 1];
                    int dz = dzs[color - 1];
                    int bound_x = nx;
                    int bound_z = nz;
                    int clamp_x = bound_x - 1;
                    int clamp_z = bound_z - 1;

                    flow_rate = clamp(flow_rate, 0.0f, 1.0f);

                    float i_material = _temp_material[idx];
                    float i_height = _height[idx];

                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                    if (validsource)
                    {
                        int same_node = !validsource;

                        validsource = validsource || !openborder;

                        int j_idx = Pos2Idx(samplex, samplez, nx);

                        float j_material = validsource ? _temp_material[j_idx] : 0.0f;
                        float j_height = _height[j_idx];

                        float _repose_angle = repose_angle;
                        _repose_angle = clamp(_repose_angle, 0.0f, 90.0f);
                        float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f);
                        float static_diff = _repose_angle < 90.0f ? tan(_repose_angle * M_PI / 180.0) * delta_x : 1e10f;
                        float m_diff = (j_height + j_material) - (i_height + i_material);
                        int cidx = 0;
                        int cidz = 0;

                        float c_height = 0.0f;
                        float c_material = 0.0f;
                        float n_material = 0.0f;

                        int c_idx = 0;
                        int n_idx = 0;

                        int dx_check = 0;
                        int dz_check = 0;

                        if (m_diff > 0.0f)
                        {
                            cidx = samplex;
                            cidz = samplez;

                            c_height = j_height;
                            c_material = j_material;
                            n_material = i_material;

                            c_idx = j_idx;
                            n_idx = idx;

                            dx_check = -dx;
                            dz_check = -dz;
                        }
                        else
                        {
                            cidx = id_x;
                            cidz = id_z;

                            c_height = i_height;
                            c_material = i_material;
                            n_material = j_material;

                            c_idx = idx;
                            n_idx = j_idx;

                            dx_check = dx;
                            dz_check = dz;
                        }

                        float sum_diffs[] = { 0.0f, 0.0f };
                        float dir_probs[] = { 0.0f, 0.0f };
                        float dir_prob = 0.0f;
                        for (int diff_idx = 0; diff_idx < 2; diff_idx++)
                        {
                            for (int tmp_dz = -1; tmp_dz <= 1; tmp_dz++)
                            {
                                for (int tmp_dx = -1; tmp_dx <= 1; tmp_dx++)
                                {
                                    if (!tmp_dx && !tmp_dz)
                                        continue;

                                    int tmp_samplex = clamp(cidx + tmp_dx, 0, clamp_x);
                                    int tmp_samplez = clamp(cidz + tmp_dz, 0, clamp_z);
                                    int tmp_validsource = (tmp_samplex == (cidx + tmp_dx)) && (tmp_samplez == (cidz + tmp_dz));
                                    tmp_validsource = tmp_validsource || !openborder;
                                    int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                    float n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;
                                    float n_height = _height[tmp_j_idx];
                                    float tmp_h_diff = n_height - (c_height);
                                    float tmp_m_diff = (n_height + n_material) - (c_height + c_material);
                                    float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;
                                    float _gridbias = gridbias;
                                    _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                    if (tmp_dx && tmp_dz)
                                        tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                    else
                                        tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                    if (tmp_diff <= 0.0f)
                                    {
                                        if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                            dir_probs[diff_idx] = tmp_diff;

                                        if (diff_idx && dir_prob > tmp_diff)
                                            dir_prob = tmp_diff;

                                        sum_diffs[diff_idx] += tmp_diff;
                                    }
                                }
                            }

                            if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                                dir_prob = dir_probs[diff_idx] / dir_prob;

                            if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                                dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                        }

                        float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;
                        float stability_val = 0.0f;
                        stability_val = clamp(stabilitymask[c_idx], 0.0f, 1.0f);

                        if (stability_val > 0.01f)
                            movable_mat = clamp(movable_mat * (1.0f - stability_val) * 0.5f, 0.0f, c_material);
                        else
                            movable_mat = clamp((movable_mat - static_diff) * 0.5f, 0.0f, c_material);

                        float l_rat = dir_probs[1];
                        if (quant_amt > 0.001)
                            movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                        else
                            movable_mat *= l_rat;

                        float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else
                        {
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        if (!cond || same_node)
                            diff = 0.0f;

                        diff *= flow_rate;
                        float abs_diff = (diff < 0.0f) ? -diff : diff;
                        _material[c_idx] = c_material - abs_diff;
                        _material[n_idx] = n_material + abs_diff;
                    }
                }
            }
        }

        set_output("HeightField", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v2,
           {/* inputs: */ {
                   {"", "HeightField", "", zeno::Socket_ReadOnly},

                   {"string", "stabilitymask", "_stability"},
                   {"ListObject", "perm"},
                   {"ListObject", "p_dirs"},
                   {"ListObject", "x_dirs"},

                   {"float", "seed", "15231.3"},
                   {"int", "iterations", "0"},
                   {"int", "iter", "0"},
                   {"int", "i", "0"},

                   {"int", "openborder", "0"},
                   {"float", "gridbias", "0.0"},

                   // 崩塌流淌相关
                   {"float", "repose_angle", "15.0"},
                   {"float", "quant_amt", "0.25"},
                   {"float", "flow_rate", "1.0"},
               },
               /* outputs: */
               {
                   "HeightField",
               },
               /* params: */
               {
                   //{"string", "stabilitymask", "_stability"},
               },
               /* category: */
               {
                   "erode",
               }});

// granular slump + flow                            用于子图：Erode_Granular_Slump_Flow      granular + flow
struct erode_tumble_material_v3 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();
        auto repose_angle = get_input<NumericObject>("repose_angle")->get<float>();
        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>();
        auto flow_rate = get_input<NumericObject>("flow_rate")->get<float>();

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto seed = get_input<NumericObject>("seed")->get<float>();

        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter = get_input<NumericObject>("iter")->get<int>();
        auto i = get_input<NumericObject>("i")->get<int>();
        auto openborder = get_input<NumericObject>("openborder")->get<int>();

        auto perm = get_input<ListObject>("perm")->get2<int>();
        auto p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        // 初始化网格属性
        auto stablilityMaskName = get_input2<std::string>("stabilitymask");
        // 如果此 mask 属性不存在，则添加此属性，且初始化为 0.0，并在节点处理过程的末尾将其删除
        if (!terrain->verts.has_attr(stablilityMaskName))
        {
            auto &_sta = terrain->verts.add_attr<float>(stablilityMaskName);
            std::fill(_sta.begin(), _sta.end(), 0.0);
        }
        auto &stabilitymask = terrain->verts.attr<float>(stablilityMaskName);

        // 存放地质特征的属性
        if (!terrain->verts.has_attr("height") ||
            !terrain->verts.has_attr("_material") ||
            !terrain->verts.has_attr("_temp_material") ||
            !terrain->verts.has_attr("flowdir")) {
            zeno::log_error("Node [erode_tumble_material_v3], no such data layer named '{}' or '{}' or '{}' or "
                            "'{}'.", "height", "_material", "_temp_material", "flowdir");
        }
        auto &height            = terrain->verts.attr<float>("height");
        auto &_material         = terrain->verts.attr<float>("_material");
        auto &_temp_material    = terrain->verts.attr<float>("_temp_material");
        auto &flowdir           = terrain->verts.attr<vec3f>("flowdir");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];

                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                int dxs[] = {0, p_dirs[0], 0, p_dirs[0], x_dirs[0], x_dirs[1], x_dirs[0], x_dirs[1]};
                int dzs[] = {p_dirs[1], 0, p_dirs[1], 0, x_dirs[0], -x_dirs[1], x_dirs[0], -x_dirs[1]};

                if (is_red || is_green || is_blue || is_yellow || is_x_turn_x || is_x_turn_y) {
                    int idx = Pos2Idx(id_x, id_z, nx);
                    int dx = dxs[color - 1];
                    int dz = dzs[color - 1];
                    int bound_x = nx;
                    int bound_z = nz;
                    int clamp_x = bound_x - 1;
                    int clamp_z = bound_z - 1;

                    flow_rate = clamp(flow_rate, 0.0f, 1.0f);

                    // CALC_FLOW
                    float diff_x = 0.0f;
                    float diff_z = 0.0f;

                    float i_material = _temp_material[idx];
                    float i_height = height[idx];

                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                    if (validsource)
                    {
                        int same_node = !validsource;

                        validsource = validsource || !openborder;

                        int j_idx = Pos2Idx(samplex, samplez, nx);

                        float j_material = validsource ? _temp_material[j_idx] : 0.0f;
                        float j_height = height[j_idx];

                        float _repose_angle = repose_angle;
                        _repose_angle = clamp(_repose_angle, 0.0f, 90.0f);
                        float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f);

                        float static_diff = _repose_angle < 90.0f ? tan(_repose_angle * M_PI / 180.0) * delta_x : 1e10f;

                        float m_diff = (j_height + j_material) - (i_height + i_material);

                        int cidx = 0;
                        int cidz = 0;

                        float c_height = 0.0f;
                        float c_material = 0.0f;
                        float n_material = 0.0f;

                        int c_idx = 0;
                        int n_idx = 0;

                        int dx_check = 0;
                        int dz_check = 0;

                        if (m_diff > 0.0f) {
                            cidx = samplex;
                            cidz = samplez;

                            c_height = j_height;
                            c_material = j_material;
                            n_material = i_material;

                            c_idx = j_idx;
                            n_idx = idx;

                            dx_check = -dx;
                            dz_check = -dz;
                        } else {
                            cidx = id_x;
                            cidz = id_z;

                            c_height = i_height;
                            c_material = i_material;
                            n_material = j_material;

                            c_idx = idx;
                            n_idx = j_idx;

                            dx_check = dx;
                            dz_check = dz;
                        }

                        float sum_diffs[] = {0.0f, 0.0f};
                        float dir_probs[] = {0.0f, 0.0f};
                        float dir_prob = 0.0f;
                        for (int diff_idx = 0; diff_idx < 2; diff_idx++) {
                            for (int tmp_dz = -1; tmp_dz <= 1; tmp_dz++) {
                                for (int tmp_dx = -1; tmp_dx <= 1; tmp_dx++) {
                                    if (!tmp_dx && !tmp_dz)
                                        continue;

                                    int tmp_samplex = clamp(cidx + tmp_dx, 0, clamp_x);
                                    int tmp_samplez = clamp(cidz + tmp_dz, 0, clamp_z);
                                    int tmp_validsource = (tmp_samplex == (cidx + tmp_dx)) && (tmp_samplez == (cidz + tmp_dz));

                                    tmp_validsource = tmp_validsource || !openborder;
                                    int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                    float n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;
                                    float n_height = height[tmp_j_idx];
                                    float tmp_h_diff = n_height - (c_height);
                                    float tmp_m_diff = (n_height + n_material) - (c_height + c_material);
                                    float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;
                                    float _gridbias = gridbias;

                                    _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                    if (tmp_dx && tmp_dz)
                                        tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                    else
                                        tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                    if (tmp_diff <= 0.0f)
                                    {
                                        if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                            dir_probs[diff_idx] = tmp_diff;

                                        if (diff_idx && dir_prob > tmp_diff)
                                            dir_prob = tmp_diff;

                                        sum_diffs[diff_idx] += tmp_diff;
                                    }
                                }
                            }

                            if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                                dir_prob = dir_probs[diff_idx] / dir_prob;

                            if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                                dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                        }

                        float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;
                        float stability_val = 0.0f;
                        stability_val = clamp(stabilitymask[c_idx], 0.0f, 1.0f);

                        if (stability_val > 0.01f)
                            movable_mat = clamp(movable_mat * (1.0f - stability_val) * 0.5f, 0.0f, c_material);
                        else
                            movable_mat = clamp((movable_mat - static_diff) * 0.5f, 0.0f, c_material);

                        float l_rat = dir_probs[1];
                        if (quant_amt > 0.001)
                            movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                        else
                            movable_mat *= l_rat;

                        float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else {
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        if (!cond || same_node)
                            diff = 0.0f;

                        diff *= flow_rate;

                        // CALC_FLOW
                        diff_x += (float)dx * diff;
                        diff_z += (float)dz * diff;
                        diff_x *= -1.0f;
                        diff_z *= -1.0f;

                        float abs_diff = (diff < 0.0f) ? -diff : diff;
                        _material[c_idx] = c_material - abs_diff;
                        _material[n_idx] = n_material + abs_diff;

                        // CALC_FLOW
                        float abs_c_x = flowdir[c_idx][0];
                        abs_c_x = (abs_c_x < 0.0f) ? -abs_c_x : abs_c_x;
                        float abs_c_z = flowdir[c_idx][2];
                        abs_c_z = (abs_c_z < 0.0f) ? -abs_c_z : abs_c_z;
                        flowdir[c_idx][0] += diff_x * 1.0f / (1.0f + abs_c_x);
                        flowdir[c_idx][2] += diff_z * 1.0f / (1.0f + abs_c_z);
                    }
                }
            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v3,
           {/* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},

                   {"string", "stabilitymask", "_stability"},
                   {"ListObject", "perm"},
                   {"ListObject", "p_dirs"},
                   {"ListObject", "x_dirs"},

                   {"float", "seed", "15231.3"},
                   {"int", "iterations", "0"},
                   {"int", "iter", "0"},
                   {"int", "i", "0"},

                   {"int", "openborder", "0"},
                   {"float", "gridbias", "0.0"},

                   // 崩塌流淌相关
                   {"float", "repose_angle", "0.0"},
                   {"float", "quant_amt", "0.0"},
                   {"float", "flow_rate", "1.0"},
               },
               /* outputs: */
               {
                   "prim_2DGrid",
               },
               /* params: */
               {
                   //{"string", "stabilitymask", "_stability"},
               },
               /* category: */
               {
                   "erode",
               }});

// granular slump + erosion                         用于子图：Erode_Hydro                    granular + erosion
struct erode_tumble_material_v4 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        // 侵蚀主参数
        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>(); // 1 全局侵蚀率
        auto erodability = get_input<NumericObject>("erodability")->get<float>();               // 1.0 侵蚀能力
        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>();               // 0.4 侵蚀率
        auto bank_angle = get_input<NumericObject>("bank_angle")->get<float>(); // 70.0 河堤侵蚀角度
        auto seed = get_input<NumericObject>("seed")->get<float>();             // 12.34

        // 高级参数
        auto removalrate = get_input<NumericObject>("removalrate")->get<float>(); // 0.0 风化率/水吸收率
        auto max_debris_depth = get_input<NumericObject>("max_debris_depth")->get<float>(); // 5	碎屑最大深度
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();                 // 0.0

        // 侵蚀能力调整
        auto max_erodability_iteration = get_input<NumericObject>("max_erodability_iteration")->get<int>();     // 5
        auto initial_erodability_factor = get_input<NumericObject>("initial_erodability_factor")->get<float>(); // 0.5
        auto slope_contribution_factor = get_input<NumericObject>("slope_contribution_factor")->get<float>();   // 0.8

        // 河床参数
        auto bed_erosionrate_factor =
            get_input<NumericObject>("bed_erosionrate_factor")->get<float>();           // 1 河床侵蚀率因子
        auto depositionrate = get_input<NumericObject>("depositionrate")->get<float>(); // 0.01 沉积率
        auto sedimentcap = get_input<NumericObject>("sedimentcap")
            ->get<float>(); // 10.0 高度差转变为沉积物的比率 / 泥沙容量，每单位流动水可携带的泥沙量

        // 河堤参数
        auto bank_erosionrate_factor =
            get_input<NumericObject>("bank_erosionrate_factor")->get<float>(); // 1.0 河堤侵蚀率因子
        auto max_bank_bed_ratio = get_input<NumericObject>("max_bank_bed_ratio")
            ->get<float>(); // 0.5 The maximum of bank to bed water column height ratio
        // 高于这个比值的河岸将不会在侵蚀中被视为河岸，会停止侵蚀
        // 河流控制
        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>(); // 0.05 流量维持率，越高流量越稳定
        auto iterations = get_input<NumericObject>("iterations")->get<int>(); // 流淌的总迭代次数

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto iter = get_input<NumericObject>("iter")->get<int>();
        auto i = get_input<NumericObject>("i")->get<int>();
        auto openborder = get_input<NumericObject>("openborder")->get<int>();

        auto perm = get_input<ListObject>("perm")->get2<int>();
        auto p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        // 初始化网格属性
        if (!terrain->verts.has_attr("_height") || !terrain->verts.has_attr("_temp_height") ||
            !terrain->verts.has_attr("_material") || !terrain->verts.has_attr("_temp_material") ||
            !terrain->verts.has_attr("_debris") || !terrain->verts.has_attr("_temp_debris") ||
            !terrain->verts.has_attr("_sediment")) {
            zeno::log_error("Node [erode_tumble_material_v4], no such data layer named '{}' or '{}' or '{}' or '{}' or "
                            "'{}' or '{}' or '{}'.",
                            "_height", "_temp_height", "_material", "_temp_material", "_debris", "_temp_debris",
                            "_sediment");
        }
        auto &_height = terrain->verts.attr<float>("_height");
        auto &_temp_height = terrain->verts.attr<float>("_temp_height");
        auto &_material = terrain->verts.attr<float>("_material");
        auto &_temp_material = terrain->verts.attr<float>("_temp_material");
        auto &_debris = terrain->verts.attr<float>("_debris");
        auto &_temp_debris = terrain->verts.attr<float>("_temp_debris");
        auto &_sediment = terrain->verts.attr<float>("_sediment");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];
                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                int dxs[] = { 0, p_dirs[0], 0, p_dirs[0], x_dirs[0], x_dirs[1], x_dirs[0], x_dirs[1] };
                int dzs[] = { p_dirs[1], 0, p_dirs[1], 0, x_dirs[0],-x_dirs[1], x_dirs[0],-x_dirs[1] };

                if (is_red || is_green || is_blue || is_yellow || is_x_turn_x || is_x_turn_y)
                {
                    int idx = Pos2Idx(id_x, id_z, nx);
                    int dx = dxs[color - 1];
                    int dz = dzs[color - 1];
                    int bound_x = nx;
                    int bound_z = nz;
                    int clamp_x = bound_x - 1;
                    int clamp_z = bound_z - 1;

                    float i_height = _temp_height[idx];
                    float i_material = _temp_material[idx];
                    float i_debris = _temp_debris[idx];
                    float i_sediment = _sediment[idx];

                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                    if (validsource)
                    {
                        validsource = validsource || !openborder;

                        int j_idx = Pos2Idx(samplex, samplez, nx);

                        float j_height = _temp_height[j_idx];
                        float j_material = validsource ? _temp_material[j_idx] : 0.0f;
                        float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;

                        float j_sediment = validsource ? _sediment[j_idx] : 0.0f;
                        float m_diff = (j_height + j_debris + j_material) - (i_height + i_debris + i_material);
                        float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f);

                        int cidx = 0;
                        int cidz = 0;

                        float c_height = 0.0f;

                        float c_material = 0.0f;
                        float n_material = 0.0f;

                        float c_sediment = 0.0f;
                        float n_sediment = 0.0f;

                        float c_debris = 0.0f;
                        float n_debris = 0.0f;

                        float h_diff = 0.0f;

                        int c_idx = 0;
                        int n_idx = 0;
                        int dx_check = 0;
                        int dz_check = 0;
                        int is_mh_diff_same_sign = 0;

                        if (m_diff > 0.0f)
                        {
                            cidx = samplex;
                            cidz = samplez;

                            c_height = j_height;
                            c_material = j_material;
                            n_material = i_material;
                            c_sediment = j_sediment;
                            n_sediment = i_sediment;
                            c_debris = j_debris;
                            n_debris = i_debris;

                            c_idx = j_idx;
                            n_idx = idx;

                            dx_check = -dx;
                            dz_check = -dz;

                            h_diff = j_height + j_debris - (i_height + i_debris);
                            is_mh_diff_same_sign = (h_diff * m_diff) > 0.0f;
                        }
                        else
                        {
                            cidx = id_x;
                            cidz = id_z;

                            c_height = i_height;
                            c_material = i_material;
                            n_material = j_material;
                            c_sediment = i_sediment;
                            n_sediment = j_sediment;
                            c_debris = i_debris;
                            n_debris = j_debris;

                            c_idx = idx;
                            n_idx = j_idx;

                            dx_check = dx;
                            dz_check = dz;

                            h_diff = i_height + i_debris - (j_height + j_debris);
                            is_mh_diff_same_sign = (h_diff * m_diff) > 0.0f;
                        }
                        h_diff = (h_diff < 0.0f) ? -h_diff : h_diff;

                        float sum_diffs[] = { 0.0f, 0.0f };
                        float dir_probs[] = { 0.0f, 0.0f };
                        float dir_prob = 0.0f;
                        for (int diff_idx = 0; diff_idx < 2; diff_idx++)
                        {
                            for (int tmp_dz = -1; tmp_dz <= 1; tmp_dz++)
                            {
                                for (int tmp_dx = -1; tmp_dx <= 1; tmp_dx++)
                                {
                                    if (!tmp_dx && !tmp_dz)
                                        continue;

                                    int tmp_samplex = clamp(cidx + tmp_dx, 0, clamp_x);
                                    int tmp_samplez = clamp(cidz + tmp_dz, 0, clamp_z);

                                    int tmp_validsource = (tmp_samplex == (cidx + tmp_dx)) && (tmp_samplez == (cidz + tmp_dz));
                                    tmp_validsource = tmp_validsource || !openborder;
                                    int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                    float tmp_n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;
                                    float tmp_n_debris = tmp_validsource ? _temp_debris[tmp_j_idx] : 0.0f;

                                    float n_height = _temp_height[tmp_j_idx];
                                    float tmp_h_diff = n_height + tmp_n_debris - (c_height + c_debris);
                                    float tmp_m_diff = (n_height + tmp_n_debris + tmp_n_material) - (c_height + c_debris + c_material);
                                    float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;
                                    float _gridbias = gridbias;
                                    _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                    if (tmp_dx && tmp_dz)
                                        tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                    else
                                        tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                    if (tmp_diff <= 0.0f)
                                    {
                                        if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                            dir_probs[diff_idx] = tmp_diff;

                                        if (diff_idx && (tmp_diff < dir_prob))
                                            dir_prob = tmp_diff;

                                        sum_diffs[diff_idx] += tmp_diff;
                                    }
                                }
                            }

                            if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                                dir_prob = dir_probs[diff_idx] / dir_prob;
                            else
                                dir_prob = 0.0f;

                            if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                                dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                            else
                                dir_probs[diff_idx] = 0.0f;
                        }

                        float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;
                        movable_mat = clamp(movable_mat * 0.5f, 0.0f, c_material);
                        float l_rat = dir_probs[1];

                        if (quant_amt > 0.001)
                            movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                        else
                            movable_mat *= l_rat;

                        float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else
                        {
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        if (!cond)
                            diff = 0.0f;

                        float slope_cont = (delta_x > 0.0f) ? (h_diff / delta_x) : 0.0f;
                        float kd_factor = clamp((1 / (1 + (slope_contribution_factor * slope_cont))), 0.0f, 1.0f);
                        float norm_iter = clamp(((float)iter / (float)max_erodability_iteration), 0.0f, 1.0f);
                        float ks_factor = clamp((1 - (slope_contribution_factor * exp(-slope_cont))) * sqrt(dir_probs[0]) *
                                                    (initial_erodability_factor + ((1.0f - initial_erodability_factor) * sqrt(norm_iter))),
                                                0.0f, 1.0f);

                        float c_ks = global_erosionrate * erosionrate * erodability * ks_factor;

                        float n_kd = depositionrate * kd_factor;
                        n_kd = clamp(n_kd, 0.0f, 1.0f);

                        float _removalrate = removalrate;
                        float bedrock_density = 1.0f - _removalrate;
                        float abs_diff = (diff < 0.0f) ? -diff : diff;
                        float sediment_limit = sedimentcap * abs_diff;
                        float ent_check_diff = sediment_limit - c_sediment;

                        if (ent_check_diff > 0.0f)
                        {
                            float dissolve_amt = c_ks * bed_erosionrate_factor * abs_diff;
                            float dissolved_debris = min(c_debris, dissolve_amt);
                            _debris[c_idx] -= dissolved_debris;
                            _height[c_idx] -= (dissolve_amt - dissolved_debris);
                            _sediment[c_idx] -= c_sediment / 2;
                            if (bedrock_density > 0.0f)
                            {
                                float newsediment = c_sediment / 2 + (dissolve_amt * bedrock_density);
                                if (n_sediment + newsediment > max_debris_depth)
                                {
                                    float rollback = n_sediment + newsediment - max_debris_depth;
                                    rollback = min(rollback, newsediment);
                                    _height[c_idx] += rollback / bedrock_density;
                                    newsediment -= rollback;
                                }
                                _sediment[n_idx] += newsediment;
                            }
                        }
                        else
                        {
                            float c_kd = depositionrate * kd_factor;
                            c_kd = clamp(c_kd, 0.0f, 1.0f);
                            {
                                _debris[c_idx] += (c_kd * -ent_check_diff);
                                _sediment[c_idx] = (1 - c_kd) * -ent_check_diff;

                                n_sediment += sediment_limit;
                                _debris[n_idx] += (n_kd * n_sediment);
                                _sediment[n_idx] = (1 - n_kd) * n_sediment;
                            }

                            int b_idx = 0;
                            int r_idx = 0;
                            float b_material = 0.0f;
                            float r_material = 0.0f;
                            float b_debris = 0.0f;
                            float r_debris = 0.0f;
                            float r_sediment = 0.0f;

                            if (is_mh_diff_same_sign)
                            {
                                b_idx = c_idx;
                                r_idx = n_idx;

                                b_material = c_material;
                                r_material = n_material;

                                b_debris = c_debris;
                                r_debris = n_debris;

                                r_sediment = n_sediment;
                            }
                            else
                            {
                                b_idx = n_idx;
                                r_idx = c_idx;

                                b_material = n_material;
                                r_material = c_material;

                                b_debris = n_debris;
                                r_debris = c_debris;

                                r_sediment = c_sediment;
                            }

                            float erosion_per_unit_water = global_erosionrate * erosionrate * bed_erosionrate_factor * erodability * ks_factor;
                            if (r_material != 0.0f &&
                                (b_material / r_material) < max_bank_bed_ratio &&
                                r_sediment > (erosion_per_unit_water * max_bank_bed_ratio))
                            {
                                float height_to_erode = global_erosionrate * erosionrate * bank_erosionrate_factor * erodability * ks_factor;

                                float _bank_angle = bank_angle;

                                _bank_angle = clamp(_bank_angle, 0.0f, 90.0f);
                                float safe_diff = _bank_angle < 90.0f ? tan(_bank_angle * M_PI / 180.0) * delta_x : 1e10f;
                                float target_height_removal = (h_diff - safe_diff) < 0.0f ? 0.0f : h_diff - safe_diff;

                                float dissolve_amt = clamp(height_to_erode, 0.0f, target_height_removal);
                                float dissolved_debris = min(b_debris, dissolve_amt);

                                _debris[b_idx] -= dissolved_debris;

                                float division = 1 / (1 + safe_diff);

                                _height[b_idx] -= (dissolve_amt - dissolved_debris);

                                if (bedrock_density > 0.0f)
                                {
                                    float newdebris = (1 - division) * (dissolve_amt * bedrock_density);
                                    if (b_debris + newdebris > max_debris_depth)
                                    {
                                        float rollback = b_debris + newdebris - max_debris_depth;
                                        rollback = min(rollback, newdebris);
                                        _height[b_idx] += rollback / bedrock_density;
                                        newdebris -= rollback;
                                    }
                                    _debris[b_idx] += newdebris;

                                    newdebris = division * (dissolve_amt * bedrock_density);

                                    if (r_debris + newdebris > max_debris_depth)
                                    {
                                        float rollback = r_debris + newdebris - max_debris_depth;
                                        rollback = min(rollback, newdebris);
                                        _height[b_idx] += rollback / bedrock_density;
                                        newdebris -= rollback;
                                    }
                                    _debris[r_idx] += newdebris;
                                }
                            }
                        }

                        _material[idx] = i_material + diff;
                        _material[j_idx] = j_material - diff;
                    }
                }
            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v4,
           {/* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},

                   {"ListObject", "perm"},
                   {"ListObject", "p_dirs"},
                   {"ListObject", "x_dirs"},

                   {"float", "seed", "12.34"},
                   {"int", "iterations", "40"}, // 流淌的总迭代次数
                   {"int", "iter", "0"},
                   {"int", "i", "0"},

                   {"int", "openborder", "0"},
                   {"float", "gridbias", "0.0"},

                   // 侵蚀主参数
                   {"float", "global_erosionrate", "1.0"}, // 全局侵蚀率
                   {"float", "erodability", "1.0"},        // 侵蚀能力
                   {"float", "erosionrate", "0.4"},        // 侵蚀率
                   {"float", "bank_angle", "70.0"},        // 河堤侵蚀角度

                   // 高级参数
                   {"float", "removalrate", "0.1"},      // 风化率/水吸收率
                   {"float", "max_debris_depth", "5.0"}, // 碎屑最大深度

                   // 侵蚀能力调整
                   {"int", "max_erodability_iteration", "5"},      // 最大侵蚀能力迭代次数
                   {"float", "initial_erodability_factor", "0.5"}, // 初始侵蚀能力因子
                   {"float", "slope_contribution_factor", "0.8"}, // “地面斜率”对“侵蚀”和“沉积”的影响，“地面斜率大” -> 侵蚀因子大，沉积因子小

                   // 河床参数
                   {"float", "bed_erosionrate_factor", "1.0"}, // 河床侵蚀率因子
                   {"float", "depositionrate", "0.01"},        // 沉积率
                   {"float", "sedimentcap", "10.0"}, // 高度差转变为沉积物的比率 / 泥沙容量，每单位流动水可携带的泥沙量

                   // 河堤参数
                   {"float", "bank_erosionrate_factor", "1.0"}, // 河堤侵蚀率因子
                   {"float", "max_bank_bed_ratio", "0.5"}, // 高于这个比值的河岸将不会在侵蚀中被视为河岸，会停止侵蚀

                   // 河网控制
                   {"float", "quant_amt", "0.05"}, // 流量维持率，越高河流流量越稳定
               },
               /* outputs: */
               {
                   "prim_2DGrid",
               },
               /* params: */
               {

               },
               /* category: */
               {
                   "erode",
               }});

//                                                  还未实现                                granular + erosion + flow

// smooth flow
struct erode_smooth_flow : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto smooth_rate = get_input<NumericObject>("smoothRate")->get<float>();
        auto flowName = get_input2<std::string>("flowName");

        // 初始化网格属性
        auto &flow = terrain->verts.attr<float>(flowName);
        auto &_lap = terrain->verts.add_attr<float>("_lap");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);

                float net_diff = 0.0f;
                net_diff += flow[idx - 1 * (id_x > 0)];
                net_diff += flow[idx + 1 * (id_x < nx - 1)];
                net_diff += flow[idx - nx * (id_z > 0)];
                net_diff += flow[idx + nx * (id_z < nz - 1)];

                net_diff *= 0.25f;
                net_diff -= flow[idx];

                _lap[idx] = net_diff;
            }
        }

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);

                float net_diff = 0.0f;
                net_diff += _lap[idx - 1 * (id_x > 0)];
                net_diff += _lap[idx + 1 * (id_x < nx - 1)];
                net_diff += _lap[idx - nx * (id_z > 0)];
                net_diff += _lap[idx + nx * (id_z < nz - 1)];

                net_diff *= 0.25f;
                net_diff -= _lap[idx];

                flow[idx] -= smooth_rate * 0.5f * net_diff;
            }
        }

        terrain->verts.erase_attr("_lap");

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_smooth_flow,
           {/* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},
                   {"float", "smoothRate", "1.0"},
                   {"string", "flowName", "flow"},
               },
               /* outputs: */
               {
                   "prim_2DGrid",
               },
               /* params: */
               {

               },
               /* category: */
               {
                   "erode",
               }});

// ######################################################
// ######################################################
// erode ################################################


struct erode_terrainHiMeLo : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto& ud = terrain->userData();
        if ((!ud.has<float>("hi")) ||
            (!ud.has<float>("me")) ||
            (!ud.has<float>("lo")))
        {
            zeno::log_error("no such UserData named '{}' or '{}' or '{}'.", "hi", "me", "lo");
        }

        auto attrName = get_input<StringObject>("attrName")->get();
        if (!terrain->verts.has_attr(attrName))
        {
            zeno::log_error("no such data named '{}'.", attrName);
        }
        auto& attr = terrain->verts.attr<float>(attrName);

        float hi = -10000000;
        float me = 0;
        float lo = 10000000;
        float all = 0;

#pragma omp parallel for
        for (int i = 0; i < attr.size(); i++)
        {
            if (attr[i] > hi)
            {
                hi = attr[i];
            }

            if (attr[i] < lo)
            {
                lo = attr[i];
            }

            all += attr[i];
        }

        ud.set2("hi", hi);
        ud.set2("lo", lo);
        ud.set2("me", all / attr.size());

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_terrainHiMeLo,
           { /* inputs: */ {
                   {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},
                   { "string", "attrName", "fbm" },
               }, /* outputs: */ {
                   "prim_2DGrid",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               } });


float fit(const float data, const float ss, const float se, const float ds, const float de) {
    float b = std::numeric_limits<float>::epsilon();
    b = max(abs(se - ss), b);
    b = se - ss >= 0 ? b : -b;
    float alpha = (data - ss)/b;
    return ds + (de - ds) * alpha;
}

float chramp(const float inputData) {
    float data = min(max(inputData, 0.0), 1.0);
    float outputData = 0;
    if (data <= 0.1) {
        outputData = fit(data, 0, 0.1, 0, 1);
    } else if (data >= 0.9) {
        outputData = fit(data, 0.9, 1.0, 1, 0);
    } else {
        outputData = 1;
    }
    return outputData;
}


struct HF_maskByFeature : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("HeightField");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

        // 获取面板参数
        auto heightLayer = get_input2<std::string>("height_layer");
        auto maskLayer = get_input2<std::string>("mask_layer");
        auto smoothRadius = get_input2<int>("smooth_radius");
        auto invertMask = get_input2<bool>("invert_mask");

        auto useSlope = get_input2<bool>("use_slope");
        auto minSlope = get_input2<float>("min_slopeangle");
        auto maxSlope = get_input2<float>("max_slopeangle");
        auto curve_slope = get_input_prim<CurvesData>("slope_ramp");

        auto useDir = get_input2<bool>("use_direction");
        auto goalAngle = get_input2<float>("goal_angle");
        auto angleSpread = get_input2<float>("angle_spread");
        auto curve_dir = get_input_prim<CurvesData>("dir_ramp");

        auto useHeight = get_input2<bool>("use_height");
        auto minHeight = get_input2<float>("min_height");
        auto maxHeight = get_input2<float>("max_height");
        auto curve_height = get_input_prim<CurvesData>("height_ramp");

        // 初始化网格属性
        if (!terrain->verts.has_attr(heightLayer) || !terrain->verts.has_attr(maskLayer)) {
            zeno::log_error("Node [HF_maskByFeature], no such data layer named '{}' or '{}'.",
                            heightLayer, maskLayer);
        }
        auto &height = terrain->verts.attr<float>(heightLayer);
        auto &mask = terrain->verts.attr<float>(maskLayer);

        auto &_grad = terrain->verts.add_attr<vec3f>("_grad");
        std::fill(_grad.begin(), _grad.end(), vec3f(0,0,0));

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////
#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                int idx_xl, idx_xr, idx_zl, idx_zr;
                int scale_x = 0;
                int scale_z = 0;

                if (id_x == 0) {
                    idx_xl = idx;
                    idx_xr = Pos2Idx(id_x + 1, id_z, nx);
                    scale_x = 1;
                } else if (id_x == nx - 1) {
                    idx_xl = Pos2Idx(id_x - 1, id_z, nx);
                    idx_xr = idx;
                    scale_x = 1;
                } else {
                    idx_xl = Pos2Idx(id_x - 1, id_z, nx);
                    idx_xr = Pos2Idx(id_x + 1, id_z, nx);
                    scale_x = 2;
                }

                if (id_z == 0) {
                    idx_zl = idx;
                    idx_zr = Pos2Idx(id_x, id_z + 1, nx);
                    scale_z = 1;
                } else if (id_z == nz - 1) {
                    idx_zl = Pos2Idx(id_x, id_z - 1, nx);
                    idx_zr = idx;
                    scale_z = 1;
                } else {
                    idx_zl = Pos2Idx(id_x, id_z - 1, nx);
                    idx_zr = Pos2Idx(id_x, id_z + 1, nx);
                    scale_z = 2;
                }

                // debug
//                if(id_x >= 570 && id_z >= 570)
//                {
//                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
//                    printf("nx = %i, nz = %i\n", nx, nz);
//                    printf("id_x = %i, id_z = %i\n", id_x, id_z);
//                    printf("scale_x = %i, scale_z = %i, cellSize = %f\n", scale_x, scale_z, cellSize);
//                    printf("-------------------\n");
//                    printf("idx_xr = %i, idx_xl = %i\n", idx_xr, idx_xl);
//                    printf("idx_zr = %i, idx_zl = %i\n", idx_zr, idx_zl);
//                    printf("-------------------\n");
//                    //printf("height[idx_xr] = %f, height[idx_xl] = %f\n", height[idx_xr], height[idx_xl]);
//                    //printf("height[idx_zr] = %f, height[idx_zl] = %f\n", height[idx_zr], height[idx_zl]);
//                    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
//                }

                _grad[idx][0] = (height[idx_xr] - height[idx_xl]) / (float(scale_x) * cellSize);
                _grad[idx][2] = (height[idx_zr] - height[idx_zl]) / (float(scale_z) * cellSize);

                vec3f dx = normalizeSafe(vec3f(1, 0, _grad[idx][0]));
                vec3f dy = normalizeSafe(vec3f(0, 1, _grad[idx][2]));
                vec3f n = normalizeSafe(cross(dx, dy));

                mask[idx] = 1;
                if (!useSlope &&
                    !useDir &&
                    !useHeight)// &&
//                    //!useCurvature &&
//                    //!useOcclusion)
                {
//                    mask[idx] = 0;
                }

                if (useSlope) {
                    float slope = 180 * acos(n[2]) / M_PI;
                    slope = fit(slope, minSlope, maxSlope, 0, 1);
//                    slope = chramp(slope);
                    slope = curve_slope->eval(slope);
                    mask[idx] *= slope;
                }

                if (useDir) {
                    float direction = 180 * atan2(n[0], n[1]) / M_PI;
                    direction -= goalAngle;
                    direction -= 360 * floor(direction / 360);   // Get in range -180 to 180
                    direction -= 180;
                    direction = fit(direction, -angleSpread, angleSpread, 0, 1);
//                    direction = chramp(direction);
                    direction = curve_dir->eval(direction);
                    mask[idx] *= direction;
                }

                if (useHeight)
                {
                    float h = fit(height[idx], minHeight, maxHeight, 0, 1);
//                    mask[idx] *= chramp(h);
                    mask[idx] *= curve_height->eval(h);
                }

                if(invertMask)
                {
                    mask[idx] = min(max(mask[idx], 0), 1);
                    mask[idx] = 1 - mask[idx];
                }
            }
        }
        terrain->verts.erase_attr("_grad");
        set_output("HeightField", std::move(terrain));
    }
};
ZENDEFNODE(HF_maskByFeature,
           {/* inputs: */ {
                   {"", "HeightField", "", zeno::Socket_ReadOnly},
                   {"bool", "invert_mask", "0"},
                   {"string", "height_layer", "height"},
                   {"string", "mask_layer", "mask"},
                   {"int", "smooth_radius", "1"},
                   {"bool", "use_slope", "0"},
                   {"float", "min_slopeangle", "0"},
                   {"float", "max_slopeangle", "90"},
                   {"curve", "slope_ramp"},
                   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   {"bool", "use_direction", "0"},
                   {"float", "goal_angle", "0"},
                   {"float", "angle_spread", "30"},
                   {"curve", "dir_ramp"},
                   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   {"bool", "use_height", "0"},
                   {"float", "min_height", "0.5"},
                   {"float", "max_height", "1"},
                   {"curve", "height_ramp"},
               },
               /* outputs: */
               {
                   "HeightField",
               },
               /* params: */
               {
               },
               /* category: */
               {
                   "erode",
               }});

struct HF_rotate_displacement_2d : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto& var = terrain->verts.attr<vec3f>("var"); // hardcode
        auto& pos = terrain->verts.attr<vec3f>("tempPos"); // hardcode

        auto angle = get_input<NumericObject>("Rotate Displacement")->get<float>();
        float gl_angle = glm::radians(angle);
        glm::vec3 gl_axis(0.0, 1.0, 0.0); // hardcode
        glm::quat gl_quat = glm::angleAxis(gl_angle, gl_axis);

#pragma omp parallel for
        for (int i = 0; i < terrain->verts.size(); i++)
        {
            glm::vec3 ret{};// = glm::vec3(0, 0, 0);
            ret = glm::rotate(
                    gl_quat,
                    glm::vec3(var[i][0], var[i][1], var[i][2])
            );
            pos[i] -= vec3f(ret.x, ret.y, ret.z);
        }

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(HF_rotate_displacement_2d,
           { /* inputs: */ {
               {"", "prim_2DGrid", "", zeno::Socket_ReadOnly},
               {"float", "Rotate Displacement", "0"}
           }, /* outputs: */ {
               "prim_2DGrid",
           }, /* params: */ {
           }, /* category: */ {
               "erode",
           } });

struct HF_remap : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim");
        auto remapLayer = get_input2<std::string>("remap layer");
        if (!terrain->verts.has_attr(remapLayer)) {
            zeno::log_error("Node [HF_remap], no such data layer named '{}'.",
                            remapLayer);
        }
        auto& var = terrain->verts.attr<float>(remapLayer);
        auto autoCompute = get_input2<bool>("Auto Compute input range");
        auto inMin = get_input2<float>("input min");
        auto inMax = get_input2<float>("input max");
        auto outMin = get_input2<float>("output min");
        auto outMax = get_input2<float>("output max");
        auto curve = get_input_prim<CurvesData>("remap ramp");
        auto clampMin = get_input2<bool>("clamp min");
        auto clampMax = get_input2<bool>("clamp max");

        if (autoCompute) {
            inMin = zeno::parallel_reduce_array<float>(var.size(), var[0], [&] (size_t i) -> float { return var[i]; },
            [&] (float i, float j) -> float { return zeno::min(i, j); });
            inMax = zeno::parallel_reduce_array<float>(var.size(), var[0], [&] (size_t i) -> float { return var[i]; },
            [&] (float i, float j) -> float { return zeno::max(i, j); });
        }
#pragma omp parallel for
        for (int i = 0; i < terrain->verts.size(); i++)
        {
            if (var[i] < inMin)
            {
                if (clampMin)
                {
                    var[i] = outMin;
                }
                else
                {
                    var[i] -= inMin;
                    var[i] += outMin;
                }
            }
            else if (var[i] > inMax)
            {
                if (clampMax)
                {
                    var[i] = outMax;
                }
                else
                {
                    var[i] -= inMax;
                    var[i] += outMax;
                }
            }
            else
            {
                var[i] = fit(var[i], inMin, inMax, 0, 1);
                var[i] = curve->eval(var[i]);
                var[i] = fit(var[i], 0, 1, outMin, outMax);
            }
            if (remapLayer == "height"){
                terrain->verts.attr<vec3f>("pos")[i][1] = var[i];
            }
        }

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(HF_remap,
           { /* inputs: */ {
               {"", "prim", "", zeno::Socket_ReadOnly},
               {"string", "remap layer", "height"},
               {"bool", "Auto Compute input range", "0"},
               {"float", "input min", "0"},
               {"float", "input max", "1"},
               {"float", "output min", "0"},
               {"float", "output max", "1"},
               {"curve", "remap ramp"},
               {"bool", "clamp min", "0"},
               {"bool", "clamp max", "0"}
           }, /* outputs: */ {
               "prim",
           }, /* params: */ {
           }, /* category: */ {
               "deprecated",
           } });

struct HF_maskbyOcclusion : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim");

        auto invert_mask = get_input2<bool>("invert mask");
        auto view_radius = get_input2<int>("view distance");
        auto step_scale = get_input2<float>("step scale");
        auto axis_count = get_input2<int>("num of searches");
        auto dohemisphere = get_input2<bool>("dohemisphere");

        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        vec3f p0 = pos[0];
        vec3f p1 = pos[1];
        float cellSize = length(p1 - p0);

//        auto heightLayer = get_input2<std::string>("height_layer");
//        if (!terrain->verts.has_attr(heightLayer)) {
//            zeno::log_error("Node [HF_maskByFeature], no such data layer named '{}'.",
//                            heightLayer);
//        }
        auto &height = terrain->verts.attr<float>("height");

        auto &ao = terrain->verts.add_attr<float>("ao");
        std::fill(ao.begin(), ao.end(), 0.0);

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++) {
            for (int id_x = 0; id_x < nx; id_x++) {
                int idx = Pos2Idx(id_x, id_z, nx);

                float h_start = height[idx];

                // Lower bound the step scale to at least 0.5
                step_scale = max(step_scale, 0.5f);
                // If we have a finite view radius, upper bound the step scale
                // at half that view radius so each ray can get at least two samples.
                if (view_radius) step_scale = min(step_scale, 0.499f * view_radius);

                // The step limit is the number of world units that fit in the
                // view radius. If the view radius is <= 0, we use the full world
                // size as the limit.
                int step_limit = view_radius > 0.0f ?
                                 ceil(view_radius / (cellSize * step_scale)) :
                                 ceil((hypot(nx, nz)) / step_scale);

                // Calculate the sweep angle for each concavity axis
                float sweep_angle = 3.14159f / (float) max(axis_count, 1);
                float cur_angle = 0.0f;

                // Accumulate field of view
                float total_fov = 0.0f;
                float successful_rays = 0;


                // Calculate the step in the x and y direction
                // based on the light direction
                float z_step = sin(cur_angle);
                float x_step = cos(cur_angle);
                x_step *= step_scale;
                z_step *= step_scale;

                // Sweep a full circle around the point
                for (int i = 0; i < axis_count; i++) {
                    // Calculate the step in the x and y direction
                    // based on the light direction
                    float z_step = sin(cur_angle);
                    float x_step = cos(cur_angle);
                    x_step *= step_scale;
                    z_step *= step_scale;

                    float speed = hypot(x_step, z_step) * cellSize;

                    // Walk a line that intersects this point.
                    // We start from our point and walk twice in opposite directions.
                    for (int j = 0; j < 2; j++) {
                        // Start from the current point
                        float x = id_x + x_step;
                        float z = id_z + z_step;
                        float distance = speed;
                        int steps = 1;

                        float start_slope;

                        // Find the max slope in our current direction
                        float finalslope = 0.0f;
                        float maxslope = -1e10f;
                        while (steps < step_limit &&
                               x > 0 && x < (nx-1) &&
                               z > 0 && z < (nz-1)) {
                            // Calculate the height at our current position
                            // by doing a bilerp with the nearest 4 points

                            // tofix: out of bound possibility?
                            // clamp to boundaries
                            x = clamp(x, 0.0f, nx-1);
                            z = clamp(z, 0.0f, nz-1);

                            const int int_x = (int)floor(x);
                            const int int_z = (int)floor(z);

                            const float fract_x = x - int_x;
                            const float fract_z = z - int_z;

                            int srcidx = Pos2Idx(int_x, int_z, nx);
                            const float i00 = height[srcidx];
                            const float i10 = height[srcidx + 1];
                            const float i01 = height[srcidx + nz];
                            const float i11 = height[srcidx + nz + 1];

                            float h_current = (i00 * (1-fract_x) + i10 * (fract_x)) * (1-fract_z) +
                                              (i01 * (1-fract_x) + i11 * (fract_x)) * (  fract_z);

                            // Calculate the slope
                            float dh = h_current - h_start;
                            float curslope = dh / distance;
                            if (steps == 1) start_slope = curslope;
                            maxslope = max(maxslope, curslope);
                            finalslope = maxslope;

                            x += x_step;
                            z += z_step;
                            distance += speed;
                            steps++;
                        }

                        if (steps > 1) {
                            successful_rays += 1.0f;

                            // Add the cosine of the max slope to our field of view.
                            // The cosine of the slope is essentially a measure of
                            // the 'breadth' of the view from the vertical.
                            // No light comes from below the horizon, so a negative
                            // slope is exposed to as much light as a zero slope.
                            if (dohemisphere) start_slope = 0;
                            float slope = max(start_slope, finalslope);
                            total_fov += 1 - slope / hypot(slope, 1);
                        }

                        // Walk in the reverse direction next iteration
                        x_step = -x_step;
                        z_step = -z_step;
                    }

                    // Proceed to the next sweep angle
                    cur_angle += sweep_angle;
                }

                // Normalize the value
                if (successful_rays != 0)
                    total_fov /= successful_rays;

                // The bounds of this value should already be in [0, 1], but do
                // a last clamp anyway to cull any possible floating point error
                total_fov = clamp(total_fov, 0.0f, 1.0f);

                ao[idx] = invert_mask ? 1-total_fov : total_fov;
            }
        }



        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(HF_maskbyOcclusion,
           { /* inputs: */ {
                   {"", "prim", "", zeno::Socket_ReadOnly},
                   {"bool", "invert mask", "0"},
                   {"int", "view distance", "200"},
                   {"float", "step scale", "1"},
                   {"int", "num of searches", "16"},
                   {"bool", "dohemisphere", "0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               } });
} // namespace
} // namespace zeno
