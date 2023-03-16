//
// WangBo 2022/11/28.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <random>

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

// 降水/蒸发
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
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

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
#pragma omp parallel for
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
ZENDEFNODE(erode_value2cond, {/* inputs: */ {
                                  "prim_2DGrid",
                                  {"float", "value", "1.0"}, // 0.0 ~ 1.0
                                  {"float", "seed", "0.0"},
                              },
                              /* outputs: */
                              {
                                  "prim_2DGrid",
                              },
                              /* params: */ {}, /* category: */
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
            list->arr.push_back(num);
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(erode_rand_color, {/* inputs: */ {
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
            list->arr.push_back(num);
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(erode_rand_dir, {/* inputs: */ {
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

// 热侵蚀
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
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 获取面板参数
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();
        auto cut_angle = get_input<NumericObject>("cut_angle")->get<float>();
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
#pragma omp parallel for
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

                                float _gridbias = clamp(gridbias, -1.0f, 1.0f);

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
                            float _cut_angle = clamp(cut_angle, 0.0f, 90.0f);
                            float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f);
                            float height_removed = _cut_angle < 90.0f ? tan(_cut_angle * M_PI / 180) * delta_x : 1e10f;
                            float height_diff = abs_h_diff - height_removed;
                            if (height_diff < 0.0f)
                                height_diff = 0.0f;
                            float prob = ((n_debris + c_debris) != 0.0f) ? clamp((height_diff / (n_debris + c_debris)), 0.0f, 1.0f) : 1.0f;
                            unsigned int cutoff = (unsigned int)(prob * 4294967295.0);
                            unsigned int randval = erode_random(seed * 3.14, (idx + nx * nz) * 8 + color + iterseed);
                            int do_erode = randval < cutoff;

                            float height_removal_amt = do_erode * clamp(global_erosionrate * erosionrate * erodability, 0.0f, height_diff);

                            _height[c_idx] -= height_removal_amt;

                            float bedrock_density = 1.0f - (removalrate);
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
ZENDEFNODE(erode_tumble_material_v0, {/* inputs: */ {
                                          "prim_2DGrid",

                                          {"ListObject", "perm"},
                                          {"ListObject", "p_dirs"},
                                          {"ListObject", "x_dirs"},

                                          {"float", "seed", "9676.79"},
                                          {"int", "iterations", "0"},
                                          {"int", "iter", "0"},
                                          {"int", "i", "0"},

                                          {"int", "openborder", "0"},
                                          {"float", "gridbias", "0.0"},

                                          {"float", "cut_angle", "35"},
                                          {"float", "global_erosionrate", "1.0"},
                                          {"float", "erosionrate", "0.03"},
                                          {"float", "erodability", "0.4"},
                                          {"float", "removalrate", "0.7"},
                                          {"float", "maxdepth", "5.0"},

                                      },
                                      /* outputs: */
                                      {
                                          "prim_2DGrid",
                                      },
                                      /* params: */ {}, /* category: */
                                      {
                                          "erode",
                                      }});

// 崩塌
struct erode_tumble_material_v2 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////

        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz"))) zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

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
        if (!terrain->verts.has_attr(stablilityMaskName)) {
            auto &_sta = terrain->verts.add_attr<float>(stablilityMaskName);
            std::fill(_sta.begin(), _sta.end(), 0.0);
        }
        auto &stabilitymask = terrain->verts.attr<float>(stablilityMaskName);

        if (!terrain->verts.has_attr("height") ||
            !terrain->verts.has_attr("_material") ||
            !terrain->verts.has_attr("_temp_material")) {
            zeno::log_error("Node [erode_tumble_material_v2], no such data layer named '{}' or '{}' or '{}'.",
                            "height", "_material", "_temp_material");
        }
        auto &height            = terrain->verts.attr<float>("height");
        auto &_material         = terrain->verts.attr<float>("_material");
        auto &_temp_material    = terrain->verts.attr<float>("_temp_material");


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
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

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v2, {/* inputs: */ {
                                          "prim_2DGrid",

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

// 崩塌 + flow
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
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

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
        if (!terrain->verts.has_attr(stablilityMaskName))
        {
            auto &_sta = terrain->verts.add_attr<float>(stablilityMaskName);
            std::fill(_sta.begin(), _sta.end(), 0.0);
        }
        auto &stabilitymask = terrain->verts.attr<float>(stablilityMaskName);

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
        for (int id_z = 0; id_z < nz; id_z++) {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++) {
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
ZENDEFNODE(erode_tumble_material_v3, {/* inputs: */ {
                                          "prim_2DGrid",

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
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

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
        for (int id_z = 0; id_z < nz; id_z++) {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++) {
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
        for (int id_z = 0; id_z < nz; id_z++) {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++) {
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
ZENDEFNODE(erode_smooth_flow, {/* inputs: */ {
                                   "prim_2DGrid",
                                   {"float", "smoothRate", "1.0"},
                                   {"string", "flowName", "flow"},
                               },
                               /* outputs: */
                               {
                                   "prim_2DGrid",
                               },
                               /* params: */ {}, /* category: */
                               {
                                   "erode",
                               }});

// 崩塌 + 侵蚀
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
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

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
#pragma omp parallel for
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
                "prim_2DGrid",

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
                {"float", "slope_contribution_factor",
                 "0.8"}, // “地面斜率”对“侵蚀”和“沉积”的影响，“地面斜率大” -> 侵蚀因子大，沉积因子小

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
            /* params: */ {}, /* category: */
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

        float hi = 0;
        float me = 0;
        float lo = 0;
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
            "prim_2DGrid",
            { "string", "attrName", "fbm" },
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
        }, /* category: */ {
            "erode",
        } });






} // namespace
} // namespace zeno