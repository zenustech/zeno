//
// Created by WangBo on 2022/11/28.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <random>
//#include <time.h>

namespace zeno
{
namespace
{

///////////////////////////////////////////////////////////////////////////////
// 2022.10.10 Erode
///////////////////////////////////////////////////////////////////////////////

int Pos2Idx(const int x, const int z, const int nx)
{
    return z * nx + x;
}

static unsigned int erode_random(float seed, int idx)
{
    unsigned int s = *(unsigned int*)(&seed);
    s ^= idx << 3;
    s *= 179424691; // a magic prime number
    s ^= s << 13 | s >> (32 - 13);
    s ^= s >> 17 | s << (32 - 17);
    s ^= s << 23;
    s *= 179424691;
    return s;
}

struct erode_precipitation : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化数据层
        ////////////////////////////////////////////////////////////////////////////////////////

        // water 水层比较特别，可以从内部创建，但是要保留从外部接受水层的能力，所以只能创建的时候初始化
        auto waterLayerName = get_input<StringObject>("waterLayerName")->get();
        if (!terrain->verts.has_attr(waterLayerName))
        {
            // 要求 water 从外部读取
            zeno::log_error("no such data layer named '{}'.", waterLayerName);
        }
        auto& water = terrain->verts.attr<float>(waterLayerName); // 读取外部数据


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 创建临时属性，将外部数据拷贝到临时属性，我们将使用临时属性进行计算
        ////////////////////////////////////////////////////////////////////////////////////////
        auto& _added_water = terrain->verts.add_attr<float>("_added_water");

        // _added_water 不依赖之前的计算结果，天上降下来的雨跟地面的积水没有关系，所以不需要从外部拷贝数据


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 获取计算所需参数
        ////////////////////////////////////////////////////////////////////////////////////////

        // 获取雨水分布参数
        auto amount = get_input<NumericObject>("amount")->get<float>();
        auto density = get_input<NumericObject>("density")->get<float>();
        auto seed = get_input<NumericObject>("seed")->get<float>();

        // 雨滴扩散和模糊的算法，目前还没有实现
        /*
        auto expandradius = get_input<NumericObject>("expandradius")->get<float>();
        float multiplier = 1 + floor(expandradius / cellSize);
        density *= 1 / (multiplier * multiplier);
        */


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

        // 下雨(amount>0) 或 蒸发(amount<0)
#pragma omp parallel for
        for (int z = 0; z < nz; z++)
        {
#pragma omp parallel for
            for (int x = 0; x < nx; x++)
            {
                int idx = Pos2Idx(x, z, nx);

                int cond = 0;
                if (density >= 1.0f)
                {
                    cond = 1;
                }
                else
                {
                    density = clamp(density, 0, 1);
                    unsigned int cutoff = (unsigned int)(density * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                    unsigned int randval = erode_random(seed, idx + nx * nz);
                    cond = randval < cutoff;
                }

                _added_water[idx] = amount * cond;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 将计算结果返回给外部数据，并删除临时属性
        ////////////////////////////////////////////////////////////////////////////////////////
#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                water[idx] = max(0.0f, water[idx] + _added_water[idx]); // 计算结果返回给外部数据

                if (visualEnable)
                {
                    float coef = min(1, (water[idx] / 1.0));
                    attr_color[idx] = (1 - coef) * attr_color[idx] + coef * vec3f(0.15, 0.45, 0.9);
                }
            }
        }
        terrain->verts.erase_attr("_added_water");

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_precipitation,
    { /* inputs: */ {
            "prim_2DGrid",
            {"string", "waterLayerName", "water"},
            {"float", "amount", "0.3"},
            {"float", "density", "0.05"},
            {"float", "seed", "12.34"},
            //{"float", "expandradius", "0.0"},
            //{"float", "blurradius", "0.0"},
            {"int", "visualEnable", "1"},
        }, /* outputs: */ {
            "prim_2DGrid"
        }, /* params: */ {
        }, /* category: */ {
            "deprecated"
        } });
// 上面的（降水）节点 erode_precipitation 可以废弃了，由下面的 
// erode_value2cond
// 节点代替
struct erode_value2cond : INode {
    void apply() override {

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        ///////////////////////////////////////////////////////////////////////

        auto value = get_input<NumericObject>("value")->get<float>();
        auto seed = get_input<NumericObject>("seed")->get<float>();

        if (!terrain->verts.has_attr("cond"))
        {
            terrain->verts.add_attr<float>("cond");
        }
        auto& attr_cond = terrain->verts.attr<float>("cond");
        std::fill(attr_cond.begin(), attr_cond.end(), 0.0);

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
ZENDEFNODE(erode_value2cond,
    { /* inputs: */ {
        "prim_2DGrid",
        { "float", "value", "1.0" }, // 0.0 ~ 1.0
        { "float", "seed", "0.0" },
    }, /* outputs: */{
        "prim_2DGrid",
    }, /* params: */{
    }, /* category: */{
        "erode",
    } });
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// 此节点描绘（热侵蚀）过程
struct erode_compute_erosion : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化数据层
        ////////////////////////////////////////////////////////////////////////////////////////

        // height 和 debris 只能从外部获取，不能从内部创建，因为本节点要被嵌入循环中
        // 初始化 height 和 debris 的过程 应该 在此节点的外部
        auto heightLayerName = get_input<StringObject>("heightLayerName")->get();
        auto debrisLayerName = get_input<StringObject>("debrisLayerName")->get();
        if (!terrain->verts.has_attr(heightLayerName) || !terrain->verts.has_attr(debrisLayerName))
        {
            // height 和 debris 数据要从外面读取，所以属性中要有 height 和 debris
            zeno::log_error("no such data layer named '{}' or '{}'.", heightLayerName, debrisLayerName);
        }
        auto& height = terrain->verts.attr<float>(heightLayerName);   // 读取外部数据
        auto& debris = terrain->verts.attr<float>(debrisLayerName);


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 创建临时属性，将外部数据拷贝到临时属性，我们将使用临时属性进行计算
        ////////////////////////////////////////////////////////////////////////////////////////

        auto& _height = terrain->verts.add_attr<float>("_height");              // 计算用的临时属性
        auto& _debris = terrain->verts.add_attr<float>("_debris");
        auto& _temp_height = terrain->verts.add_attr<float>("_temp_height");    // 备份用的临时属性
        auto& _temp_debris = terrain->verts.add_attr<float>("_temp_debris");

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                _height[idx] = height[idx];     // 外部数据拷贝到临时属性
                _debris[idx] = debris[idx];     // 外部数据拷贝到临时属性
                _temp_height[idx] = 0;          // 正式计算前会把 _height 的数据保存在这里
                _temp_debris[idx] = 0;          // 正式计算前会把 _debris 的数据保存在这里
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 获取计算所需参数
        ////////////////////////////////////////////////////////////////////////////////////////

        std::uniform_real_distribution<float> distr(0.0, 1.0);                  // 设置随机分布

        auto seed = get_input<NumericObject>("seed")->get<float>();

        auto iterations = get_input<NumericObject>("iterations")->get<int>();   // 获取迭代次数

        auto openborder = get_input<NumericObject>("openborder")->get<int>();   // 获取边界标记

        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();

        auto cut_angle = get_input<NumericObject>("cut_angle")->get<float>();

        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>();

        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>();

        auto erodability = get_input<NumericObject>("erodability")->get<float>();

        auto removalrate = get_input<NumericObject>("removalrate")->get<float>();

        auto maxdepth = get_input<NumericObject>("maxdepth")->get<float>();

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

        for (int iter = 1; iter <= iterations; iter++)
        {
            // 准备随机数组，每次迭代都都会有变化，用于网格随机取半，以及产生随机方向
            int perm[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
            for (int i = 0; i < 8; i++)
            {
                vec2f vec;
                std::mt19937 mt(iterations * iter * 8 * i + i);	// 梅森旋转算法
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

            int p_dirs[] = { -1, -1 };
            for (int i = 0; i < 2; i++)
            {
                std::mt19937 mt(iterations * iter * 2 * i + i);
                float rand_val = distr(mt);
                if (rand_val > 0.5)
                {
                    p_dirs[i] = 1;
                }
                else
                {
                    p_dirs[i] = -1;
                }
            }

            int x_dirs[] = { -1, -1 };
            for (int i = 0; i < 2; i++)
            {
                std::mt19937 mt(iterations * iter * 2 * i * 10 + i);
                float rand_val = distr(mt);
                if (rand_val > 0.5)
                {
                    x_dirs[i] = 1;
                }
                else
                {
                    x_dirs[i] = -1;
                }
            }

            // 分别按 8 个随机方向，每个方向算一遍
            for (int i = 0; i < 8; i++)
            {
                // 保存上次的计算结果
#pragma omp parallel for
                for (int id_z = 0; id_z < nz; id_z++)
                {
#pragma omp parallel for
                    for (int id_x = 0; id_x < nx; id_x++)
                    {
                        int idx = Pos2Idx(id_x, id_z, nx);
                        _temp_height[idx] = _height[idx];
                        _temp_debris[idx] = _debris[idx];
                    }
                }

                // 新的，确定的，随机方向，依据上次的计算结果进行计算
#pragma omp parallel for
                for (int id_z = 0; id_z < nz; id_z++)
                {
#pragma omp parallel for
                    for (int id_x = 0; id_x < nx; id_x++)
                    {
                        int iterseed = iter * 134775813;
                        int color = perm[i];

                        // randomized color order，6 种网格随机取半模式
                        int is_red = ((id_z & 1) == 1) && (color == 1);
                        int is_green = ((id_x & 1) == 1) && (color == 2);
                        int is_blue = ((id_z & 1) == 0) && (color == 3);
                        int is_yellow = ((id_x & 1) == 0) && (color == 4);
                        int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                        int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                        // randomized direction，其实只有 4 种模式
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

                            // 读取上次计算的结果
                            float i_debris = _temp_debris[idx];
                            float i_height = _temp_height[idx];

                            // 移除 邻格 被边界 clamp 的格子
                            int samplex = clamp(id_x + dx, 0, clamp_x);
                            int samplez = clamp(id_z + dz, 0, clamp_z);
                            int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);
                            // If we have closed borders, pretend a valid source to create
                            // a streak condition
                            if (validsource)
                            {
                                // 移除被标记为边界的格子
                                validsource = validsource || !openborder;

                                // 邻格 的索引号
                                int j_idx = Pos2Idx(samplex, samplez, nx);

                                // 邻格 的 height 和 debris
                                float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;
                                float j_height = _temp_height[j_idx];

                                // 邻格 跟 本格 比，高的是 中格，另一个是 邻格
                                int cidx = 0;   // 中格的 id_x
                                int cidz = 0;   // 中格的 id_y

                                float c_height = 0.0f;
                                float c_debris = 0.0f;
                                float n_debris = 0.0f;

                                int c_idx = 0;  // 中格的 idx
                                int n_idx = 0;  // 邻格的 idx

                                int dx_check = 0;   // 中格 指向 邻格 的方向
                                int dz_check = 0;

                                float h_diff = 0.0f;    // 高度差，>=0

                                if ((j_height - i_height) > 0.0f) // TODO: What to do when validsource is FALSE?
                                {
                                    // look at j's neighbours
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
                                    // look at i's neighbours
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
                                        // If we have closed borders, pretend a valid source to create
                                        // a streak condition
                                        // TODO: what is streak condition?
                                        tmp_validsource = tmp_validsource || !openborder;
                                        int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                        float n_height = _temp_height[tmp_j_idx];

                                        float tmp_diff = n_height - (c_height);

                                        float _gridbias = clamp(gridbias, -1.0f, 1.0f);

                                        if (tmp_dx && tmp_dz)
                                            tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                        else // !tmp_dx || !tmp_dz
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
                                    // Making sure all drops are moving
                                    dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                                    // Get the lower 32 bits and divide by max int
                                    unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                                    unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                                    cond = randval < cutoff;
                                }

                                if (cond)
                                {
                                    float abs_h_diff = h_diff < 0.0f ? -h_diff : h_diff;
                                    // Note: Used the neighbour mask value.
                                    float _cut_angle = clamp(cut_angle, 0.0f, 90.0f);
                                    float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f); // 用于计算斜率的底边长度
                                    float height_removed = _cut_angle < 90.0f ? tan(_cut_angle * M_PI / 180) * delta_x : 1e10f;
                                    float height_diff = abs_h_diff - height_removed;
                                    if (height_diff < 0.0f)
                                        height_diff = 0.0f;

                                    float prob = ((n_debris + c_debris) != 0.0f) ? clamp((height_diff / (n_debris + c_debris)), 0.0f, 1.0f) : 1.0f;
                                    // Get the lower 32 bits and divide by max int
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
                                            // return the excess debris as sediment to the higher elevation cell
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
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 将计算结果返回给外部数据，并删除临时属性
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                height[idx] = _height[idx];
                debris[idx] = _debris[idx];

                if (visualEnable)
                {
                    float coef = min(1, (debris[idx] / 1.0));
                    attr_color[idx] = (1 - coef) * attr_color[idx] + coef * vec3f(0.8, 0.6, 0.4);
                }
            }
        }

        terrain->verts.erase_attr("_height");
        terrain->verts.erase_attr("_debris");
        terrain->verts.erase_attr("_temp_height");
        terrain->verts.erase_attr("_temp_debris");

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_compute_erosion,
    { /* inputs: */ {
            "prim_2DGrid",
            {"string", "heightLayerName", "height"},
            {"string", "debrisLayerName", "debris"},

            {"int", "iterations", "10"},
            {"int", "openborder", "0"},
            {"float", "gridbias", "0"},
            {"float", "seed", "9676.79"},
            {"float", "cut_angle", "35"},
            {"float", "global_erosionrate", "1.0"},
            {"float", "erosionrate", "0.03"},
            {"float", "erodability", "0.4"},
            {"float", "removalrate", "0.7"},
            {"float", "maxdepth", "5.0"},

            {"int", "visualEnable", "0"},

        }, /* outputs: */ {
            "prim_2DGrid"
        }, /* params: */ {
        }, /* category: */ {
            "deprecated"
        } });
// 上面的（热侵蚀）节点 erode_compute_erosion 可以废弃了，由下面的 
// erode_rand_color
// erode_rand_dir
// erode_tumble_material_v0
// 节点代替
struct erode_rand_color : INode {
    void apply() override {

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter = get_input<NumericObject>("iter")->get<int>();

        int perm[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        for (int i = 0; i < 8; i++)
        {
            vec2f vec;
            std::mt19937 mt(iterations * iter * 8 * i + i);	// 梅森旋转算法
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

        auto& list = std::make_shared<zeno::ListObject>();
        for (int i = 0; i < 8; i++)
        {
            auto& num = std::make_shared<zeno::NumericObject>();
            num->set<int>(perm[i]);
            list->arr.push_back(num);
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(erode_rand_color,
    { /* inputs: */ {
        { "int", "iterations", "0" },
        { "int", "iter", "0" },
    }, /* outputs: */{
        "list",
    }, /* params: */{
    }, /* category: */{
        "erode",
    } });

struct erode_rand_dir : INode {
    void apply() override {

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter = get_input<NumericObject>("iter")->get<int>();

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

        auto& list = std::make_shared<zeno::ListObject>();
        for (int i = 0; i < 2; i++)
        {
            auto& num = std::make_shared<zeno::NumericObject>();
            num->set<int>(dirs[i]);
            list->arr.push_back(num);
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(erode_rand_dir,
    { /* inputs: */ {
        { "int", "iterations", "0" },
        { "int", "iter", "0" },
    }, /* outputs: */{
        "list",
    }, /* params: */{
    }, /* category: */{
        "erode",
    } });

struct erode_tumble_material_v0 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }

        ///////////////////////////////////////////////////////////////////////

        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();

        auto cut_angle = get_input<NumericObject>("cut_angle")->get<float>();

        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>();

        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>();

        auto erodability = get_input<NumericObject>("erodability")->get<float>();

        auto removalrate = get_input<NumericObject>("removalrate")->get<float>();

        auto maxdepth = get_input<NumericObject>("maxdepth")->get<float>();

        ///////////////////////////////////////////////////////////////////////

        std::uniform_real_distribution<float> distr(0.0, 1.0);                  // 设置随机分布
        auto seed = get_input<NumericObject>("seed")->get<float>();
        auto iterations = get_input<NumericObject>("iterations")->get<int>();   // 外部迭代总次数      10
        auto iter = get_input<NumericObject>("iter")->get<int>();               // 外部迭代当前次数    1~10
        auto i = get_input<NumericObject>("i")->get<int>();                     // 内部迭代当前次数    0~7
        auto openborder = get_input<NumericObject>("openborder")->get<int>();   // 获取边界标记

        auto& perm = get_input<ListObject>("perm")->get2<int>();
        auto& p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto& x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算用的临时属性，必须要有
        ////////////////////////////////////////////////////////////////////////////////////////
        if (!terrain->verts.has_attr("_height") ||
            !terrain->verts.has_attr("_debris") ||
            !terrain->verts.has_attr("_temp_height") ||
            !terrain->verts.has_attr("_temp_debris"))
        {
            // height 和 debris 数据要从外面读取，所以属性中要有 height 和 debris
            zeno::log_error("Node [erode_tumble_material_v0], no such data layer named '{}' or '{}' or '{}' or '{}'.",
                "_height", "_debris", "_temp_height", "_temp_debris");
        }
        auto& _height = terrain->verts.attr<float>("_height");              // 计算用的临时属性
        auto& _debris = terrain->verts.attr<float>("_debris");
        auto& _temp_height = terrain->verts.attr<float>("_temp_height");    // 备份用的临时属性
        auto& _temp_debris = terrain->verts.attr<float>("_temp_debris");

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////
        // 新的，确定的，随机方向，依据上次的计算结果进行计算
#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];

                // randomized color order，6 种网格随机取半模式
                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                // randomized direction，其实只有 4 种模式
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

                    // 读取上次计算的结果
                    float i_debris = _temp_debris[idx];
                    float i_height = _temp_height[idx];

                    // 移除 邻格 被边界 clamp 的格子
                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);
                    // If we have closed borders, pretend a valid source to create
                    // a streak condition
                    if (validsource)
                    {
                        // 移除被标记为边界的格子
                        validsource = validsource || !openborder;

                        // 邻格 的索引号
                        int j_idx = Pos2Idx(samplex, samplez, nx);

                        // 邻格 的 height 和 debris
                        float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;
                        float j_height = _temp_height[j_idx];

                        // 邻格 跟 本格 比，高的是 中格，另一个是 邻格
                        int cidx = 0;   // 中格的 id_x
                        int cidz = 0;   // 中格的 id_y

                        float c_height = 0.0f;
                        float c_debris = 0.0f;
                        float n_debris = 0.0f;

                        int c_idx = 0;  // 中格的 idx
                        int n_idx = 0;  // 邻格的 idx

                        int dx_check = 0;   // 中格 指向 邻格 的方向
                        int dz_check = 0;

                        float h_diff = 0.0f;    // 高度差，>=0

                        if ((j_height - i_height) > 0.0f) // TODO: What to do when validsource is FALSE?
                        {
                            // look at j's neighbours
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
                            // look at i's neighbours
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
                                // If we have closed borders, pretend a valid source to create
                                // a streak condition
                                // TODO: what is streak condition?
                                tmp_validsource = tmp_validsource || !openborder;
                                int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                float n_height = _temp_height[tmp_j_idx];

                                float tmp_diff = n_height - (c_height);

                                float _gridbias = clamp(gridbias, -1.0f, 1.0f);

                                if (tmp_dx && tmp_dz)
                                    tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                else // !tmp_dx || !tmp_dz
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
                            // Making sure all drops are moving
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            // Get the lower 32 bits and divide by max int
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        if (cond)
                        {
                            float abs_h_diff = h_diff < 0.0f ? -h_diff : h_diff;
                            // Note: Used the neighbour mask value.
                            float _cut_angle = clamp(cut_angle, 0.0f, 90.0f);
                            float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f); // 用于计算斜率的底边长度
                            float height_removed = _cut_angle < 90.0f ? tan(_cut_angle * M_PI / 180) * delta_x : 1e10f;
                            float height_diff = abs_h_diff - height_removed;
                            if (height_diff < 0.0f)
                                height_diff = 0.0f;
                            /*
                            float rock_adj = height_diff / (abs_h_diff + 0.000001);
                            rock_adj = clamp(rock_adj, 0.0f, 1.0f);
                            if (rock_adj > 0.2)
                            {
                                rock_adj = (rock_adj - 0.25) * 4;
                                rock_adj = clamp(rock_adj, 0.0f, 1.0f);
                            }
                            else
                            {
                                rock_adj = 0.0f;
                            }
                            */
                            float prob = ((n_debris + c_debris) != 0.0f) ? clamp((height_diff / (n_debris + c_debris)), 0.0f, 1.0f) : 1.0f;
                            // Get the lower 32 bits and divide by max int
                            unsigned int cutoff = (unsigned int)(prob * 4294967295.0);
                            unsigned int randval = erode_random(seed * 3.14, (idx + nx * nz) * 8 + color + iterseed);
                            int do_erode = randval < cutoff;

                            float height_removal_amt = do_erode * clamp(global_erosionrate * erosionrate * erodability, 0.0f, height_diff);
                            //float height_removal_amt = do_erode * clamp(global_erosionrate * erosionrate * erodability, 0.0f, height_diff) * (1 - rock_adj);

                            _height[c_idx] -= height_removal_amt;

                            float bedrock_density = 1.0f - (removalrate);
                            if (bedrock_density > 0.0f)
                            {
                                float newdebris = bedrock_density * height_removal_amt;
                                if (n_debris + newdebris > maxdepth)
                                {
                                    float rollback = n_debris + newdebris - maxdepth;
                                    rollback = min(rollback, newdebris);
                                    // return the excess debris as sediment to the higher elevation cell
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
    { /* inputs: */ {
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
            {"int", "visualEnable", "0"},


            {"float", "cut_angle", "35"},
            {"float", "global_erosionrate", "1.0"},
            {"float", "erosionrate", "0.03"},
            {"float", "erodability", "0.4"},
            {"float", "removalrate", "0.7"},
            {"float", "maxdepth", "5.0"},

        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
        }, /* category: */ {
            "erode",
        } });
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// 此节点描绘（崩塌）过程
struct erode_slump_b2 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化数据层
        ////////////////////////////////////////////////////////////////////////////////////////

        // height 和 debris 只能从外部获取，不能从内部创建，因为本节点要被嵌入循环中
        // 初始化 height 和 debris 的过程 应该 在此节点的外部
        auto heightLayerName = get_input<StringObject>("heightLayerName")->get();
        auto materialLayerName = get_input<StringObject>("materialLayerName")->get();
        auto stabilitymaskLayerName = get_input<StringObject>("stabilitymaskLayerName")->get();
        if (!terrain->verts.has_attr(heightLayerName) ||
            !terrain->verts.has_attr(materialLayerName) ||
            !terrain->verts.has_attr(stabilitymaskLayerName))
        {
            // height 和 debris 数据要从外面读取，所以属性中要有 height 和 debris
            zeno::log_error("no such data layer named '{}' or '{}' or '{}'.", heightLayerName, materialLayerName, stabilitymaskLayerName);
        }
        auto& height = terrain->verts.attr<float>(heightLayerName);   // 读取外部数据
        auto& debris = terrain->verts.attr<float>(materialLayerName);
        auto& stabilitymask = terrain->verts.attr<float>(stabilitymaskLayerName);

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 创建临时属性，将外部数据拷贝到临时属性，我们将使用临时属性进行计算
        ////////////////////////////////////////////////////////////////////////////////////////

        auto& _material = terrain->verts.add_attr<float>("_material");         // 计算用的临时属性
        auto& _temp_material = terrain->verts.add_attr<float>("_temp_material");    // 备份用的临时属性

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                _material[idx] = debris[idx];     // 外部数据拷贝到临时属性
                _temp_material[idx] = 0;          // 正式计算前会把 _debris 的数据保存在这里
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 获取计算所需参数
        ////////////////////////////////////////////////////////////////////////////////////////

        std::uniform_real_distribution<float> distr(0.0, 1.0);                  // 设置随机分布

        auto seed = get_input<NumericObject>("seed")->get<float>();

        auto iterations = get_input<NumericObject>("iterations")->get<int>();   // 获取迭代次数

        auto openborder = get_input<NumericObject>("openborder")->get<int>();   // 获取边界标记

        auto repose_angle = get_input<NumericObject>("repose_angle")->get<float>();

        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();

        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>();

        auto flow_rate = get_input<NumericObject>("flow_rate")->get<float>();


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

        for (int iter = 1; iter <= iterations; iter++)
        {
            // 准备随机数组，每次迭代都都会有变化，用于网格随机取半，以及产生随机方向
            int perm[] = { 1, 2, 3, 4, 5, 6, 7, 8 };

            for (int i = 0; i < 8; i++)
            {
                vec2f vec;
                //                std::mt19937 mt(i * iterations * iter + iter);	// 梅森旋转算法
                std::mt19937 mt(iterations * iter * 8 * i + i);	// 梅森旋转算法
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

            int p_dirs[] = { -1, -1 };
            for (int i = 0; i < 2; i++)
            {
                //                std::mt19937 mt(i * iterations * iter * 20 + iter);
                std::mt19937 mt(iterations * iter * 2 * i + i);
                float rand_val = distr(mt);
                if (rand_val > 0.5)
                {
                    p_dirs[i] = 1;
                }
                else
                {
                    p_dirs[i] = -1;
                }
            }

            int x_dirs[] = { -1, -1 };
            for (int i = 0; i < 2; i++)
            {
                //                std::mt19937 mt(i * iterations * iter * 30 + iter);
                std::mt19937 mt(iterations * iter * 2 * i * 10 + i);
                float rand_val = distr(mt);
                if (rand_val > 0.5)
                {
                    x_dirs[i] = 1;
                }
                else
                {
                    x_dirs[i] = -1;
                }
            }

            // 分别按 8 个随机方向，每个方向算一遍
            for (int i = 0; i < 8; i++)
            {
                // 保存上次的计算结果
#pragma omp parallel for
                for (int id_z = 0; id_z < nz; id_z++)
                {
#pragma omp parallel for
                    for (int id_x = 0; id_x < nx; id_x++)
                    {
                        int idx = Pos2Idx(id_x, id_z, nx);
                        _temp_material[idx] = _material[idx];
                    }
                }

                // 新的，确定的，随机方向，依据上次的计算结果进行计算
#pragma omp parallel for
                for (int id_z = 0; id_z < nz; id_z++)
                {
#pragma omp parallel for
                    for (int id_x = 0; id_x < nx; id_x++)
                    {
                        int iterseed = iter * 134775813;
                        int color = perm[i];

                        // randomized color order，6 种网格随机取半模式
                        int is_red = ((id_z & 1) == 1) && (color == 1);
                        int is_green = ((id_x & 1) == 1) && (color == 2);
                        int is_blue = ((id_z & 1) == 0) && (color == 3);
                        int is_yellow = ((id_x & 1) == 0) && (color == 4);
                        int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                        int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                        // randomized direction，其实只有 4 种模式
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

                            //if (idx == 249494)
                            //{
                                //printf(" 1-----> flow_rate = %f\n", flow_rate);
                            //}
                            //float flow_rate = clamp(flow_rate, 0.0f, 1.0f); // 严重错误 哈哈哈
                            flow_rate = clamp(flow_rate, 0.0f, 1.0f);
                            //if (idx == 249494)
                            //{
                                //printf(" 2-----> flow_rate = %f\n", flow_rate);
                            //}

                            // 读取上次计算的结果
                            float i_material = _temp_material[idx];	// 数据来自上次的计算结果 **************************
                            float i_height = height[idx];			// height 数据只读

                            // 移除 邻格 被边界 clamp 的格子
                            int samplex = clamp(id_x + dx, 0, clamp_x);
                            int samplez = clamp(id_z + dz, 0, clamp_z);
                            int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                            // If we have closed borders, pretend a valid source to create
                            // a streak condition
                            if (validsource)
                            {
                                int same_node = !validsource;	// 恒等于 0 ？？？ 干嘛？ 备份，因为后面 validsource 被修改了 ？？？

                                // 移除被标记为边界的格子
                                validsource = validsource || !openborder;

                                // 邻格 的索引号
                                int j_idx = Pos2Idx(samplex, samplez, nx);

                                // 邻格 的 height 和 debris
                                float j_material = validsource ? _temp_material[j_idx] : 0.0f;
                                float j_height = height[j_idx];


                                // The maximum slope at which we stop slumping
                                float _repose_angle = repose_angle;
                                _repose_angle = clamp(_repose_angle, 0.0f, 90.0f);
                                float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f); // 用于计算斜率的底边长度

                                // repose_angle 对应的高度差，停止崩塌的高度差
                                float static_diff = _repose_angle < 90.0f ? tan(_repose_angle * M_PI / 180.0) * delta_x : 1e10f;

                                // 包含 height 和 debris 的高度差，注意这里是 邻格 - 本格
                                float m_diff = (j_height + j_material) - (i_height + i_material);

                                // 邻格 跟 本格 比，高的是 中格，另一个是 邻格
                                int cidx = 0;   // 中格的 id_x
                                int cidz = 0;   // 中格的 id_y

                                float c_height = 0.0f;      // 中格
                                float c_material = 0.0f;    // 中格
                                float n_material = 0.0f;    // 邻格

                                int c_idx = 0;  // 中格的 idx
                                int n_idx = 0;  // 邻格的 idx

                                int dx_check = 0;   // 中格 指向 邻格 的方向
                                int dz_check = 0;

                                // 如果邻格比本格高，邻格->中格，本格->邻格
                                // 高的是 中格
                                if (m_diff > 0.0f)
                                {
                                    // look at j's neighbours
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
                                    // look at i's neighbours
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
                                            // If we have closed borders, pretend a valid source to create
                                            // a streak condition
                                            // TODO: what is streak condition?
                                            tmp_validsource = tmp_validsource || !openborder;
                                            int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                            // 中格周围的邻格 碎屑 的高度
                                            float n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;

                                            // 中格周围邻格 地面 高度
                                            float n_height = height[tmp_j_idx];

                                            // 中格周围的邻格 地面 高度 - 中格 地面 高度
                                            float tmp_h_diff = n_height - (c_height);

                                            // 中格周围的邻格 高度 - 中格 高度
                                            float tmp_m_diff = (n_height + n_material) - (c_height + c_material);

                                            // 地面高度差 : 总高度差
                                            float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;

                                            float _gridbias = gridbias;

                                            _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                            // 修正高度差
                                            if (tmp_dx && tmp_dz)
                                                tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                            else // !tmp_dx || !tmp_dz
                                                tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                            // diff_idx = 1 的时候，前面比较过格子的总高度差，此时
                                            // 如果周边格子 不比我高，因为前面有过交换，所以至少有一个格子满足这个要求
                                            // diff_idx = 0 的时候，下面的条件不一定能满足。格子的地面有可能是最低的
                                            if (tmp_diff <= 0.0f)	// 只统计比我低的邻格，所以 高度差 的说法改为 深度差
                                            {
                                                // 指定方向上，中格(我) 与 邻格 的深度差
                                                // dir_probs[0] 可能此时 >0 不会进来，此时 dir_probs[0] 保持默认值 0
                                                if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                                    dir_probs[diff_idx] = tmp_diff;

                                                // 按格子总高度计算的时候，记录 tmp_diff 最深的深度，作为 dir_prob
                                                if (diff_idx && dir_prob > tmp_diff)
                                                {
                                                    dir_prob = tmp_diff;
                                                }

                                                // 记录比 中格 低的邻格的深度和
                                                sum_diffs[diff_idx] += tmp_diff;
                                            }
                                        }
                                    }

                                    if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                                    {
                                        // 按 (地面高度差+碎屑高度差)来计算时，流动概率 = 指定方向上的深度差 / 最大深度差
                                        dir_prob = dir_probs[diff_idx] / dir_prob;
                                    }

                                    // 另一种计算方法：指定方向上的流动概率 = 指定方向上的深度差 / 所有比我低的邻格的深度差之和
                                    // 这种概率显然比上一种方法的计算结果要 低
                                    // diff_idx == 1 时，深度差 以 (地面高度差+碎屑高度差) 来计算时 
                                    // diff_idx == 0 时，深度差 以 (地面高度差) 来计算时，可能不存在，不过已经取默认值为 0 了
                                    if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                                        dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                                }

                                // 最多可供流失的高度差
                                float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;

                                float stability_val = 0.0f;

                                /// 这里要非常注意 ！！！！！！！！！！！！！
                                /// 大串联的时候，这里是要有输入的 ！！！！！！！！！！！！！！！！！
                                stability_val = clamp(stabilitymask[c_idx], 0.0f, 1.0f);

                                if (stability_val > 0.01f)
                                {
                                    //if (idx == 249494)
                                    //{
                                        //printf("-=WB1=- iter = %i, i = %i, movable_mat = %f, stability_val = %f, c_material = %f\n",
                                        //    iter, i, movable_mat, stability_val, c_material);
                                    //}
                                    // movement is slowed down according to the stability mask and not the repose angle
                                    // 只要有一点点遮罩，流失量至少减半，不过默认没有遮罩
                                    movable_mat = clamp(movable_mat * (1.0f - stability_val) * 0.5f, 0.0f, c_material);
                                }
                                else
                                {
                                    //if (idx == 249494)
                                    //{
                                        //printf("-=WB2=- iter = %i, i = %i, movable_mat = %f, static_diff = %f, c_material = %f\n",
                                        //    iter, i, movable_mat, static_diff, c_material);
                                    //}
                                    // 流失量根据 static_diff 修正，static_diff 是 repose angle 对应的高度差
                                    // 问题是，repose_angle 默认为 0，但可流失量仍然减半了。。。
                                    movable_mat = clamp((movable_mat - static_diff) * 0.5f, 0.0f, c_material);
                                }

                                // 以 height + debris 来计算
                                float l_rat = dir_probs[1];
                                // TODO: What is a good limit here?
                                // 让水流继续保持足够的水量
                                if (quant_amt > 0.001)	// 默认 = 1.0
                                    movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                                else
                                    movable_mat *= l_rat; // 乘上概率，这样随着水量快速减少，水流很快就消失了

                                float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                                //if (idx == 249494)
                                //{
                                    //printf("diff = %f, m_diff = %f, movable_mat = %f\n", diff, m_diff, movable_mat);
                                //}

                                int cond = 0;
                                if (dir_prob >= 1.0f)
                                    cond = 1;
                                else
                                {
                                    // Making sure all drops are moving
                                    dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                                    unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                                    unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                                    cond = randval < cutoff;
                                }

                                // 不参与计算的格子，或者没有流动概率的格子
                                if (!cond || same_node)
                                    diff = 0.0f;

                                //if (idx == 249494)
                                //{
                                    //printf("flow_rate = %f, diff = %f, movable_mat = %f\n", flow_rate, diff, movable_mat);
                                //}

                                // TODO: Check if this one should be here or before quantization
                                diff *= flow_rate;	// 1.0

                                float abs_diff = (diff < 0.0f) ? -diff : diff;

                                //if (idx == 249494)
                                //{
                                    //printf(" flow_rate = %f, diff = %f, abs_diff = %f\n", flow_rate, diff, abs_diff);
                                //}

                                // Update the material level
                                // 中格失去碎屑
                                _material[c_idx] = c_material - abs_diff;
                                // 邻格得到碎屑
                                _material[n_idx] = n_material + abs_diff;

                            }

                        }
                    }
                }
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 将计算结果返回给外部数据，并删除临时属性
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                debris[idx] = _material[idx];     // 计算结果返回给外部数据

                if (visualEnable)
                {
                    float coef = min(1, (debris[idx] / 1.0));
                    attr_color[idx] = (1 - coef) * attr_color[idx] + coef * vec3f(0.8, 0.6, 0.4);
                }
            }
        }

        terrain->verts.erase_attr("_material");
        terrain->verts.erase_attr("_temp_material");

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_slump_b2,
    { /* inputs: */ {
            "prim_2DGrid",
            {"string", "heightLayerName", "height"},
            {"string", "materialLayerName", "debris"},
            {"string", "stabilitymaskLayerName", "_stability"},

            {"int", "iterations", "10"},
            {"int", "openborder", "0"},
            {"float", "gridbias", "0.0"},
            {"float", "seed", "15231.3"},
            {"float", "repose_angle", "15.0"},
            {"float", "quant_amt", "0.25"},
            {"float", "flow_rate", "1.0"},

            {"int", "visualEnable", "0"},

        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
        }, /* category: */ {
            "deprecated",
        } });
// 上面的（崩塌）节点 erode_slump_b2 可以废弃了，由下面的 
// erode_tumble_material_v2
// 节点代替
struct erode_tumble_material_v2 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }

        ///////////////////////////////////////////////////////////////////////

        auto gridbias = get_input<NumericObject>("gridbias")->get<float>();

        auto repose_angle = get_input<NumericObject>("repose_angle")->get<float>();

        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>();

        auto flow_rate = get_input<NumericObject>("flow_rate")->get<float>();

        ///////////////////////////////////////////////////////////////////////

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        auto seed = get_input<NumericObject>("seed")->get<float>();
        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter = get_input<NumericObject>("iter")->get<int>();
        auto i = get_input<NumericObject>("i")->get<int>();
        auto openborder = get_input<NumericObject>("openborder")->get<int>();

        auto& perm = get_input<ListObject>("perm")->get2<int>();
        auto& p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto& x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算用的临时属性，必须要有
        ////////////////////////////////////////////////////////////////////////////////////////
        if (!terrain->verts.has_attr("height") ||
            !terrain->verts.has_attr("_stability") ||
            !terrain->verts.has_attr("_material") ||
            !terrain->verts.has_attr("_temp_material"))
        {
            // height 和 debris 数据要从外面读取，所以属性中要有 height 和 debris
            zeno::log_error("Node [erode_tumble_material_v2], no such data layer named '{}' or '{}' or '{}' or '{}'.",
                "height", "_stability", "_material", "_temp_material");
        }
        auto& height = terrain->verts.add_attr<float>("height");
        auto& stabilitymask = terrain->verts.add_attr<float>("_stability");
        auto& _material = terrain->verts.add_attr<float>("_material");
        auto& _temp_material = terrain->verts.add_attr<float>("_temp_material");

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////
        // 新的，确定的，随机方向，依据上次的计算结果进行计算
#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];

                // randomized color order，6 种网格随机取半模式
                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                // randomized direction，其实只有 4 种模式
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

                    //if (idx == 249494)
                    //{
                        //printf(" 1-----> flow_rate = %f\n", flow_rate);
                    //}
                    //float flow_rate = clamp(flow_rate, 0.0f, 1.0f); // 严重错误 哈哈哈
                    flow_rate = clamp(flow_rate, 0.0f, 1.0f);
                    //if (idx == 249494)
                    //{
                        //printf(" 2-----> flow_rate = %f\n", flow_rate);
                    //}

                    // 读取上次计算的结果
                    float i_material = _temp_material[idx];	// 数据来自上次的计算结果 **************************
                    float i_height = height[idx];			// height 数据只读

                    // 移除 邻格 被边界 clamp 的格子
                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                    // If we have closed borders, pretend a valid source to create
                    // a streak condition
                    if (validsource)
                    {
                        int same_node = !validsource;	// 恒等于 0 ？？？ 干嘛？ 备份，因为后面 validsource 被修改了 ？？？

                        // 移除被标记为边界的格子
                        validsource = validsource || !openborder;

                        // 邻格 的索引号
                        int j_idx = Pos2Idx(samplex, samplez, nx);

                        // 邻格 的 height 和 debris
                        float j_material = validsource ? _temp_material[j_idx] : 0.0f;
                        float j_height = height[j_idx];


                        // The maximum slope at which we stop slumping
                        float _repose_angle = repose_angle;
                        _repose_angle = clamp(_repose_angle, 0.0f, 90.0f);
                        float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f); // 用于计算斜率的底边长度

                        // repose_angle 对应的高度差，停止崩塌的高度差
                        float static_diff = _repose_angle < 90.0f ? tan(_repose_angle * M_PI / 180.0) * delta_x : 1e10f;

                        // 包含 height 和 debris 的高度差，注意这里是 邻格 - 本格
                        float m_diff = (j_height + j_material) - (i_height + i_material);

                        // 邻格 跟 本格 比，高的是 中格，另一个是 邻格
                        int cidx = 0;   // 中格的 id_x
                        int cidz = 0;   // 中格的 id_y

                        float c_height = 0.0f;      // 中格
                        float c_material = 0.0f;    // 中格
                        float n_material = 0.0f;    // 邻格

                        int c_idx = 0;  // 中格的 idx
                        int n_idx = 0;  // 邻格的 idx

                        int dx_check = 0;   // 中格 指向 邻格 的方向
                        int dz_check = 0;

                        // 如果邻格比本格高，邻格->中格，本格->邻格
                        // 高的是 中格
                        if (m_diff > 0.0f)
                        {
                            // look at j's neighbours
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
                            // look at i's neighbours
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
                                    // If we have closed borders, pretend a valid source to create
                                    // a streak condition
                                    // TODO: what is streak condition?
                                    tmp_validsource = tmp_validsource || !openborder;
                                    int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                    // 中格周围的邻格 碎屑 的高度
                                    float n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;

                                    // 中格周围邻格 地面 高度
                                    float n_height = height[tmp_j_idx];

                                    // 中格周围的邻格 地面 高度 - 中格 地面 高度
                                    float tmp_h_diff = n_height - (c_height);

                                    // 中格周围的邻格 高度 - 中格 高度
                                    float tmp_m_diff = (n_height + n_material) - (c_height + c_material);

                                    // 地面高度差 : 总高度差
                                    float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;

                                    float _gridbias = gridbias;

                                    _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                    // 修正高度差
                                    if (tmp_dx && tmp_dz)
                                        tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                    else // !tmp_dx || !tmp_dz
                                        tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                    // diff_idx = 1 的时候，前面比较过格子的总高度差，此时
                                    // 如果周边格子 不比我高，因为前面有过交换，所以至少有一个格子满足这个要求
                                    // diff_idx = 0 的时候，下面的条件不一定能满足。格子的地面有可能是最低的
                                    if (tmp_diff <= 0.0f)	// 只统计比我低的邻格，所以 高度差 的说法改为 深度差
                                    {
                                        // 指定方向上，中格(我) 与 邻格 的深度差
                                        // dir_probs[0] 可能此时 >0 不会进来，此时 dir_probs[0] 保持默认值 0
                                        if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                            dir_probs[diff_idx] = tmp_diff;

                                        // 按格子总高度计算的时候，记录 tmp_diff 最深的深度，作为 dir_prob
                                        if (diff_idx && dir_prob > tmp_diff)
                                        {
                                            dir_prob = tmp_diff;
                                        }

                                        // 记录比 中格 低的邻格的深度和
                                        sum_diffs[diff_idx] += tmp_diff;
                                    }
                                }
                            }

                            if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                            {
                                // 按 (地面高度差+碎屑高度差)来计算时，流动概率 = 指定方向上的深度差 / 最大深度差
                                dir_prob = dir_probs[diff_idx] / dir_prob;
                            }

                            // 另一种计算方法：指定方向上的流动概率 = 指定方向上的深度差 / 所有比我低的邻格的深度差之和
                            // 这种概率显然比上一种方法的计算结果要 低
                            // diff_idx == 1 时，深度差 以 (地面高度差+碎屑高度差) 来计算时 
                            // diff_idx == 0 时，深度差 以 (地面高度差) 来计算时，可能不存在，不过已经取默认值为 0 了
                            if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                                dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                        }

                        // 最多可供流失的高度差
                        float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;

                        float stability_val = 0.0f;

                        /// 这里要非常注意 ！！！！！！！！！！！！！
                        /// 大串联的时候，这里是要有输入的 ！！！！！！！！！！！！！！！！！
                        stability_val = clamp(stabilitymask[c_idx], 0.0f, 1.0f);

                        if (stability_val > 0.01f)
                        {
                            //if (idx == 249494)
                            //{
                                //printf("-=WB1=- iter = %i, i = %i, movable_mat = %f, stability_val = %f, c_material = %f\n",
                                //    iter, i, movable_mat, stability_val, c_material);
                            //}
                            // movement is slowed down according to the stability mask and not the repose angle
                            // 只要有一点点遮罩，流失量至少减半，不过默认没有遮罩
                            movable_mat = clamp(movable_mat * (1.0f - stability_val) * 0.5f, 0.0f, c_material);
                        }
                        else
                        {
                            //if (idx == 249494)
                            //{
                                //printf("-=WB2=- iter = %i, i = %i, movable_mat = %f, static_diff = %f, c_material = %f\n",
                                //    iter, i, movable_mat, static_diff, c_material);
                            //}
                            // 流失量根据 static_diff 修正，static_diff 是 repose angle 对应的高度差
                            // 问题是，repose_angle 默认为 0，但可流失量仍然减半了。。。
                            movable_mat = clamp((movable_mat - static_diff) * 0.5f, 0.0f, c_material);
                        }

                        // 以 height + debris 来计算
                        float l_rat = dir_probs[1];
                        // TODO: What is a good limit here?
                        // 让水流继续保持足够的水量
                        if (quant_amt > 0.001)	// 默认 = 1.0
                            movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                        else
                            movable_mat *= l_rat; // 乘上概率，这样随着水量快速减少，水流很快就消失了

                        float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                        //if (idx == 249494)
                        //{
                            //printf("diff = %f, m_diff = %f, movable_mat = %f\n", diff, m_diff, movable_mat);
                        //}

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else
                        {
                            // Making sure all drops are moving
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        // 不参与计算的格子，或者没有流动概率的格子
                        if (!cond || same_node)
                            diff = 0.0f;

                        //if (idx == 249494)
                        //{
                            //printf("flow_rate = %f, diff = %f, movable_mat = %f\n", flow_rate, diff, movable_mat);
                        //}

                        // TODO: Check if this one should be here or before quantization
                        diff *= flow_rate;	// 1.0

                        float abs_diff = (diff < 0.0f) ? -diff : diff;

                        //if (idx == 249494)
                        //{
                            //printf(" flow_rate = %f, diff = %f, abs_diff = %f\n", flow_rate, diff, abs_diff);
                        //}

                        // Update the material level
                        // 中格失去碎屑
                        _material[c_idx] = c_material - abs_diff;
                        // 邻格得到碎屑
                        _material[n_idx] = n_material + abs_diff;

                    }

                }
            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v2,
    { /* inputs: */ {
            "prim_2DGrid",

            {"ListObject", "perm"},
            {"ListObject", "p_dirs"},
            {"ListObject", "x_dirs"},

            {"float", "seed", "15231.3"},
            {"int", "iterations", "0"},
            {"int", "iter", "0"},
            {"int", "i", "0"},

            {"int", "openborder", "0"},
            {"float", "gridbias", "0.0"},
            {"int", "visualEnable", "0"},

            // 崩塌流淌相关
            {"float", "repose_angle", "15.0"},
            {"float", "quant_amt", "0.25"},
            {"float", "flow_rate", "1.0"},

        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
        }, /* category: */ {
            "erode",
        } });
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// 此节点描绘（崩塌+侵蚀）过程
struct erode_slump_b4 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化数据层
        ////////////////////////////////////////////////////////////////////////////////////////

        // height 和 debris 只能从外部获取，不能从内部创建，因为本节点要被嵌入循环中
        // 初始化 height 和 debris 的过程 应该 在此节点的外部
        auto heightLayerName = get_input<StringObject>("heightLayerName")->get();     // height
        auto materialLayerName = get_input<StringObject>("materialLayerName")->get();   // water
        auto sedimentLayerName = get_input<StringObject>("sedimentLayerName")->get();   // sediment
        auto debrisLayerName = get_input<StringObject>("debrisLayerName")->get();     // debris
        if (!terrain->verts.has_attr(heightLayerName) ||
            !terrain->verts.has_attr(materialLayerName) ||
            !terrain->verts.has_attr(sedimentLayerName) ||
            !terrain->verts.has_attr(debrisLayerName))
        {
            // 需要从外部读取的数据
            zeno::log_error("no such data layer named '{}' or '{}' or '{}' or '{}'.",
                heightLayerName, materialLayerName, sedimentLayerName, debrisLayerName);
        }
        auto& height = terrain->verts.attr<float>(heightLayerName);     // 读取外部数据
        auto& material = terrain->verts.attr<float>(materialLayerName);
        auto& sediment = terrain->verts.attr<float>(sedimentLayerName);
        auto& debris = terrain->verts.attr<float>(debrisLayerName);


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 创建临时属性，将外部数据拷贝到临时属性，我们将使用临时属性进行计算
        ////////////////////////////////////////////////////////////////////////////////////////
        auto& _height = terrain->verts.add_attr<float>("_height");
        auto& _temp_height = terrain->verts.add_attr<float>("_temp_height");

        auto& _material = terrain->verts.add_attr<float>("_material");
        auto& _temp_material = terrain->verts.add_attr<float>("_temp_material");

        auto& _debris = terrain->verts.add_attr<float>("_debris");
        auto& _temp_debris = terrain->verts.add_attr<float>("_temp_debris");

        auto& _sediment = terrain->verts.add_attr<float>("_sediment");

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);

                _height[idx] = height[idx];	// 用于获取外部数据
                _temp_height[idx] = 0;      // 用于存放数据备份

                _material[idx] = material[idx];
                _temp_material[idx] = 0;

                _debris[idx] = debris[idx];
                _temp_debris[idx] = 0;

                _sediment[idx] = sediment[idx];
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 获取计算所需参数
        ////////////////////////////////////////////////////////////////////////////////////////

        std::uniform_real_distribution<float> distr(0.0, 1.0);                  // 设置随机分布

        auto openborder = get_input<NumericObject>("openborder")->get<int>();   // 获取边界标记

        // 侵蚀主参数
        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>(); // 1 全局侵蚀率
        auto erodability = get_input<NumericObject>("erodability")->get<float>(); // 1.0 侵蚀能力
        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>(); // 0.4 侵蚀率
        auto bank_angle = get_input<NumericObject>("bank_angle")->get<float>(); // 70.0 河堤侵蚀角度
        auto seed = get_input<NumericObject>("seed")->get<float>(); // 12.34

        // 高级参数
        auto removalrate = get_input<NumericObject>("removalrate")->get<float>(); // 0.0 风化率/水吸收率
        auto max_debris_depth = get_input<NumericObject>("max_debris_depth")->get<float>();// 5	碎屑最大深度
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>(); // 0.0

        // 侵蚀能力调整
        auto max_erodability_iteration = get_input<NumericObject>("max_erodability_iteration")->get<int>(); // 5
        auto initial_erodability_factor = get_input<NumericObject>("initial_erodability_factor")->get<float>(); // 0.5
        auto slope_contribution_factor = get_input<NumericObject>("slope_contribution_factor")->get<float>(); // 0.8

        // 河床参数
        auto bed_erosionrate_factor = get_input<NumericObject>("bed_erosionrate_factor")->get<float>(); // 1 河床侵蚀率因子
        auto depositionrate = get_input<NumericObject>("depositionrate")->get<float>(); // 0.01 沉积率
        auto sedimentcap = get_input<NumericObject>("sedimentcap")->get<float>(); // 10.0 高度差转变为沉积物的比率 / 泥沙容量，每单位流动水可携带的泥沙量

        // 河堤参数
        auto bank_erosionrate_factor = get_input<NumericObject>("bank_erosionrate_factor")->get<float>(); // 1.0 河堤侵蚀率因子
        auto max_bank_bed_ratio = get_input<NumericObject>("max_bank_bed_ratio")->get<float>(); // 0.5 The maximum of bank to bed water column height ratio
                                                                                                // 高于这个比值的河岸将不会在侵蚀中被视为河岸，会停止侵蚀
        // 河流控制
        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>(); // 0.05 流量维持率，越高流量越稳定
        auto iterations = get_input<NumericObject>("iterations")->get<int>(); // 流淌的总迭代次数


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

        // 测量计算时间 开始
        clock_t start, finish;
        printf("~~~~~~~~~~~~~~~~~~~~\n");
        printf("start ... ...\n");
        start = clock();

        //#pragma omp parallel for // 会出错
        for (int iter = 1; iter <= iterations; iter++)
        {
            // 准备随机数组，每次迭代都都会有变化，用于网格随机取半，以及产生随机方向
            int perm[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
            //#pragma omp parallel for // 会导致每次结果不一样
            for (int i = 0; i < 8; i++)
            {
                vec2f vec;
                //                std::mt19937 mt(i * iterations * iter + iter);	// 梅森旋转算法
                std::mt19937 mt(iterations * iter * 8 * i + i);	// 梅森旋转算法
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

            int p_dirs[] = { -1, -1 };
            for (int i = 0; i < 2; i++)
            {
                //                std::mt19937 mt(i * iterations * iter * 20 + iter);
                std::mt19937 mt(iterations * iter * 2 * i + i);
                float rand_val = distr(mt);
                if (rand_val > 0.5)
                {
                    p_dirs[i] = 1;
                }
                else
                {
                    p_dirs[i] = -1;
                }
            }

            int x_dirs[] = { -1, -1 };
            for (int i = 0; i < 2; i++)
            {
                //                std::mt19937 mt(i * iterations * iter * 30 + iter);
                std::mt19937 mt(iterations * iter * 2 * i * 10 + i);
                float rand_val = distr(mt);
                if (rand_val > 0.5)
                {
                    x_dirs[i] = 1;
                }
                else
                {
                    x_dirs[i] = -1;
                }
            }

            // 分别按 8 个随机方向，每个方向算一遍，其实只有 4 个方向模式
//#pragma omp parallel for // 会出错
            for (int i = 0; i < 8; i++)
            {
                // 保存上次的计算结果
#pragma omp parallel for
                for (int id_z = 0; id_z < nz; id_z++)
                {
#pragma omp parallel for
                    for (int id_x = 0; id_x < nx; id_x++)
                    {
                        int idx = Pos2Idx(id_x, id_z, nx);
                        _temp_height[idx] = _height[idx];
                        _temp_material[idx] = _material[idx];
                        _temp_debris[idx] = _debris[idx];
                    }
                }

                // 新的，确定的，随机方向，依据上次的计算结果进行计算
#pragma omp parallel for
                for (int id_z = 0; id_z < nz; id_z++)
                {
#pragma omp parallel for
                    for (int id_x = 0; id_x < nx; id_x++)
                    {
                        int iterseed = iter * 134775813;
                        int color = perm[i];
                        // randomized color order，6 种网格随机取半模式
                        int is_red = ((id_z & 1) == 1) && (color == 1);
                        int is_green = ((id_x & 1) == 1) && (color == 2);
                        int is_blue = ((id_z & 1) == 0) && (color == 3);
                        int is_yellow = ((id_x & 1) == 0) && (color == 4);
                        int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                        int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                        // randomized direction，其实只有 4 种模式
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

                            // 读取上次计算的结果
                            float i_height = _temp_height[idx];
                            float i_material = _temp_material[idx];
                            float i_debris = _temp_debris[idx];
                            float i_sediment = _sediment[idx];

                            // 移除 邻格 被边界 clamp 的格子
                            int samplex = clamp(id_x + dx, 0, clamp_x);
                            int samplez = clamp(id_z + dz, 0, clamp_z);
                            int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                            // If we have closed borders, pretend a valid source to create
                            // a streak condition
                            if (validsource)
                            {
                                // 移除被标记为边界的格子 ？？？
                                validsource = validsource || !openborder;

                                // 邻格 的索引号
                                int j_idx = Pos2Idx(samplex, samplez, nx);

                                // 邻格 的数据
                                float j_height = _temp_height[j_idx]; // height 的值一定是有的
                                float j_material = validsource ? _temp_material[j_idx] : 0.0f;	// 无效的格子不会被计算，所以可能没有值
                                float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;

                                float j_sediment = validsource ? _sediment[j_idx] : 0.0f;

                                // 包含 height，debris，water 的高度差，注意这里是 邻格 - 本格
                                float m_diff = (j_height + j_debris + j_material) - (i_height + i_debris + i_material);

                                float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f); // 用于计算斜率的底边长度

                                // 邻格 跟 本格 比，高的是 中格，另一个是 邻格
                                int cidx = 0;   // 中格的 id_x
                                int cidz = 0;   // 中格的 id_z

                                float c_height = 0.0f;      // 中格

                                float c_material = 0.0f;    // 中格
                                float n_material = 0.0f;    // 邻格

                                float c_sediment = 0.0f;
                                float n_sediment = 0.0f;

                                float c_debris = 0.0f;
                                float n_debris = 0.0f;

                                float h_diff = 0.0f;

                                int c_idx = 0;  // 中格的 idx
                                int n_idx = 0;  // 邻格的 idx
                                int dx_check = 0;   // 中格 指向 邻格 的方向
                                int dz_check = 0;
                                int is_mh_diff_same_sign = 0;

                                // 如果邻格比本格高，邻格->中格，本格->邻格
                                // 高的是 中格
                                if (m_diff > 0.0f)
                                {
                                    // look at j's neighbours
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
                                    // look at i's neighbours
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
                                            // If we have closed borders, pretend a valid source to create
                                            // a streak condition
                                            // TODO: what is streak condition?
                                            tmp_validsource = tmp_validsource || !openborder;
                                            int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                            // 中格周围的邻格 水，碎屑 的高度
                                            float tmp_n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;
                                            float tmp_n_debris = tmp_validsource ? _temp_debris[tmp_j_idx] : 0.0f;

                                            // 中格周围的邻格 地面 的高度
                                            float n_height = _temp_height[tmp_j_idx];

                                            // 中格周围的邻格 无水高度 - 中格 无水高度
                                            float tmp_h_diff = n_height + tmp_n_debris - (c_height + c_debris);

                                            // 中格周围的邻格 带水高度 - 中格 带水高度
                                            float tmp_m_diff = (n_height + tmp_n_debris + tmp_n_material) - (c_height + c_debris + c_material);

                                            float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;

                                            float _gridbias = gridbias;

                                            _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                            // 修正高度差
                                            if (tmp_dx && tmp_dz)
                                                tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                            else // !tmp_dx || !tmp_dz
                                                tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                            // diff_idx = 1 的时候，前面比较过格子的总高度差，此时
                                            // 如果周边格子 不比我高，因为前面有过交换，所以至少有一个格子满足这个要求
                                            // diff_idx = 0 的时候，下面的条件不一定能满足。格子的地面有可能是最低的
                                            if (tmp_diff <= 0.0f)	// 只统计比我低的邻格，所以 高度差 的说法改为 深度差
                                            {
                                                // 指定方向上，中格(我) 与 邻格 的深度差
                                                // dir_probs[0] 可能此时 >0 不会进来，此时 dir_probs[0] 保持默认值 0
                                                if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                                    dir_probs[diff_idx] = tmp_diff;

                                                // 按格子总高度计算的时候，记录 tmp_diff 最深的深度，作为 dir_prob
                                                if (diff_idx && (tmp_diff < dir_prob))
                                                    dir_prob = tmp_diff;

                                                // 记录比 中格 低的邻格的深度和
                                                sum_diffs[diff_idx] += tmp_diff;
                                            }
                                        }
                                    }

                                    if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                                    {
                                        // 按 (地面高度差+碎屑高度差)来计算时，流动概率 = 指定方向上的深度差 / 最大深度差
                                        dir_prob = dir_probs[diff_idx] / dir_prob;
                                    }
                                    else // add by wangbo
                                    {
                                        dir_prob = 0.0f;
                                    }

                                    // 另一种计算方法：指定方向上的流动概率 = 指定方向上的深度差 / 所有比我低的邻格的深度差之和
                                    // 这种概率显然比上一种方法的计算结果要 低
                                    // diff_idx == 1 时，深度差 以 (地面高度差+碎屑高度差) 来计算时
                                    // diff_idx == 0 时，深度差 以 (地面高度差) 来计算时，可能不存在，不过已经取默认值为 0 了
                                    if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                                    {
                                        dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                                    }
                                    else // add by wangbo
                                    {
                                        dir_probs[diff_idx] = 0.0f;
                                    }
                                }

                                // 最多可供流失的高度差
                                float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;

                                // 它首先会被clamp(0,c_material)，以保证有足够的材料被移动
                                movable_mat = clamp(movable_mat * 0.5f, 0.0f, c_material);

                                // 以 height + debris + water 来计算
                                float l_rat = dir_probs[1];

                                // TODO: What is a good limit here?
                                // 让水流继续保持足够的水量
                                if (quant_amt > 0.001)	// 默认 = 1.0
                                    movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                                else
                                    movable_mat *= l_rat; // 乘上概率，这样随着水量快速减少，水流很快就消失了

                                float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                                int cond = 0;
                                if (dir_prob >= 1.0f)
                                    cond = 1;
                                else
                                {
                                    // Making sure all drops are moving
                                    dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                                    unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                                    unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                                    cond = randval < cutoff;
                                }

                                // 不参与计算的格子，或者没有流动概率的格子
                                if (!cond)
                                    diff = 0.0f;

                                /////////////////////////////////////////////////////////////
                                // 下面开始计算侵蚀，河床，河堤
                                /////////////////////////////////////////////////////////////

                                // 通过 h_diff 计算 沉积条件，用于产生 河床 和 河堤
                                float slope_cont = (delta_x > 0.0f) ? (h_diff / delta_x) : 0.0f;	// 斜率=对边/临边

                                // 沉积因子 = 1 / (1 + 斜率)，斜率大的地方沉积的倾向小
                                float kd_factor = clamp((1 / (1 + (slope_contribution_factor * slope_cont))), 0.0f, 1.0f);

                                // 当前迭代索引iter(1~50) / 最大侵蚀迭代次数(参数面板填写)
                                float norm_iter = clamp(((float)iter / (float)max_erodability_iteration), 0.0f, 1.0f);

                                // 侵蚀因子：地面(height+debris)斜率，斜率贡献因子，深度差/深度差之和，初始侵蚀因子，迭代递加侵蚀因子
                                float ks_factor = clamp((1 - (slope_contribution_factor * exp(-slope_cont))) * sqrt(dir_probs[0]) *
                                    (initial_erodability_factor + ((1.0f - initial_erodability_factor) * sqrt(norm_iter))),
                                    0.0f, 1.0f);

                                // 中格侵蚀率
                                float c_ks = global_erosionrate * erosionrate * erodability * ks_factor;

                                // 邻格沉积率
                                float n_kd = depositionrate * kd_factor;
                                n_kd = clamp(n_kd, 0.0f, 1.0f);

                                // 类似通过风化率计算侵蚀产生的碎屑
                                float _removalrate = removalrate;
                                float bedrock_density = 1.0f - _removalrate;

                                // 通过 m_diff 可能包含水面的高度差
                                //Kc <Sediment capacity>
                                //Kd <Deposition rate>
                                //Ks <Erodability>
                                float abs_diff = (diff < 0.0f) ? -diff : diff;		// 可能包含水面的高度差

                                // ​sedimentcap：泥沙容量，每单位流动水可携带的泥沙量。
                                // 容量越大，材料在开始沉积多余沉积物之前被侵蚀的时间就越长。
                                float sediment_limit = sedimentcap * abs_diff;		// 根据泥沙容量计算的水中泥沙上限

                                float ent_check_diff = sediment_limit - c_sediment;

                                // sediment_limit - c_sediment > 0，
                                // 意味着水中可以携带的泥沙上限超过了 中格 的沉积物
                                // 这会导致更大的侵蚀，倾向挖宽河床，这主要是一个侵蚀过程

                                // sediment_limit - c_sediment < 0，
                                // 意味着水中可以携带的泥沙的能力降低，
                                // 倾向于水中的泥沙向地面沉积，这是主要是一个沉积过程

                                // for current cell
                                if (ent_check_diff > 0.0f) // sediment_limit > c_sediment
                                {
                                    // 中格因为侵蚀而被溶解的物质
                                    float dissolve_amt = c_ks * bed_erosionrate_factor * abs_diff;

                                    // 优先溶解碎屑层，但是碎屑层最大的量也只有 c_debris
                                    float dissolved_debris = min(c_debris, dissolve_amt);

                                    // 中格碎屑被溶解后，还剩下的部分
                                    _debris[c_idx] -= dissolved_debris;	// 碎屑被侵蚀

                                    // 如果中格碎屑被溶完了，还不够，就开始溶解 height 层
                                    _height[c_idx] -= (dissolve_amt - dissolved_debris);  // height 被侵蚀

                                    // 沉积，数据来自上一 frame 计算的结果，50{8{}} 循环内简单重复计算
                                    // 中格的沉积物被冲走一半
                                    _sediment[c_idx] -= c_sediment / 2;	// 沉积物被侵蚀

                                    // 风化后仍有剩余
                                    if (bedrock_density > 0.0f)
                                    {
                                        // 被冲走的那一半沉积物 + 溶解物的沉积，这些沉积会堆积到 邻格
                                        float newsediment = c_sediment / 2 + (dissolve_amt * bedrock_density);

                                        // 假设沉积物都会堆积到邻格，如果超过最大碎屑高度
                                        if (n_sediment + newsediment > max_debris_depth)
                                        {
                                            // 回滚
                                            float rollback = n_sediment + newsediment - max_debris_depth;
                                            // 回滚量不可以超过刚刚计算的沉积高度
                                            rollback = min(rollback, newsediment);
                                            // return the excess sediment
                                            _height[c_idx] += rollback / bedrock_density; 	// 向上修正 height 高度
                                            newsediment -= rollback;						// 向下修正 沉积高度
                                        }
                                        _sediment[n_idx] += newsediment;		// 邻格沉积物增加
                                    }
                                }
                                else // sediment_limit <= c_sediment，这主要是一个沉积过程
                                {
                                    float c_kd = depositionrate * kd_factor;      // 计算沉积系数

                                    c_kd = clamp(c_kd, 0.0f, 1.0f);

                                    {
                                        // -ent_check_diff = 高度差产生的泥沙 - 能被水流携带走的泥沙
                                        // 这些过剩的泥沙会成为 碎屑 和 沉积物
                                        // 碎屑的定义：高度差侵蚀的直接结果
                                        // 沉积的定义：泥沙被被水搬运到邻格沉降的结果
                                        _debris[c_idx] += (c_kd * -ent_check_diff);	    	// 中格碎屑增加
                                        _sediment[c_idx] = (1 - c_kd) * -ent_check_diff;	// 剩下的变成中格的沉积物

                                        n_sediment += sediment_limit;	    				// 被带走的沉积物到了邻格
                                        _debris[n_idx] += (n_kd * n_sediment);  			// 邻格的碎屑增加
                                        _sediment[n_idx] = (1 - n_kd) * n_sediment; 		// 剩下的变成邻格的沉积物
                                    }

                                    // 河岸 河床的侵蚀，碎屑转移过程，不涉及 sediment
                                    int b_idx = 0;					// 岸的位置索引
                                    int r_idx = 0;					// 河的位置索引
                                    float b_material = 0.0f;		// 岸的水高度
                                    float r_material = 0.0f;		// 河的水高度
                                    float b_debris = 0.0f;			// 岸的碎屑高度
                                    float r_debris = 0.0f;			// 河的碎屑高度
                                    float r_sediment = 0.0f;		// 河的沉积高度

                                    if (is_mh_diff_same_sign)		// 中格的水高，地也高
                                    {
                                        b_idx = c_idx;				// 中格地高，是岸
                                        r_idx = n_idx;				// 邻格地低，是河

                                        b_material = c_material;	// 岸的水高度
                                        r_material = n_material;	// 河的水高度

                                        b_debris = c_debris;		// 岸的碎屑高度
                                        r_debris = n_debris;		// 河的碎屑高度

                                        r_sediment = n_sediment;	// 河的沉积高度
                                    }
                                    else							// 中格 水高，地低
                                    {
                                        b_idx = n_idx;				// 邻格地高，是岸
                                        r_idx = c_idx;				// 中格地低，是河

                                        b_material = n_material;
                                        r_material = c_material;

                                        b_debris = n_debris;
                                        r_debris = c_debris;

                                        r_sediment = c_sediment;
                                    }

                                    // 河中每单位水的侵蚀量
                                    float erosion_per_unit_water = global_erosionrate * erosionrate * bed_erosionrate_factor * erodability * ks_factor;

                                    // 河中有水 && 岸的水量/河的水量<max_bank_bed_ratio(0.5) && 河的沉积量>
                                    if (r_material != 0.0f &&
                                        (b_material / r_material) < max_bank_bed_ratio &&
                                        r_sediment > (erosion_per_unit_water * max_bank_bed_ratio))
                                    {
                                        // NOTE: Increase the bank erosion to get a certain
                                        // angle faster. This would make the river cuts less
                                        // deep.
                                        float height_to_erode = global_erosionrate * erosionrate * bank_erosionrate_factor * erodability * ks_factor;

                                        float _bank_angle = bank_angle;

                                        _bank_angle = clamp(_bank_angle, 0.0f, 90.0f);
                                        float safe_diff = _bank_angle < 90.0f ? tan(_bank_angle * M_PI / 180.0) * delta_x : 1e10f;
                                        float target_height_removal = (h_diff - safe_diff) < 0.0f ? 0.0f : h_diff - safe_diff;

                                        float dissolve_amt = clamp(height_to_erode, 0.0f, target_height_removal);
                                        float dissolved_debris = min(b_debris, dissolve_amt);

                                        _debris[b_idx] -= dissolved_debris; // 岸的碎屑被侵蚀

                                        float division = 1 / (1 + safe_diff);

                                        _height[b_idx] -= (dissolve_amt - dissolved_debris); // 岸的 height 被侵蚀

                                        if (bedrock_density > 0.0f) // 有沉积
                                        {
                                            float newdebris = (1 - division) * (dissolve_amt * bedrock_density);
                                            if (b_debris + newdebris > max_debris_depth)
                                            {
                                                float rollback = b_debris + newdebris - max_debris_depth;
                                                rollback = min(rollback, newdebris);
                                                // return the excess debris
                                                _height[b_idx] += rollback / bedrock_density;
                                                newdebris -= rollback;
                                            }
                                            _debris[b_idx] += newdebris; // 河岸沉积

                                            newdebris = division * (dissolve_amt * bedrock_density);

                                            if (r_debris + newdebris > max_debris_depth)
                                            {
                                                float rollback = r_debris + newdebris - max_debris_depth;
                                                rollback = min(rollback, newdebris);
                                                // return the excess debris
                                                _height[b_idx] += rollback / bedrock_density;
                                                newdebris -= rollback;
                                            }
                                            _debris[r_idx] += newdebris; // 河床沉积
                                        }
                                    }
                                }

                                // Update the material level
                                // 水往低处流，这很简单，麻烦的是上面，注意这里的索引号，不是中格-邻格模式，而是本格-邻格模式
                                _material[idx] = i_material + diff;		// 本格更新水的高度
                                _material[j_idx] = j_material - diff;	// 邻格更新水的高度

                            }
                        }
                    }
                }
            }
        }

        // 测量计算时间 结束
        finish = clock();
        double duration = ((double)finish - (double)start) / CLOCKS_PER_SEC;
        printf("... ... end!\n");
        printf("-= %f seconds =-\n", duration);


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 将计算结果返回给外部数据，并删除临时属性
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                height[idx] = _height[idx]; // 计算结果返回给外部数据
                material[idx] = _material[idx];
                debris[idx] = _debris[idx];
                sediment[idx] = _sediment[idx];

                if (visualEnable)
                {
                    float coef = min(1, (material[idx] / 1.0));
                    attr_color[idx] = (1 - coef) * attr_color[idx] + coef * vec3f(0.15, 0.45, 0.9);
                }
            }
        }

        terrain->verts.erase_attr("_sediment");
        terrain->verts.erase_attr("_height");
        terrain->verts.erase_attr("_temp_height");
        terrain->verts.erase_attr("_debris");
        terrain->verts.erase_attr("_temp_debris");
        terrain->verts.erase_attr("_material");
        terrain->verts.erase_attr("_temp_material");

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_slump_b4,
    { /* inputs: */ {
            "prim_2DGrid",

            // 需要用到的属性/数据
            {"string", "heightLayerName", "height"},
            {"string", "materialLayerName", "water"},
            {"string", "sedimentLayerName", "sediment"},
            {"string", "debrisLayerName", "debris"},

            // 杂项
            {"int", "openborder", "0"}, // 获取边界标记
            {"int", "visualEnable", "0"}, // 开启可视化

            // 侵蚀主参数
            {"float", "global_erosionrate", "1.0"}, // 全局侵蚀率
            {"float", "erodability", "1.0"}, // 侵蚀能力
            {"float", "erosionrate", "0.4"}, // 侵蚀率
            {"float", "bank_angle", "70.0"}, // 河堤侵蚀角度
            {"float", "seed", "12.34"},

            // 高级参数
            {"float", "removalrate", "0.1"}, // 风化率/水吸收率
            {"float", "max_debris_depth", "5.0"}, // 碎屑最大深度
            {"float", "gridbias", "0.0"},

            // 侵蚀能力调整
            {"int", "max_erodability_iteration", "5"}, // 最大侵蚀能力迭代次数
            {"float", "initial_erodability_factor", "0.5"}, // 初始侵蚀能力因子
            {"float", "slope_contribution_factor", "0.8"}, // “地面斜率”对“侵蚀”和“沉积”的影响，“地面斜率大” -> 侵蚀因子大，沉积因子小

            // 河床参数
            {"float", "bed_erosionrate_factor", "1.0"}, // 河床侵蚀率因子
            {"float", "depositionrate", "0.01"}, // 沉积率
            {"float", "sedimentcap", "10.0"}, // 高度差转变为沉积物的比率 / 泥沙容量，每单位流动水可携带的泥沙量

            // 河堤参数
            {"float", "bank_erosionrate_factor", "1.0"}, // 河堤侵蚀率因子
            {"float", "max_bank_bed_ratio", "0.5"}, // 高于这个比值的河岸将不会在侵蚀中被视为河岸，会停止侵蚀

            // 河网控制
            {"float", "quant_amt", "0.05"}, // 流量维持率，越高河流流量越稳定
            {"int", "iterations", "40"}, // 流淌的总迭代次数

        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
        }, /* category: */ {
            "deprecated",
        } });
// 上面的（崩塌+侵蚀）节点 erode_slump_b4 可以废弃了，由下面的 
// erode_tumble_material_v4
// 节点代替
struct erode_tumble_material_v4 : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        // 用于调试和可视化
        auto visualEnable = get_input<NumericObject>("visualEnable")->get<int>();
        //  if (visualEnable) {
        if (!terrain->verts.has_attr("clr"))
        {
            auto& _clr = terrain->verts.add_attr<vec3f>("clr");
            std::fill(_clr.begin(), _clr.end(), vec3f(1.0, 1.0, 1.0));
        }
        auto& attr_color = terrain->verts.attr<vec3f>("clr");

        if (!terrain->verts.has_attr("debug"))
        {
            auto& _debug = terrain->verts.add_attr<float>("debug");
            std::fill(_debug.begin(), _debug.end(), 0);
        }
        auto& attr_debug = terrain->verts.attr<float>("debug");
        //  }

        ///////////////////////////////////////////////////////////////////////

        // 侵蚀主参数
        auto global_erosionrate = get_input<NumericObject>("global_erosionrate")->get<float>(); // 1 全局侵蚀率
        auto erodability = get_input<NumericObject>("erodability")->get<float>(); // 1.0 侵蚀能力
        auto erosionrate = get_input<NumericObject>("erosionrate")->get<float>(); // 0.4 侵蚀率
        auto bank_angle = get_input<NumericObject>("bank_angle")->get<float>(); // 70.0 河堤侵蚀角度
        auto seed = get_input<NumericObject>("seed")->get<float>(); // 12.34

        // 高级参数
        auto removalrate = get_input<NumericObject>("removalrate")->get<float>(); // 0.0 风化率/水吸收率
        auto max_debris_depth = get_input<NumericObject>("max_debris_depth")->get<float>();// 5	碎屑最大深度
        auto gridbias = get_input<NumericObject>("gridbias")->get<float>(); // 0.0

        // 侵蚀能力调整
        auto max_erodability_iteration = get_input<NumericObject>("max_erodability_iteration")->get<int>(); // 5
        auto initial_erodability_factor = get_input<NumericObject>("initial_erodability_factor")->get<float>(); // 0.5
        auto slope_contribution_factor = get_input<NumericObject>("slope_contribution_factor")->get<float>(); // 0.8

        // 河床参数
        auto bed_erosionrate_factor = get_input<NumericObject>("bed_erosionrate_factor")->get<float>(); // 1 河床侵蚀率因子
        auto depositionrate = get_input<NumericObject>("depositionrate")->get<float>(); // 0.01 沉积率
        auto sedimentcap = get_input<NumericObject>("sedimentcap")->get<float>(); // 10.0 高度差转变为沉积物的比率 / 泥沙容量，每单位流动水可携带的泥沙量

        // 河堤参数
        auto bank_erosionrate_factor = get_input<NumericObject>("bank_erosionrate_factor")->get<float>(); // 1.0 河堤侵蚀率因子
        auto max_bank_bed_ratio = get_input<NumericObject>("max_bank_bed_ratio")->get<float>(); // 0.5 The maximum of bank to bed water column height ratio
                                                                                                // 高于这个比值的河岸将不会在侵蚀中被视为河岸，会停止侵蚀
        // 河流控制
        auto quant_amt = get_input<NumericObject>("quant_amt")->get<float>(); // 0.05 流量维持率，越高流量越稳定
        auto iterations = get_input<NumericObject>("iterations")->get<int>(); // 流淌的总迭代次数

        ///////////////////////////////////////////////////////////////////////

        std::uniform_real_distribution<float> distr(0.0, 1.0);
        //        auto seed = get_input<NumericObject>("seed")->get<float>();
        //        auto iterations = get_input<NumericObject>("iterations")->get<int>();
        auto iter = get_input<NumericObject>("iter")->get<int>();
        auto i = get_input<NumericObject>("i")->get<int>();
        auto openborder = get_input<NumericObject>("openborder")->get<int>();

        auto& perm = get_input<ListObject>("perm")->get2<int>();
        auto& p_dirs = get_input<ListObject>("p_dirs")->get2<int>();
        auto& x_dirs = get_input<ListObject>("x_dirs")->get2<int>();

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算用的临时属性，必须要有
        ////////////////////////////////////////////////////////////////////////////////////////
        if (!terrain->verts.has_attr("_height") ||
            !terrain->verts.has_attr("_temp_height") ||
            !terrain->verts.has_attr("_material") ||
            !terrain->verts.has_attr("_temp_material") ||
            !terrain->verts.has_attr("_debris") ||
            !terrain->verts.has_attr("_temp_debris") ||
            !terrain->verts.has_attr("_sediment"))
        {
            // height 和 debris 数据要从外面读取，所以属性中要有 height 和 debris
            zeno::log_error("Node [erode_tumble_material_v4], no such data layer named '{}' or '{}' or '{}' or '{}' or '{}' or '{}' or '{}'.",
                "_height", "_temp_height", "_material", "_temp_material", "_debris", "_temp_debris", "_sediment");
        }
        auto& _height = terrain->verts.add_attr<float>("_height");
        auto& _temp_height = terrain->verts.add_attr<float>("_temp_height");
        auto& _material = terrain->verts.add_attr<float>("_material");
        auto& _temp_material = terrain->verts.add_attr<float>("_temp_material");
        auto& _debris = terrain->verts.add_attr<float>("_debris");
        auto& _temp_debris = terrain->verts.add_attr<float>("_temp_debris");
        auto& _sediment = terrain->verts.add_attr<float>("_sediment");

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////
        // 新的，确定的，随机方向，依据上次的计算结果进行计算
#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int iterseed = iter * 134775813;
                int color = perm[i];
                // randomized color order，6 种网格随机取半模式
                int is_red = ((id_z & 1) == 1) && (color == 1);
                int is_green = ((id_x & 1) == 1) && (color == 2);
                int is_blue = ((id_z & 1) == 0) && (color == 3);
                int is_yellow = ((id_x & 1) == 0) && (color == 4);
                int is_x_turn_x = ((id_x & 1) == 1) && ((color == 5) || (color == 6));
                int is_x_turn_y = ((id_x & 1) == 0) && ((color == 7) || (color == 8));
                // randomized direction，其实只有 4 种模式
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

                    // 读取上次计算的结果
                    float i_height = _temp_height[idx];
                    float i_material = _temp_material[idx];
                    float i_debris = _temp_debris[idx];
                    float i_sediment = _sediment[idx];

                    // 移除 邻格 被边界 clamp 的格子
                    int samplex = clamp(id_x + dx, 0, clamp_x);
                    int samplez = clamp(id_z + dz, 0, clamp_z);
                    int validsource = (samplex == id_x + dx) && (samplez == id_z + dz);

                    // If we have closed borders, pretend a valid source to create
                    // a streak condition
                    if (validsource)
                    {
                        // 移除被标记为边界的格子 ？？？
                        validsource = validsource || !openborder;

                        // 邻格 的索引号
                        int j_idx = Pos2Idx(samplex, samplez, nx);

                        // 邻格 的数据
                        float j_height = _temp_height[j_idx]; // height 的值一定是有的
                        float j_material = validsource ? _temp_material[j_idx] : 0.0f;	// 无效的格子不会被计算，所以可能没有值
                        float j_debris = validsource ? _temp_debris[j_idx] : 0.0f;

                        float j_sediment = validsource ? _sediment[j_idx] : 0.0f;

                        // 包含 height，debris，water 的高度差，注意这里是 邻格 - 本格
                        float m_diff = (j_height + j_debris + j_material) - (i_height + i_debris + i_material);

                        float delta_x = cellSize * (dx && dz ? 1.4142136f : 1.0f); // 用于计算斜率的底边长度

                        // 邻格 跟 本格 比，高的是 中格，另一个是 邻格
                        int cidx = 0;   // 中格的 id_x
                        int cidz = 0;   // 中格的 id_z

                        float c_height = 0.0f;      // 中格

                        float c_material = 0.0f;    // 中格
                        float n_material = 0.0f;    // 邻格

                        float c_sediment = 0.0f;
                        float n_sediment = 0.0f;

                        float c_debris = 0.0f;
                        float n_debris = 0.0f;

                        float h_diff = 0.0f;

                        int c_idx = 0;  // 中格的 idx
                        int n_idx = 0;  // 邻格的 idx
                        int dx_check = 0;   // 中格 指向 邻格 的方向
                        int dz_check = 0;
                        int is_mh_diff_same_sign = 0;

                        // 如果邻格比本格高，邻格->中格，本格->邻格
                        // 高的是 中格
                        if (m_diff > 0.0f)
                        {
                            // look at j's neighbours
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
                            // look at i's neighbours
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
                                    // If we have closed borders, pretend a valid source to create
                                    // a streak condition
                                    // TODO: what is streak condition?
                                    tmp_validsource = tmp_validsource || !openborder;
                                    int tmp_j_idx = Pos2Idx(tmp_samplex, tmp_samplez, nx);

                                    // 中格周围的邻格 水，碎屑 的高度
                                    float tmp_n_material = tmp_validsource ? _temp_material[tmp_j_idx] : 0.0f;
                                    float tmp_n_debris = tmp_validsource ? _temp_debris[tmp_j_idx] : 0.0f;

                                    // 中格周围的邻格 地面 的高度
                                    float n_height = _temp_height[tmp_j_idx];

                                    // 中格周围的邻格 无水高度 - 中格 无水高度
                                    float tmp_h_diff = n_height + tmp_n_debris - (c_height + c_debris);

                                    // 中格周围的邻格 带水高度 - 中格 带水高度
                                    float tmp_m_diff = (n_height + tmp_n_debris + tmp_n_material) - (c_height + c_debris + c_material);

                                    float tmp_diff = diff_idx == 0 ? tmp_h_diff : tmp_m_diff;

                                    float _gridbias = gridbias;

                                    _gridbias = clamp(_gridbias, -1.0f, 1.0f);

                                    // 修正高度差
                                    if (tmp_dx && tmp_dz)
                                        tmp_diff *= clamp(1.0f - _gridbias, 0.0f, 1.0f) / 1.4142136f;
                                    else // !tmp_dx || !tmp_dz
                                        tmp_diff *= clamp(1.0f + _gridbias, 0.0f, 1.0f);

                                    // diff_idx = 1 的时候，前面比较过格子的总高度差，此时
                                    // 如果周边格子 不比我高，因为前面有过交换，所以至少有一个格子满足这个要求
                                    // diff_idx = 0 的时候，下面的条件不一定能满足。格子的地面有可能是最低的
                                    if (tmp_diff <= 0.0f)	// 只统计比我低的邻格，所以 高度差 的说法改为 深度差
                                    {
                                        // 指定方向上，中格(我) 与 邻格 的深度差
                                        // dir_probs[0] 可能此时 >0 不会进来，此时 dir_probs[0] 保持默认值 0
                                        if ((dx_check == tmp_dx) && (dz_check == tmp_dz))
                                            dir_probs[diff_idx] = tmp_diff;

                                        // 按格子总高度计算的时候，记录 tmp_diff 最深的深度，作为 dir_prob
                                        if (diff_idx && (tmp_diff < dir_prob))
                                            dir_prob = tmp_diff;

                                        // 记录比 中格 低的邻格的深度和
                                        sum_diffs[diff_idx] += tmp_diff;
                                    }
                                }
                            }

                            if (diff_idx && (dir_prob > 0.001f || dir_prob < -0.001f))
                            {
                                // 按 (地面高度差+碎屑高度差)来计算时，流动概率 = 指定方向上的深度差 / 最大深度差
                                dir_prob = dir_probs[diff_idx] / dir_prob;
                            }
                            else // add by wangbo
                            {
                                dir_prob = 0.0f;
                            }

                            // 另一种计算方法：指定方向上的流动概率 = 指定方向上的深度差 / 所有比我低的邻格的深度差之和
                            // 这种概率显然比上一种方法的计算结果要 低
                            // diff_idx == 1 时，深度差 以 (地面高度差+碎屑高度差) 来计算时
                            // diff_idx == 0 时，深度差 以 (地面高度差) 来计算时，可能不存在，不过已经取默认值为 0 了
                            if (sum_diffs[diff_idx] > 0.001f || sum_diffs[diff_idx] < -0.001f)
                            {
                                dir_probs[diff_idx] = dir_probs[diff_idx] / sum_diffs[diff_idx];
                            }
                            else // add by wangbo
                            {
                                dir_probs[diff_idx] = 0.0f;
                            }
                        }

                        // 最多可供流失的高度差
                        float movable_mat = (m_diff < 0.0f) ? -m_diff : m_diff;

                        // 它首先会被clamp(0,c_material)，以保证有足够的材料被移动
                        movable_mat = clamp(movable_mat * 0.5f, 0.0f, c_material);

                        // 以 height + debris + water 来计算
                        float l_rat = dir_probs[1];

                        // TODO: What is a good limit here?
                        // 让水流继续保持足够的水量
                        if (quant_amt > 0.001)	// 默认 = 1.0
                            movable_mat = clamp(quant_amt * ceil((movable_mat * l_rat) / quant_amt), 0.0f, c_material);
                        else
                            movable_mat *= l_rat; // 乘上概率，这样随着水量快速减少，水流很快就消失了

                        float diff = (m_diff > 0.0f) ? movable_mat : -movable_mat;

                        int cond = 0;
                        if (dir_prob >= 1.0f)
                            cond = 1;
                        else
                        {
                            // Making sure all drops are moving
                            dir_prob = dir_prob * dir_prob * dir_prob * dir_prob;
                            unsigned int cutoff = (unsigned int)(dir_prob * 4294967295.0);   // 0 ~ 1 映射到 0 ~ 4294967295.0
                            unsigned int randval = erode_random(seed, (idx + nx * nz) * 8 + color + iterseed);
                            cond = randval < cutoff;
                        }

                        // 不参与计算的格子，或者没有流动概率的格子
                        if (!cond)
                            diff = 0.0f;

                        /////////////////////////////////////////////////////////////
                        // 下面开始计算侵蚀，河床，河堤
                        /////////////////////////////////////////////////////////////

                        // 通过 h_diff 计算 沉积条件，用于产生 河床 和 河堤
                        float slope_cont = (delta_x > 0.0f) ? (h_diff / delta_x) : 0.0f;	// 斜率=对边/临边

                        // 沉积因子 = 1 / (1 + 斜率)，斜率大的地方沉积的倾向小
                        float kd_factor = clamp((1 / (1 + (slope_contribution_factor * slope_cont))), 0.0f, 1.0f);

                        // 当前迭代索引iter(1~50) / 最大侵蚀迭代次数(参数面板填写)
                        float norm_iter = clamp(((float)iter / (float)max_erodability_iteration), 0.0f, 1.0f);

                        // 侵蚀因子：地面(height+debris)斜率，斜率贡献因子，深度差/深度差之和，初始侵蚀因子，迭代递加侵蚀因子
                        float ks_factor = clamp((1 - (slope_contribution_factor * exp(-slope_cont))) * sqrt(dir_probs[0]) *
                            (initial_erodability_factor + ((1.0f - initial_erodability_factor) * sqrt(norm_iter))),
                            0.0f, 1.0f);

                        // 中格侵蚀率
                        float c_ks = global_erosionrate * erosionrate * erodability * ks_factor;

                        // 邻格沉积率
                        float n_kd = depositionrate * kd_factor;
                        n_kd = clamp(n_kd, 0.0f, 1.0f);

                        // 类似通过风化率计算侵蚀产生的碎屑
                        float _removalrate = removalrate;
                        float bedrock_density = 1.0f - _removalrate;

                        // 通过 m_diff 可能包含水面的高度差
                        //Kc <Sediment capacity>
                        //Kd <Deposition rate>
                        //Ks <Erodability>
                        float abs_diff = (diff < 0.0f) ? -diff : diff;		// 可能包含水面的高度差

                        // ​sedimentcap：泥沙容量，每单位流动水可携带的泥沙量。
                        // 容量越大，材料在开始沉积多余沉积物之前被侵蚀的时间就越长。
                        float sediment_limit = sedimentcap * abs_diff;		// 根据泥沙容量计算的水中泥沙上限

                        float ent_check_diff = sediment_limit - c_sediment;

                        // sediment_limit - c_sediment > 0，
                        // 意味着水中可以携带的泥沙上限超过了 中格 的沉积物
                        // 这会导致更大的侵蚀，倾向挖宽河床，这主要是一个侵蚀过程

                        // sediment_limit - c_sediment < 0，
                        // 意味着水中可以携带的泥沙的能力降低，
                        // 倾向于水中的泥沙向地面沉积，这是主要是一个沉积过程

                        // for current cell
                        if (ent_check_diff > 0.0f) // sediment_limit > c_sediment
                        {
                            // 中格因为侵蚀而被溶解的物质
                            float dissolve_amt = c_ks * bed_erosionrate_factor * abs_diff;

                            // 优先溶解碎屑层，但是碎屑层最大的量也只有 c_debris
                            float dissolved_debris = min(c_debris, dissolve_amt);

                            // 中格碎屑被溶解后，还剩下的部分
                            _debris[c_idx] -= dissolved_debris;	// 碎屑被侵蚀

                            // 如果中格碎屑被溶完了，还不够，就开始溶解 height 层
                            _height[c_idx] -= (dissolve_amt - dissolved_debris);  // height 被侵蚀

                            // 沉积，数据来自上一 frame 计算的结果，50{8{}} 循环内简单重复计算
                            // 中格的沉积物被冲走一半
                            _sediment[c_idx] -= c_sediment / 2;	// 沉积物被侵蚀

                            // 风化后仍有剩余
                            if (bedrock_density > 0.0f)
                            {
                                // 被冲走的那一半沉积物 + 溶解物的沉积，这些沉积会堆积到 邻格
                                float newsediment = c_sediment / 2 + (dissolve_amt * bedrock_density);

                                // 假设沉积物都会堆积到邻格，如果超过最大碎屑高度
                                if (n_sediment + newsediment > max_debris_depth)
                                {
                                    // 回滚
                                    float rollback = n_sediment + newsediment - max_debris_depth;
                                    // 回滚量不可以超过刚刚计算的沉积高度
                                    rollback = min(rollback, newsediment);
                                    // return the excess sediment
                                    _height[c_idx] += rollback / bedrock_density; 	// 向上修正 height 高度
                                    newsediment -= rollback;						// 向下修正 沉积高度
                                }
                                _sediment[n_idx] += newsediment;		// 邻格沉积物增加
                            }
                        }
                        else // sediment_limit <= c_sediment，这主要是一个沉积过程
                        {
                            float c_kd = depositionrate * kd_factor;      // 计算沉积系数

                            c_kd = clamp(c_kd, 0.0f, 1.0f);

                            {
                                // -ent_check_diff = 高度差产生的泥沙 - 能被水流携带走的泥沙
                                // 这些过剩的泥沙会成为 碎屑 和 沉积物
                                // 碎屑的定义：高度差侵蚀的直接结果
                                // 沉积的定义：泥沙被被水搬运到邻格沉降的结果
                                _debris[c_idx] += (c_kd * -ent_check_diff);	    	// 中格碎屑增加
                                _sediment[c_idx] = (1 - c_kd) * -ent_check_diff;	// 剩下的变成中格的沉积物

                                n_sediment += sediment_limit;	    				// 被带走的沉积物到了邻格
                                _debris[n_idx] += (n_kd * n_sediment);  			// 邻格的碎屑增加
                                _sediment[n_idx] = (1 - n_kd) * n_sediment; 		// 剩下的变成邻格的沉积物
                            }

                            // 河岸 河床的侵蚀，碎屑转移过程，不涉及 sediment
                            int b_idx = 0;					// 岸的位置索引
                            int r_idx = 0;					// 河的位置索引
                            float b_material = 0.0f;		// 岸的水高度
                            float r_material = 0.0f;		// 河的水高度
                            float b_debris = 0.0f;			// 岸的碎屑高度
                            float r_debris = 0.0f;			// 河的碎屑高度
                            float r_sediment = 0.0f;		// 河的沉积高度

                            if (is_mh_diff_same_sign)		// 中格的水高，地也高
                            {
                                b_idx = c_idx;				// 中格地高，是岸
                                r_idx = n_idx;				// 邻格地低，是河

                                b_material = c_material;	// 岸的水高度
                                r_material = n_material;	// 河的水高度

                                b_debris = c_debris;		// 岸的碎屑高度
                                r_debris = n_debris;		// 河的碎屑高度

                                r_sediment = n_sediment;	// 河的沉积高度
                            }
                            else							// 中格 水高，地低
                            {
                                b_idx = n_idx;				// 邻格地高，是岸
                                r_idx = c_idx;				// 中格地低，是河

                                b_material = n_material;
                                r_material = c_material;

                                b_debris = n_debris;
                                r_debris = c_debris;

                                r_sediment = c_sediment;
                            }

                            // 河中每单位水的侵蚀量
                            float erosion_per_unit_water = global_erosionrate * erosionrate * bed_erosionrate_factor * erodability * ks_factor;

                            // 河中有水 && 岸的水量/河的水量<max_bank_bed_ratio(0.5) && 河的沉积量>
                            if (r_material != 0.0f &&
                                (b_material / r_material) < max_bank_bed_ratio &&
                                r_sediment > (erosion_per_unit_water * max_bank_bed_ratio))
                            {
                                // NOTE: Increase the bank erosion to get a certain
                                // angle faster. This would make the river cuts less
                                // deep.
                                float height_to_erode = global_erosionrate * erosionrate * bank_erosionrate_factor * erodability * ks_factor;

                                float _bank_angle = bank_angle;

                                _bank_angle = clamp(_bank_angle, 0.0f, 90.0f);
                                float safe_diff = _bank_angle < 90.0f ? tan(_bank_angle * M_PI / 180.0) * delta_x : 1e10f;
                                float target_height_removal = (h_diff - safe_diff) < 0.0f ? 0.0f : h_diff - safe_diff;

                                float dissolve_amt = clamp(height_to_erode, 0.0f, target_height_removal);
                                float dissolved_debris = min(b_debris, dissolve_amt);

                                _debris[b_idx] -= dissolved_debris; // 岸的碎屑被侵蚀

                                float division = 1 / (1 + safe_diff);

                                _height[b_idx] -= (dissolve_amt - dissolved_debris); // 岸的 height 被侵蚀

                                if (bedrock_density > 0.0f) // 有沉积
                                {
                                    float newdebris = (1 - division) * (dissolve_amt * bedrock_density);
                                    if (b_debris + newdebris > max_debris_depth)
                                    {
                                        float rollback = b_debris + newdebris - max_debris_depth;
                                        rollback = min(rollback, newdebris);
                                        // return the excess debris
                                        _height[b_idx] += rollback / bedrock_density;
                                        newdebris -= rollback;
                                    }
                                    _debris[b_idx] += newdebris; // 河岸沉积

                                    newdebris = division * (dissolve_amt * bedrock_density);

                                    if (r_debris + newdebris > max_debris_depth)
                                    {
                                        float rollback = r_debris + newdebris - max_debris_depth;
                                        rollback = min(rollback, newdebris);
                                        // return the excess debris
                                        _height[b_idx] += rollback / bedrock_density;
                                        newdebris -= rollback;
                                    }
                                    _debris[r_idx] += newdebris; // 河床沉积
                                }
                            }
                        }

                        // Update the material level
                        // 水往低处流，这很简单，麻烦的是上面，注意这里的索引号，不是中格-邻格模式，而是本格-邻格模式
                        _material[idx] = i_material + diff;		// 本格更新水的高度
                        _material[j_idx] = j_material - diff;	// 邻格更新水的高度

                    }
                }

            }
        }

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_tumble_material_v4,
    { /* inputs: */ {
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
            {"int", "visualEnable", "0"},

            // 侵蚀主参数
            {"float", "global_erosionrate", "1.0"}, // 全局侵蚀率
            {"float", "erodability", "1.0"}, // 侵蚀能力
            {"float", "erosionrate", "0.4"}, // 侵蚀率
            {"float", "bank_angle", "70.0"}, // 河堤侵蚀角度

            // 高级参数
            {"float", "removalrate", "0.1"}, // 风化率/水吸收率
            {"float", "max_debris_depth", "5.0"}, // 碎屑最大深度

            // 侵蚀能力调整
            {"int", "max_erodability_iteration", "5"}, // 最大侵蚀能力迭代次数
            {"float", "initial_erodability_factor", "0.5"}, // 初始侵蚀能力因子
            {"float", "slope_contribution_factor", "0.8"}, // “地面斜率”对“侵蚀”和“沉积”的影响，“地面斜率大” -> 侵蚀因子大，沉积因子小

            // 河床参数
            {"float", "bed_erosionrate_factor", "1.0"}, // 河床侵蚀率因子
            {"float", "depositionrate", "0.01"}, // 沉积率
            {"float", "sedimentcap", "10.0"}, // 高度差转变为沉积物的比率 / 泥沙容量，每单位流动水可携带的泥沙量

            // 河堤参数
            {"float", "bank_erosionrate_factor", "1.0"}, // 河堤侵蚀率因子
            {"float", "max_bank_bed_ratio", "0.5"}, // 高于这个比值的河岸将不会在侵蚀中被视为河岸，会停止侵蚀

            // 河网控制
            {"float", "quant_amt", "0.05"}, // 流量维持率，越高河流流量越稳定

        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
        }, /* category: */ {
            "erode",
        } });
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


struct erode_terrainHiMeLo : INode {
    void apply() override {
        auto& terrain = get_input<PrimitiveObject>("prim_2DGrid");

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