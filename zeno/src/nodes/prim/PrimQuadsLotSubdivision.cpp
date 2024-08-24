#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>

#include <unordered_set>
#include <random>
#include <algorithm>

namespace {

struct PrimQuadsLotSubdivision : zeno::INode {
    virtual void apply() override {
        auto inprim = get_input<zeno::PrimitiveObject>("input_quads_model"); //输入的多边形物体
        size_t num = get_input2<int>("num");
        auto outprim = std::make_shared<zeno::PrimitiveObject>(); //新生成的多边形物体
        auto roworcolumns = get_input2<bool>("row_or_columns");     //横着连线or竖着连线
        auto rcrc = get_input2<bool>("rcrc");
        auto random_rc = get_input2<bool>("random_rc");
        auto first_second_same = get_input2<bool>("first_second_same");
        float minoffset = get_input2<float>("min_offset");
        float maxoffset = get_input2<float>("max_offset");
        float first_edge_minoffset = get_input2<float>("first_edge_minoffset");
        float first_edge_maxoffset = get_input2<float>("first_edge_maxoffset");
        float second_edge_minoffset = get_input2<float>("second_edge_minoffset");
        float second_edge_maxoffset = get_input2<float>("second_edge_maxoffset");
        int same_seed = get_input2<int>("same_seed");
        int random_seed = get_input2<int>("random_seed");
        int first_seed = get_input2<int>("first_seed");
        int second_seed = get_input2<int>("second_seed");
        auto addattr = get_input2<bool>("add_attr");
        auto tagAttr = get_input2<std::string>("tag_attr");

        if (random_rc) {
            for (size_t rc_num = 0; rc_num < num; ++rc_num) {
                int r_seed = 0;
                if (random_seed == -1) {
                    r_seed = std::random_device{}();
                } else {
                    r_seed = random_seed + rc_num;
                }
                std::mt19937 gen(r_seed);
                std::uniform_int_distribution<int> uni(0, 1);
                int value = uni(gen);

                if (value == 0) {
                    int ss = 0;

                    for (size_t i = 0; i < inprim->polys.size(); i++) //遍历每一个四边面
                    {
                        size_t i0 = i * 4 + 0; //0   4
                        size_t i1 = i * 4 + 1; //1   5
                        size_t i2 = i * 4 + 2; //2   6
                        size_t i3 = i * 4 + 3; //3   7

                        size_t loop0 = i * 8;     //0   8
                        size_t loop1 = i * 8 + 1; //1   9
                        size_t loop2 = i * 8 + 2; //2   10
                        size_t loop3 = i * 8 + 3; //3   11

                        size_t loop4 = i * 8 + 4; //4
                        size_t loop5 = i * 8 + 5; //5
                        size_t loop6 = i * 8 + 6; //6
                        size_t loop7 = i * 8 + 7; //7

                        auto apos = inprim->verts[(inprim->loops[i0])];
                        auto bpos = inprim->verts[(inprim->loops[i1])];
                        auto ab_center_pos = (apos + bpos) / 2; //ab中点坐标

                        auto cpos = inprim->verts[(inprim->loops[i2])];
                        auto dpos = inprim->verts[(inprim->loops[i3])];
                        auto dc_center_pos = (cpos + dpos) / 2; //cd中点坐标

                        //----------------------------------------------------------------------------------------------
                        //求ab两点之间的距离
                        float it1_x = apos.at(0);
                        float it1_y = apos.at(1);
                        float it1_z = apos.at(2);

                        float it2_x = bpos.at(0);
                        float it2_y = bpos.at(1);
                        float it2_z = bpos.at(2);

                        float ab_dis = ((it1_x - it2_x) * (it1_x - it2_x)) + ((it1_y - it2_y) * (it1_y - it2_y)) +
                                       ((it1_z - it2_z) * (it1_z - it2_z));

                        float ab_distance = 0;
                        ab_distance = sqrt(ab_dis); //求得ab两点之间的距离
                        float half_ab_distance = 0;
                        half_ab_distance = ab_distance / 2; //求得ab两点之间距离的半值
                        /*····························································································*/
                        //求cd两点之间的距离
                        float it3_x = cpos.at(0);
                        float it3_y = cpos.at(1);
                        float it3_z = cpos.at(2);

                        float it4_x = dpos.at(0);
                        float it4_y = dpos.at(1);
                        float it4_z = dpos.at(2);

                        float dc_dis = ((it4_x - it3_x) * (it4_x - it3_x)) + ((it4_y - it3_y) * (it4_y - it3_y)) +
                                       ((it4_z - it3_z) * (it4_z - it3_z));

                        float dc_distance = 0;
                        dc_distance = sqrt(dc_dis); //求得cd两点之间的距离
                        float half_dc_distance = 0;
                        half_dc_distance = dc_distance / 2; //求得cd两点之间距离的半值

                        //----------------------------------------------------------------------------------------------
                        //获取偏移方向
                        auto vector_ab = bpos - apos;                       //ab向量
                        auto vector_normalize_ad = vector_ab / ab_distance; //ab向量归一化

                        auto vector_dc = cpos - dpos;                       //dc向量
                        auto vector_normalize_dc = vector_dc / dc_distance; //dc向量归一化
                        //----------------------------------------------------------------------------------------------

                        std::clamp(first_edge_minoffset, float(-0.99), float(0));
                        std::clamp(first_edge_maxoffset, float(0), float(0.99));

                        std::clamp(second_edge_minoffset, float(-0.99), float(0));
                        std::clamp(second_edge_maxoffset, float(0), float(0.99));

                        //float mindis = dis * first_edge_minoffset;
                        //float maxdis = dis * first_edge_maxoffset;

                        //----------------------------------------------------------------------------------------------

                        if (!first_second_same) //不相同
                        {
                            //first_edge：ab边随机偏移值
                            unsigned int f_seed = 0;
                            if (first_seed == -1) {
                                f_seed = std::random_device{}();
                            } else {
                                f_seed = first_seed + i;
                            }
                            std::mt19937 gen(f_seed);
                            std::uniform_real_distribution<float> uni(first_edge_minoffset, first_edge_maxoffset);
                            float ab_value =
                                uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                            //second_edge：dc边随机偏移值
                            unsigned int s_seed = 0;
                            if (second_seed == -1) {
                                s_seed = std::random_device{}();
                            } else {
                                s_seed = second_seed + i;
                            }
                            std::mt19937 gen1(s_seed);
                            std::uniform_real_distribution<float> uni1(second_edge_minoffset, second_edge_maxoffset);
                            float dc_value =
                                uni1(gen1); //在second_edge_minoffset ~ second_edge_maxoffset之间得到一个随机值。

                            auto ab_offset = ab_value * half_ab_distance * vector_normalize_ad;
                            auto dc_offset = dc_value * half_dc_distance * vector_normalize_dc;

                            ab_center_pos += ab_offset;
                            dc_center_pos += dc_offset;
                        } else //相同
                        {
                            int r_seed = 0;
                            if (same_seed == -1) {
                                r_seed = std::random_device{}();
                            } else {
                                r_seed = same_seed + i;
                            }
                            std::mt19937 gen(r_seed);
                            std::uniform_real_distribution<float> uni(minoffset, maxoffset);
                            float value = uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                            auto ab_offset = value * half_ab_distance * vector_normalize_ad;
                            auto dc_offset = value * half_dc_distance * vector_normalize_dc;

                            ab_center_pos += ab_offset;
                            dc_center_pos += dc_offset;
                        }

                        //----------------------------------------------------------------------------------------------

                        outprim->verts.push_back(apos);
                        outprim->verts.push_back(ab_center_pos);
                        outprim->verts.push_back(dc_center_pos);
                        outprim->verts.push_back(dpos);

                        outprim->verts.push_back(ab_center_pos);
                        outprim->verts.push_back(bpos);
                        outprim->verts.push_back(cpos);
                        outprim->verts.push_back(dc_center_pos);

                        outprim->loops.push_back(loop0);
                        outprim->loops.push_back(loop1);
                        outprim->loops.push_back(loop2);
                        outprim->loops.push_back(loop3);

                        outprim->loops.push_back(loop4);
                        outprim->loops.push_back(loop5);
                        outprim->loops.push_back(loop6);
                        outprim->loops.push_back(loop7);

                        outprim->polys.push_back({ss * 4, 4});
                        ++ss;
                        outprim->polys.push_back({ss * 4, 4});
                        ++ss;
                    }
                }

                if (value == 1) {
                    int ss = 0;

                    for (size_t o = 0; o < inprim->polys.size(); o++) //遍历每一个四边面
                    {
                        size_t i0 = o * 4 + 0; //0   2
                        size_t i1 = o * 4 + 1; //1   3
                        size_t i2 = o * 4 + 2; //2   4
                        size_t i3 = o * 4 + 3; //3   5

                        size_t loop0 = o * 8;     //0
                        size_t loop1 = o * 8 + 1; //1
                        size_t loop2 = o * 8 + 2; //2
                        size_t loop3 = o * 8 + 3; //3

                        size_t loop4 = o * 8 + 4; //0
                        size_t loop5 = o * 8 + 5; //1
                        size_t loop6 = o * 8 + 6; //2
                        size_t loop7 = o * 8 + 7; //3

                        auto apos = inprim->verts[(inprim->loops[i0])];
                        auto dpos = inprim->verts[(inprim->loops[i3])];
                        auto ad_center_pos = (apos + dpos) / 2; //ad中点坐标

                        auto bpos = inprim->verts[(inprim->loops[i1])];
                        auto cpos = inprim->verts[(inprim->loops[i2])];
                        auto bc_center_pos = (bpos + cpos) / 2; //bc中点坐标

                        //----------------------------------------------------------------------------------------------
                        //ad两点之间求距离
                        float it1_x = apos.at(0);
                        float it1_y = apos.at(1);
                        float it1_z = apos.at(2);

                        float it2_x = dpos.at(0);
                        float it2_y = dpos.at(1);
                        float it2_z = dpos.at(2);

                        float ad_dis1 = ((it1_x - it2_x) * (it1_x - it2_x)) + ((it1_y - it2_y) * (it1_y - it2_y)) +
                                        ((it1_z - it2_z) * (it1_z - it2_z));

                        float ad_distance = 0;
                        ad_distance = sqrt(ad_dis1); //求得ad两点之间的距离
                        float half_ad_distance = 0;
                        half_ad_distance = ad_distance / 2; //求得ad两点之间距离的半值

                        /*····························································································*/

                        //bc两点之间求距离
                        float it3_x = bpos.at(0);
                        float it3_y = bpos.at(1);
                        float it3_z = bpos.at(2);

                        float it4_x = cpos.at(0);
                        float it4_y = cpos.at(1);
                        float it4_z = cpos.at(2);

                        float bc_dis1 = ((it3_x - it4_x) * (it3_x - it4_x)) + ((it3_y - it4_y) * (it3_y - it4_y)) +
                                        ((it3_z - it4_z) * (it3_z - it4_z));

                        /*····························································································*/

                        float bc_distance = 0;
                        bc_distance = sqrt(bc_dis1); //求得bc两点之间的距离
                        float half_bc_distance = 0;
                        half_bc_distance = bc_distance / 2; //求得bc两点之间距离的半值

                        //----------------------------------------------------------------------------------------------
                        //获取偏移方向
                        auto vector_ad = dpos - apos;                       //ad向量
                        auto vector_normalize_ad = vector_ad / ad_distance; //ad向量归一化

                        auto vector_bc = cpos - bpos;                       //bc向量
                        auto vector_normalize_bc = vector_bc / bc_distance; //bc向量归一化
                        //----------------------------------------------------------------------------------------------

                        std::clamp(first_edge_minoffset, float(-0.99), float(0));
                        std::clamp(first_edge_maxoffset, float(0), float(0.99));

                        std::clamp(second_edge_minoffset, float(-0.99), float(0));
                        std::clamp(second_edge_maxoffset, float(0), float(0.99));

                        //----------------------------------------------------------------------------------------------

                        if (!first_second_same) //不相同
                        {
                            //first_edge：ad边随机偏移值
                            unsigned int f_seed = 0;
                            if (first_seed == -1) {
                                f_seed = std::random_device{}();
                            } else {
                                f_seed = first_seed + o;
                            }
                            std::mt19937 gen(f_seed);
                            std::uniform_real_distribution<float> uni(first_edge_minoffset, first_edge_maxoffset);
                            float ad_value =
                                uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                            //second_edge：bc边随机偏移值
                            unsigned int s_seed = 0;
                            if (second_seed == -1) {
                                s_seed = std::random_device{}();
                            } else {
                                s_seed = second_seed + o;
                            }
                            std::mt19937 gen1(s_seed);
                            std::uniform_real_distribution<float> uni1(second_edge_minoffset, second_edge_maxoffset);
                            float bc_value =
                                uni1(gen1); //在second_edge_minoffset ~ second_edge_maxoffset之间得到一个随机值。

                            auto ad_offset = ad_value * half_ad_distance * vector_normalize_ad;
                            auto bc_offset = bc_value * half_bc_distance * vector_normalize_bc;

                            ad_center_pos += ad_offset;
                            bc_center_pos += bc_offset;
                        } else //相同
                        {
                            unsigned int seed = 0;
                            if (same_seed == -1) {
                                seed = std::random_device{}();
                            } else {
                                seed = same_seed + o;
                            }
                            std::mt19937 gen(seed);
                            std::uniform_real_distribution<float> uni(minoffset, maxoffset);
                            float value = uni(gen); //在minoffset ~ maxoffset之间得到一个随机值。

                            auto ab_offset = value * half_ad_distance * vector_normalize_ad;
                            auto bc_offset = value * half_bc_distance * vector_normalize_bc;

                            ad_center_pos += ab_offset;
                            bc_center_pos += bc_offset;
                        }

                        //----------------------------------------------------------------------------------------------
                        outprim->verts.push_back(apos);
                        outprim->verts.push_back(bpos);
                        outprim->verts.push_back(bc_center_pos);
                        outprim->verts.push_back(ad_center_pos);

                        outprim->verts.push_back(ad_center_pos);
                        outprim->verts.push_back(bc_center_pos);
                        outprim->verts.push_back(cpos);
                        outprim->verts.push_back(dpos);

                        outprim->loops.push_back(loop0);
                        outprim->loops.push_back(loop1);
                        outprim->loops.push_back(loop2);
                        outprim->loops.push_back(loop3);

                        outprim->loops.push_back(loop4);
                        outprim->loops.push_back(loop5);
                        outprim->loops.push_back(loop6);
                        outprim->loops.push_back(loop7);

                        outprim->polys.push_back({ss * 4, 4});
                        ++ss;
                        outprim->polys.push_back({ss * 4, 4});
                        ++ss;
                    }
                }

                //*inprim = *outprim;

                inprim->assign(outprim.get());

                if (rc_num < (num - 1)) {
                    outprim->verts.clear();
                    outprim->loops.clear();
                    outprim->polys.clear();
                }
            }
        } else {
            if (rcrc) {
                for (size_t rc_num = 0; rc_num < num; ++rc_num) {
                    if (rc_num % 2 == 0) {
                        int ss = 0;

                        for (size_t i = 0; i < inprim->polys.size(); i++) //遍历每一个四边面
                        {
                            size_t i0 = i * 4 + 0; //0   4
                            size_t i1 = i * 4 + 1; //1   5
                            size_t i2 = i * 4 + 2; //2   6
                            size_t i3 = i * 4 + 3; //3   7

                            size_t loop0 = i * 8;     //0   8
                            size_t loop1 = i * 8 + 1; //1   9
                            size_t loop2 = i * 8 + 2; //2   10
                            size_t loop3 = i * 8 + 3; //3   11

                            size_t loop4 = i * 8 + 4; //4
                            size_t loop5 = i * 8 + 5; //5
                            size_t loop6 = i * 8 + 6; //6
                            size_t loop7 = i * 8 + 7; //7

                            auto apos = inprim->verts[(inprim->loops[i0])];
                            auto bpos = inprim->verts[(inprim->loops[i1])];
                            auto ab_center_pos = (apos + bpos) / 2; //ab中点坐标

                            auto cpos = inprim->verts[(inprim->loops[i2])];
                            auto dpos = inprim->verts[(inprim->loops[i3])];
                            auto dc_center_pos = (cpos + dpos) / 2; //cd中点坐标

                            //------------------------------------------------------------------------------------------
                            //求ab两点之间的距离
                            float it1_x = apos.at(0);
                            float it1_y = apos.at(1);
                            float it1_z = apos.at(2);

                            float it2_x = bpos.at(0);
                            float it2_y = bpos.at(1);
                            float it2_z = bpos.at(2);

                            float ab_dis = ((it1_x - it2_x) * (it1_x - it2_x)) + ((it1_y - it2_y) * (it1_y - it2_y)) +
                                           ((it1_z - it2_z) * (it1_z - it2_z));

                            float ab_distance = 0;
                            ab_distance = sqrt(ab_dis); //求得ab两点之间的距离
                            float half_ab_distance = 0;
                            half_ab_distance = ab_distance / 2; //求得ab两点之间距离的半值
                            /*························································································*/
                            //求cd两点之间的距离
                            float it3_x = cpos.at(0);
                            float it3_y = cpos.at(1);
                            float it3_z = cpos.at(2);

                            float it4_x = dpos.at(0);
                            float it4_y = dpos.at(1);
                            float it4_z = dpos.at(2);

                            float dc_dis = ((it4_x - it3_x) * (it4_x - it3_x)) + ((it4_y - it3_y) * (it4_y - it3_y)) +
                                           ((it4_z - it3_z) * (it4_z - it3_z));

                            float dc_distance = 0;
                            dc_distance = sqrt(dc_dis); //求得cd两点之间的距离
                            float half_dc_distance = 0;
                            half_dc_distance = dc_distance / 2; //求得cd两点之间距离的半值

                            //------------------------------------------------------------------------------------------
                            //获取偏移方向
                            auto vector_ab = bpos - apos;                       //ab向量
                            auto vector_normalize_ad = vector_ab / ab_distance; //ab向量归一化

                            auto vector_dc = cpos - dpos;                       //dc向量
                            auto vector_normalize_dc = vector_dc / dc_distance; //dc向量归一化
                            //------------------------------------------------------------------------------------------

                            std::clamp(first_edge_minoffset, float(-0.99), float(0));
                            std::clamp(first_edge_maxoffset, float(0), float(0.99));

                            std::clamp(second_edge_minoffset, float(-0.99), float(0));
                            std::clamp(second_edge_maxoffset, float(0), float(0.99));

                            //float mindis = dis * first_edge_minoffset;
                            //float maxdis = dis * first_edge_maxoffset;

                            //------------------------------------------------------------------------------------------

                            if (!first_second_same) //不相同
                            {
                                //first_edge：ab边随机偏移值
                                unsigned int f_seed = 0;
                                if (first_seed == -1) {
                                    f_seed = std::random_device{}();
                                } else {
                                    f_seed = first_seed + i;
                                }
                                std::mt19937 gen(f_seed);
                                std::uniform_real_distribution<float> uni(first_edge_minoffset, first_edge_maxoffset);
                                float ab_value =
                                    uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                                //first_edge：dc边随机偏移值
                                unsigned int s_seed = 0;
                                if (second_seed == -1) {
                                    s_seed = std::random_device{}();
                                } else {
                                    s_seed = second_seed + i;
                                }
                                std::mt19937 gen1(s_seed);
                                std::uniform_real_distribution<float> uni1(second_edge_minoffset,
                                                                           second_edge_maxoffset);
                                float dc_value =
                                    uni1(gen1); //在second_edge_minoffset ~ second_edge_maxoffset之间得到一个随机值。

                                auto ab_offset = ab_value * half_ab_distance * vector_normalize_ad;
                                auto dc_offset = dc_value * half_dc_distance * vector_normalize_dc;

                                ab_center_pos += ab_offset;
                                dc_center_pos += dc_offset;
                            } else //相同
                            {
                                unsigned int seed = 0;
                                if (same_seed == -1) {
                                    seed = std::random_device{}();
                                } else {
                                    seed = same_seed + i;
                                }
                                std::mt19937 gen(seed);
                                std::uniform_real_distribution<float> uni(minoffset, maxoffset);
                                float value = uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                                auto ab_offset = value * half_ab_distance * vector_normalize_ad;
                                auto dc_offset = value * half_dc_distance * vector_normalize_dc;

                                ab_center_pos += ab_offset;
                                dc_center_pos += dc_offset;
                            }

                            //------------------------------------------------------------------------------------------

                            outprim->verts.push_back(apos);
                            outprim->verts.push_back(ab_center_pos);
                            outprim->verts.push_back(dc_center_pos);
                            outprim->verts.push_back(dpos);

                            outprim->verts.push_back(ab_center_pos);
                            outprim->verts.push_back(bpos);
                            outprim->verts.push_back(cpos);
                            outprim->verts.push_back(dc_center_pos);

                            outprim->loops.push_back(loop0);
                            outprim->loops.push_back(loop1);
                            outprim->loops.push_back(loop2);
                            outprim->loops.push_back(loop3);

                            outprim->loops.push_back(loop4);
                            outprim->loops.push_back(loop5);
                            outprim->loops.push_back(loop6);
                            outprim->loops.push_back(loop7);

                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                        }
                    }

                    if (rc_num % 2 == 1) {
                        int ss = 0;

                        for (size_t o = 0; o < inprim->polys.size(); o++) //遍历每一个四边面
                        {
                            size_t i0 = o * 4 + 0; //0   2
                            size_t i1 = o * 4 + 1; //1   3
                            size_t i2 = o * 4 + 2; //2   4
                            size_t i3 = o * 4 + 3; //3   5

                            size_t loop0 = o * 8;     //0
                            size_t loop1 = o * 8 + 1; //1
                            size_t loop2 = o * 8 + 2; //2
                            size_t loop3 = o * 8 + 3; //3

                            size_t loop4 = o * 8 + 4; //0
                            size_t loop5 = o * 8 + 5; //1
                            size_t loop6 = o * 8 + 6; //2
                            size_t loop7 = o * 8 + 7; //3

                            auto apos = inprim->verts[(inprim->loops[i0])];
                            auto dpos = inprim->verts[(inprim->loops[i3])];
                            auto ad_center_pos = (apos + dpos) / 2; //ad中点坐标

                            auto bpos = inprim->verts[(inprim->loops[i1])];
                            auto cpos = inprim->verts[(inprim->loops[i2])];
                            auto bc_center_pos = (bpos + cpos) / 2; //bc中点坐标

                            //------------------------------------------------------------------------------------------
                            //ad两点之间求距离
                            float it1_x = apos.at(0);
                            float it1_y = apos.at(1);
                            float it1_z = apos.at(2);

                            float it2_x = dpos.at(0);
                            float it2_y = dpos.at(1);
                            float it2_z = dpos.at(2);

                            float ad_dis1 = ((it1_x - it2_x) * (it1_x - it2_x)) + ((it1_y - it2_y) * (it1_y - it2_y)) +
                                            ((it1_z - it2_z) * (it1_z - it2_z));

                            float ad_distance = 0;
                            ad_distance = sqrt(ad_dis1); //求得ad两点之间的距离
                            float half_ad_distance = 0;
                            half_ad_distance = ad_distance / 2; //求得ad两点之间距离的半值

                            /*························································································*/

                            //bc两点之间求距离
                            float it3_x = bpos.at(0);
                            float it3_y = bpos.at(1);
                            float it3_z = bpos.at(2);

                            float it4_x = cpos.at(0);
                            float it4_y = cpos.at(1);
                            float it4_z = cpos.at(2);

                            float bc_dis1 = ((it3_x - it4_x) * (it3_x - it4_x)) + ((it3_y - it4_y) * (it3_y - it4_y)) +
                                            ((it3_z - it4_z) * (it3_z - it4_z));

                            /*························································································*/

                            float bc_distance = 0;
                            bc_distance = sqrt(bc_dis1); //求得bc两点之间的距离
                            float half_bc_distance = 0;
                            half_bc_distance = bc_distance / 2; //求得bc两点之间距离的半值

                            //------------------------------------------------------------------------------------------
                            //获取偏移方向
                            auto vector_ad = dpos - apos;                       //ad向量
                            auto vector_normalize_ad = vector_ad / ad_distance; //ad向量归一化

                            auto vector_bc = cpos - bpos;                       //bc向量
                            auto vector_normalize_bc = vector_bc / bc_distance; //bc向量归一化
                            //------------------------------------------------------------------------------------------

                            std::clamp(first_edge_minoffset, float(-0.99), float(0));
                            std::clamp(first_edge_maxoffset, float(0), float(0.99));

                            std::clamp(second_edge_minoffset, float(-0.99), float(0));
                            std::clamp(second_edge_maxoffset, float(0), float(0.99));

                            //float mindis = dis * minoffset;
                            //float maxdis = dis * maxoffset;

                            //------------------------------------------------------------------------------------------

                            if (!first_second_same) //不相同
                            {
                                //first_edge：ad边随机偏移值
                                unsigned int f_seed = 0;
                                if (first_seed == -1) {
                                    f_seed = std::random_device{}();
                                } else {
                                    f_seed = first_seed + o;
                                }
                                std::mt19937 gen(f_seed);
                                std::uniform_real_distribution<float> uni(first_edge_minoffset, first_edge_maxoffset);
                                float ad_value = uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                                //first_edge：bc边随机偏移值
                                unsigned int s_seed = 0;
                                if (second_seed == -1) {
                                    s_seed = std::random_device{}();
                                } else {
                                    s_seed = second_seed + o;
                                }
                                std::mt19937 gen1(s_seed);
                                std::uniform_real_distribution<float> uni1(second_edge_minoffset,
                                                                           second_edge_maxoffset);
                                float bc_value = uni1(gen1); //在second_edge_minoffset ~ second_edge_maxoffset之间得到一个随机值。

                                auto ad_offset = ad_value * half_ad_distance * vector_normalize_ad;
                                auto bc_offset = bc_value * half_bc_distance * vector_normalize_bc;

                                ad_center_pos += ad_offset;
                                bc_center_pos += bc_offset;
                            } else //相同
                            {
                                unsigned int seed = 0;
                                if (same_seed == -1) {
                                    seed = std::random_device{}();
                                } else {
                                    seed = same_seed + o;
                                }
                                std::mt19937 gen(seed);
                                std::uniform_real_distribution<float> uni(minoffset, maxoffset);
                                float value = uni(gen); //在minoffset ~ maxoffset之间得到一个随机值。

                                auto ab_offset = value * half_ad_distance * vector_normalize_ad;
                                auto bc_offset = value * half_bc_distance * vector_normalize_bc;

                                ad_center_pos += ab_offset;
                                bc_center_pos += bc_offset;
                            }

                            //------------------------------------------------------------------------------------------

                            outprim->verts.push_back(apos);
                            outprim->verts.push_back(bpos);
                            outprim->verts.push_back(bc_center_pos);
                            outprim->verts.push_back(ad_center_pos);

                            outprim->verts.push_back(ad_center_pos);
                            outprim->verts.push_back(bc_center_pos);
                            outprim->verts.push_back(cpos);
                            outprim->verts.push_back(dpos);

                            outprim->loops.push_back(loop0);
                            outprim->loops.push_back(loop1);
                            outprim->loops.push_back(loop2);
                            outprim->loops.push_back(loop3);

                            outprim->loops.push_back(loop4);
                            outprim->loops.push_back(loop5);
                            outprim->loops.push_back(loop6);
                            outprim->loops.push_back(loop7);

                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                        }
                    }

                    inprim->assign(outprim.get());

                    if (rc_num < (num - 1)) {
                        outprim->verts.clear();
                        outprim->loops.clear();
                        outprim->polys.clear();
                    }
                }
            }

            else {
                for (int rc_num = 0; rc_num < num; ++rc_num) {
                    if (roworcolumns) {
                        int ss = 0;

                        for (size_t i = 0; i < inprim->polys.size(); i++) //遍历每一个四边面
                        {
                            size_t i0 = i * 4 + 0; //0   4
                            size_t i1 = i * 4 + 1; //1   5
                            size_t i2 = i * 4 + 2; //2   6
                            size_t i3 = i * 4 + 3; //3   7

                            size_t loop0 = i * 8;     //0   8
                            size_t loop1 = i * 8 + 1; //1   9
                            size_t loop2 = i * 8 + 2; //2   10
                            size_t loop3 = i * 8 + 3; //3   11

                            size_t loop4 = i * 8 + 4; //4
                            size_t loop5 = i * 8 + 5; //5
                            size_t loop6 = i * 8 + 6; //6
                            size_t loop7 = i * 8 + 7; //7

                            auto apos = inprim->verts[(inprim->loops[i0])];
                            auto bpos = inprim->verts[(inprim->loops[i1])];
                            auto ab_center_pos = (apos + bpos) / 2; //ab中点坐标

                            auto cpos = inprim->verts[(inprim->loops[i2])];
                            auto dpos = inprim->verts[(inprim->loops[i3])];
                            auto dc_center_pos = (cpos + dpos) / 2; //cd中点坐标

                            //------------------------------------------------------------------------------------------
                            //求ab两点之间的距离
                            float it1_x = apos.at(0);
                            float it1_y = apos.at(1);
                            float it1_z = apos.at(2);

                            float it2_x = bpos.at(0);
                            float it2_y = bpos.at(1);
                            float it2_z = bpos.at(2);

                            float ab_dis = ((it1_x - it2_x) * (it1_x - it2_x)) + ((it1_y - it2_y) * (it1_y - it2_y)) +
                                           ((it1_z - it2_z) * (it1_z - it2_z));

                            float ab_distance = 0;
                            ab_distance = sqrt(ab_dis); //求得ab两点之间的距离
                            float half_ab_distance = 0;
                            half_ab_distance = ab_distance / 2; //求得ab两点之间距离的半值
                            /*························································································*/
                            //求cd两点之间的距离
                            float it3_x = cpos.at(0);
                            float it3_y = cpos.at(1);
                            float it3_z = cpos.at(2);

                            float it4_x = dpos.at(0);
                            float it4_y = dpos.at(1);
                            float it4_z = dpos.at(2);

                            float dc_dis = ((it4_x - it3_x) * (it4_x - it3_x)) + ((it4_y - it3_y) * (it4_y - it3_y)) +
                                           ((it4_z - it3_z) * (it4_z - it3_z));

                            float dc_distance = 0;
                            dc_distance = sqrt(dc_dis); //求得cd两点之间的距离
                            float half_dc_distance = 0;
                            half_dc_distance = dc_distance / 2; //求得cd两点之间距离的半值

                            //------------------------------------------------------------------------------------------
                            //获取偏移方向
                            auto vector_ab = bpos - apos;                       //ab向量
                            auto vector_normalize_ad = vector_ab / ab_distance; //ab向量归一化

                            auto vector_dc = cpos - dpos;                       //dc向量
                            auto vector_normalize_dc = vector_dc / dc_distance; //dc向量归一化
                            //------------------------------------------------------------------------------------------

                            std::clamp(first_edge_minoffset, float(-0.99), float(0));
                            std::clamp(first_edge_maxoffset, float(0), float(0.99));

                            std::clamp(second_edge_minoffset, float(-0.99), float(0));
                            std::clamp(second_edge_maxoffset, float(0), float(0.99));

                            //float mindis = dis * first_edge_minoffset;
                            //float maxdis = dis * first_edge_maxoffset;

                            //------------------------------------------------------------------------------------------

                            if (!first_second_same) //不相同
                            {
                                //first_edge：ab边随机偏移值
                                unsigned int f_seed = 0;
                                if (first_seed == -1) {
                                    f_seed = std::random_device{}();
                                } else {
                                    f_seed = first_seed + i;
                                }
                                std::mt19937 gen(f_seed);
                                std::uniform_real_distribution<float> uni(first_edge_minoffset, first_edge_maxoffset);
                                float ab_value = uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                                //first_edge：dc边随机偏移值
                                unsigned int s_seed = 0;
                                if (second_seed == -1) {
                                    s_seed = std::random_device{}();
                                } else {
                                    s_seed = second_seed + i;
                                }
                                std::mt19937 gen1(s_seed);
                                std::uniform_real_distribution<float> uni1(second_edge_minoffset,
                                                                           second_edge_maxoffset);
                                float dc_value =  uni1(gen1); //在second_edge_minoffset ~ second_edge_maxoffset之间得到一个随机值。

                                auto ab_offset = ab_value * half_ab_distance * vector_normalize_ad;
                                auto dc_offset = dc_value * half_dc_distance * vector_normalize_dc;

                                ab_center_pos += ab_offset;
                                dc_center_pos += dc_offset;
                            } else //相同
                            {
                                unsigned int seed = 0;
                                if (same_seed == -1) {
                                    seed = std::random_device{}();
                                } else {
                                    seed = same_seed + i;
                                }
                                std::mt19937 gen(seed);
                                std::uniform_real_distribution<float> uni(minoffset, maxoffset);
                                float value = uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                                auto ab_offset = value * half_ab_distance * vector_normalize_ad;
                                auto dc_offset = value * half_dc_distance * vector_normalize_dc;

                                ab_center_pos += ab_offset;
                                dc_center_pos += dc_offset;
                            }

                            //------------------------------------------------------------------------------------------

                            outprim->verts.push_back(apos);
                            outprim->verts.push_back(ab_center_pos);
                            outprim->verts.push_back(dc_center_pos);
                            outprim->verts.push_back(dpos);

                            outprim->verts.push_back(ab_center_pos);
                            outprim->verts.push_back(bpos);
                            outprim->verts.push_back(cpos);
                            outprim->verts.push_back(dc_center_pos);

                            outprim->loops.push_back(loop0);
                            outprim->loops.push_back(loop1);
                            outprim->loops.push_back(loop2);
                            outprim->loops.push_back(loop3);

                            outprim->loops.push_back(loop4);
                            outprim->loops.push_back(loop5);
                            outprim->loops.push_back(loop6);
                            outprim->loops.push_back(loop7);

                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                        }
                    }

                    else {
                        int ss = 0;

                        for (size_t o = 0; o < inprim->polys.size(); o++) //遍历每一个四边面
                        {
                            size_t i0 = o * 4 + 0; //0   2
                            size_t i1 = o * 4 + 1; //1   3
                            size_t i2 = o * 4 + 2; //2   4
                            size_t i3 = o * 4 + 3; //3   5

                            size_t loop0 = o * 8;     //0
                            size_t loop1 = o * 8 + 1; //1
                            size_t loop2 = o * 8 + 2; //2
                            size_t loop3 = o * 8 + 3; //3

                            size_t loop4 = o * 8 + 4; //0
                            size_t loop5 = o * 8 + 5; //1
                            size_t loop6 = o * 8 + 6; //2
                            size_t loop7 = o * 8 + 7; //3

                            auto apos = inprim->verts[(inprim->loops[i0])];
                            auto dpos = inprim->verts[(inprim->loops[i3])];
                            auto ad_center_pos = (apos + dpos) / 2; //ad中点坐标

                            auto bpos = inprim->verts[(inprim->loops[i1])];
                            auto cpos = inprim->verts[(inprim->loops[i2])];
                            auto bc_center_pos = (bpos + cpos) / 2; //bc中点坐标

                            //------------------------------------------------------------------------------------------
                            //ad两点之间求距离
                            float it1_x = apos.at(0);
                            float it1_y = apos.at(1);
                            float it1_z = apos.at(2);

                            float it2_x = dpos.at(0);
                            float it2_y = dpos.at(1);
                            float it2_z = dpos.at(2);

                            float ad_dis1 = ((it1_x - it2_x) * (it1_x - it2_x)) + ((it1_y - it2_y) * (it1_y - it2_y)) +
                                            ((it1_z - it2_z) * (it1_z - it2_z));

                            float ad_distance = 0;
                            ad_distance = sqrt(ad_dis1); //求得ad两点之间的距离
                            float half_ad_distance = 0;
                            half_ad_distance = ad_distance / 2; //求得ad两点之间距离的半值

                            /*························································································*/

                            //bc两点之间求距离
                            float it3_x = bpos.at(0);
                            float it3_y = bpos.at(1);
                            float it3_z = bpos.at(2);

                            float it4_x = cpos.at(0);
                            float it4_y = cpos.at(1);
                            float it4_z = cpos.at(2);

                            float bc_dis1 = ((it3_x - it4_x) * (it3_x - it4_x)) + ((it3_y - it4_y) * (it3_y - it4_y)) +
                                            ((it3_z - it4_z) * (it3_z - it4_z));

                            /*························································································*/

                            float bc_distance = 0;
                            bc_distance = sqrt(bc_dis1); //求得bc两点之间的距离
                            float half_bc_distance = 0;
                            half_bc_distance = bc_distance / 2; //求得bc两点之间距离的半值

                            //------------------------------------------------------------------------------------------
                            //获取偏移方向
                            auto vector_ad = dpos - apos;                       //ad向量
                            auto vector_normalize_ad = vector_ad / ad_distance; //ad向量归一化

                            auto vector_bc = cpos - bpos;                       //bc向量
                            auto vector_normalize_bc = vector_bc / bc_distance; //bc向量归一化
                            //------------------------------------------------------------------------------------------

                            std::clamp(first_edge_minoffset, float(-0.99), float(0));
                            std::clamp(first_edge_maxoffset, float(0), float(0.99));

                            std::clamp(second_edge_minoffset, float(-0.99), float(0));
                            std::clamp(second_edge_maxoffset, float(0), float(0.99));

                            //float mindis = dis * minoffset;
                            //float maxdis = dis * maxoffset;

                            //------------------------------------------------------------------------------------------
                            if (!first_second_same) //不相同
                            {
                                //first_edge：ad边随机偏移值
                                unsigned int f_seed = 0;
                                if (first_seed == -1) {
                                    f_seed = std::random_device{}();
                                } else {
                                    f_seed = first_seed + o;
                                }
                                std::mt19937 gen(f_seed);
                                std::uniform_real_distribution<float> uni(first_edge_minoffset, first_edge_maxoffset);
                                float ad_value =  uni(gen); //在first_edge_minoffset ~ first_edge_maxoffset之间得到一个随机值。

                                //first_edge：bc边随机偏移值
                                unsigned int s_seed = 0;
                                if (second_seed == -1) {
                                    s_seed = std::random_device{}();
                                } else {
                                    s_seed = second_seed + o;
                                }
                                std::mt19937 gen1(s_seed);
                                std::uniform_real_distribution<float> uni1(second_edge_minoffset,
                                                                           second_edge_maxoffset);
                                float bc_value = uni1(gen1); //在second_edge_minoffset ~ second_edge_maxoffset之间得到一个随机值。

                                auto ad_offset = ad_value * half_ad_distance * vector_normalize_ad;
                                auto bc_offset = bc_value * half_bc_distance * vector_normalize_bc;

                                ad_center_pos += ad_offset;
                                bc_center_pos += bc_offset;
                            } else //相同
                            {
                                unsigned int seed = 0;
                                if (same_seed == -1) {
                                    seed = std::random_device{}();
                                } else {
                                    seed = same_seed + o;
                                }
                                std::mt19937 gen(seed);
                                std::uniform_real_distribution<float> uni(minoffset, maxoffset);
                                float value = uni(gen); //在minoffset ~ maxoffset之间得到一个随机值。

                                auto ab_offset = value * half_ad_distance * vector_normalize_ad;
                                auto bc_offset = value * half_bc_distance * vector_normalize_bc;

                                ad_center_pos += ab_offset;
                                bc_center_pos += bc_offset;
                            }

                            //------------------------------------------------------------------------------------------

                            outprim->verts.push_back(apos);
                            outprim->verts.push_back(bpos);
                            outprim->verts.push_back(bc_center_pos);
                            outprim->verts.push_back(ad_center_pos);

                            outprim->verts.push_back(ad_center_pos);
                            outprim->verts.push_back(bc_center_pos);
                            outprim->verts.push_back(cpos);
                            outprim->verts.push_back(dpos);

                            outprim->loops.push_back(loop0);
                            outprim->loops.push_back(loop1);
                            outprim->loops.push_back(loop2);
                            outprim->loops.push_back(loop3);

                            outprim->loops.push_back(loop4);
                            outprim->loops.push_back(loop5);
                            outprim->loops.push_back(loop6);
                            outprim->loops.push_back(loop7);

                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                            outprim->polys.push_back({ss * 4, 4});
                            ++ss;
                        }
                    }

                    inprim->assign(outprim.get());

                    if (rc_num < (num - 1)) {
                        outprim->verts.clear();
                        outprim->loops.clear();
                        outprim->polys.clear();
                    }
                }
            }
        }

        if (addattr) {
            auto &tag = outprim->verts.add_attr<float>(tagAttr);

            for (size_t n = 0; n < (tag.size() / 4); ++n) {
                size_t n0 = n * 4 + 0;
                size_t n1 = n * 4 + 1;
                size_t n2 = n * 4 + 2;
                size_t n3 = n * 4 + 3;

                tag[n0] = n;
                tag[n1] = n;
                tag[n2] = n;
                tag[n3] = n;
            }
        }
        set_output("output", std::move(outprim));
    }
};
ZENDEFNODE(PrimQuadsLotSubdivision, {{
                               /* inputs: */
                               {gParamType_Primitive, "input_quads_model", "", zeno::Socket_ReadOnly},
                               {gParamType_Int, "num", "1"},
                               {gParamType_Bool, "row_or_columns", "0"},

                               {gParamType_Bool, "random_rc", "0"},
                               {gParamType_Int, "random_seed", "1"},

                               {gParamType_Bool, "rcrc", "1"},

                               {gParamType_Bool, "first_second_same", "1"},
                               {gParamType_Int, "same_seed", "1"},
                               {gParamType_Float, "min_offset", "0"},
                               {gParamType_Float, "max_offset", "0"},

                               {gParamType_Float, "first_edge_minoffset", "0"},
                               {gParamType_Float, "first_edge_maxoffset", "0"},
                               {gParamType_Int, "first_seed", "1"},

                               {gParamType_Float, "second_edge_minoffset", "0"},
                               {gParamType_Float, "second_edge_maxoffset", "0"},
                               {gParamType_Int, "second_seed", "1"},

                               {gParamType_Bool, "add_attr", "0"},
                               {gParamType_String, "tag_attr", "tag"},
                           },

                           {
                               /* outputs: */
                               {gParamType_Primitive, "prim"},
                               {gParamType_Primitive, "output"}
                           },

                           {
                               /* params: */

                           },

                           {
                               /* category: */
                               "primitive",
                           }});
}

