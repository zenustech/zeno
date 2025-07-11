//
// Created by zh on 2025/7/9.
//
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/utils/bit_operations.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include <functional>
#include <set>

namespace zeno {
struct EdgeInfo {
    //bool cutted = false;
    int v0 = 0;
    int v1 = 0;
    char mask = 0;
    int index = -1;
    int edge_array_idx = -1;
};
struct PairHashCompare {
    static size_t hash(const std::pair<int, int>& key) {
        return std::hash<int>()(key.first) ^ std::hash<int>()(key.second);
    }

    static bool equal(const std::pair<int, int>& key1, const std::pair<int, int>& key2) {
        return key1 == key2;
    }
};
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // 简单的哈希组合方式（可能会冲突，但适用于大多数情况）
        return h1 ^ (h2 << 1);
    }
};
int computeHashNumber(PrimitiveObject *prim)
{
    std::vector<int> edge_count;
    edge_count.resize(prim->verts.size());
    for(int i=0;i<prim->polys.size();i++)
    {
        int start = prim->polys[i][0];
        int v0 = prim->loops[start+0];
        int v1 = prim->loops[start+1];
        int v2 = prim->loops[start+2];
        edge_count[v0] += 1;
        edge_count[v1] += 1;
        edge_count[v2] += 1;
    }
    int res = 0;
    for(int i=0;i<edge_count.size();i++)
    {
        res = max(res, edge_count[i] + 1);
    }
    return res;
}
using EdgeInfoMap = tbb::concurrent_hash_map<std::pair<int, int>, EdgeInfo, PairHashCompare>;
bool findEdge(std::pair<int, int> &e, int hashNumber, std::vector<std::vector<EdgeInfo>> &edge_split_mask2,EdgeInfo &ei)
{
    int index = e.first;
    if(edge_split_mask2[index].size()==0)
        return false;
    bool res = false;
    for(int k=0;k<edge_split_mask2[index].size();k++)
    {
        if(edge_split_mask2[index][k].v0==e.first && edge_split_mask2[index][k].v1==e.second )
        {
            res = true;
            ei = edge_split_mask2[index][k];
        }
    }
    return res;
}
void insertEdge(std::pair<int, int> &e, int hashNumber, std::vector<std::vector<EdgeInfo>> &edge_split_mask2,EdgeInfo &ei)
{
    int index = e.first;
    edge_split_mask2[index].push_back(ei);
}
void form_edge(PrimitiveObject *prim, std::unordered_map<std::pair<int, int>, EdgeInfo, PairHash> &edge_split_mask, std::vector<EdgeInfo> &edge_flatten_info,
               std::vector<vec3i> &tri_edges, std::vector<int> &tri_new) {
    std::vector<long> callTimes;

    edge_split_mask.clear();
    edge_flatten_info.resize(0);
    tri_edges.resize(prim->polys.size());
    int h = computeHashNumber(prim);
    std::vector<std::vector<EdgeInfo>> edge_split_mask2;
    edge_split_mask2.resize(prim->verts.size());

    edge_flatten_info.reserve(prim->polys.size()*3);

    for (int i=0;i<prim->polys.size();i++)  {
        if(tri_new[i]==0)
        {
            tri_edges[i] = {-1,-1,-1};
            continue;
        }

        auto [start, _len] = prim->polys[i];
        auto vert_id0 = prim->loops[start + 0];
        auto vert_id1 = prim->loops[start + 1];
        auto vert_id2 = prim->loops[start + 2];
        auto edge_0 = std::pair<int, int>(min(vert_id0, vert_id1), max(vert_id0, vert_id1));
        auto edge_1 = std::pair<int, int>(min(vert_id1, vert_id2), max(vert_id1, vert_id2));
        auto edge_2 = std::pair<int, int>(min(vert_id0, vert_id2), max(vert_id0, vert_id2));
        EdgeInfo edgeInfo0 = {};
        EdgeInfo edgeInfo1 = {};
        EdgeInfo edgeInfo2 = {};



        bool find_e0 = findEdge(edge_0, h, edge_split_mask2, edgeInfo0);
        bool find_e1 = findEdge(edge_1, h, edge_split_mask2, edgeInfo1);
        bool find_e2 = findEdge(edge_2, h, edge_split_mask2, edgeInfo2);


        if(find_e0)
        {
            tri_edges[i][0] = edgeInfo0.edge_array_idx;
        }
        else{
            tri_edges[i][0] = edge_flatten_info.size();
            //edge_split_mask[edge_0].edge_array_idx = edge_flatten_info.size();
            EdgeInfo edgeInfo = {edge_0.first, edge_0.second, 0, -1, int(edge_flatten_info.size())};
            insertEdge(edge_0,h, edge_split_mask2, edgeInfo);
            edge_flatten_info.emplace_back(edgeInfo);
        }


        if(find_e1)
        {
            tri_edges[i][1] = edgeInfo1.edge_array_idx;
        }
        else{
            tri_edges[i][1] = edge_flatten_info.size();
            //edge_split_mask[edge_0].edge_array_idx = edge_flatten_info.size();
            EdgeInfo edgeInfo = {edge_1.first, edge_1.second, 0, -1, int(edge_flatten_info.size())};
            insertEdge(edge_1,h, edge_split_mask2, edgeInfo);
            edge_flatten_info.emplace_back(edgeInfo);
        }


        if(find_e2)
        {
            tri_edges[i][2] = edgeInfo2.edge_array_idx;
        }
        else{
            tri_edges[i][2] = edge_flatten_info.size();
            //edge_split_mask[edge_0].edge_array_idx = edge_flatten_info.size();
            EdgeInfo edgeInfo = {edge_2.first, edge_2.second, 0, -1, int(edge_flatten_info.size())};
            insertEdge(edge_2,h, edge_split_mask2, edgeInfo);
            edge_flatten_info.emplace_back(edgeInfo);
        }
    }
}
bool in_screen(float nx, float ny, glm::vec2 &pix)
{
    return pix[0]>-0.5*nx && pix[0]<0.5*nx && pix[1]>-0.5*ny && pix[1]<0.5*ny;
}
int edge_score2(vec3f v0, vec3f v1, CameraObject* cam, vec2i res, float factor, glm::mat4 const&view, float &t, int iter) {
    auto nx = res[0];
    auto ny = res[1];
    auto ratio = float(nx) / float(ny);
    auto denorm = cam->fnear * tan(cam->fov / 2);
    auto v0_vS = glm::vec3(view * glm::vec4(v0[0], v0[1], v0[2], 1));
    auto v1_vS = glm::vec3(view * glm::vec4(v1[0], v1[1], v1[2], 1));
    //zeno::log_info("view {} v0:{}", v0, bit_cast<vec3f>(v0_vS));
    //zeno::log_info("view {} v1:{}", v1, bit_cast<vec3f>(v1_vS));
    if (v0_vS[2] > -cam->fnear*0.5 && v1_vS[2] > -cam->fnear*0.5) {
        return 0;
    }
    if (v0_vS[2] * v1_vS[2] < 0 && abs(v0_vS[2] * v1_vS[2])>0.0001 && iter<20) {
        //我实在不明白为啥这条这么重要， 你试着把它改掉就不work了， 然而像右边这种情况不应该有关系啊！！
        t = v0_vS[2] / (v0_vS[2] - v1_vS[2]);
        return 1;
    }
    if(v0_vS[2] <=0 && v1_vS[2] <=0) {
        auto _y0 = v0_vS[1] * cam->fnear / (abs(v0_vS[2]) + 0.0001);
        auto _x0 = v0_vS[0] * cam->fnear / (abs(v0_vS[2]) + 0.0001);
        auto _y1 = v1_vS[1] * cam->fnear / (abs(v1_vS[2]) + 0.0001);
        auto _x1 = v1_vS[0] * cam->fnear / (abs(v1_vS[2]) + 0.0001);

        if (abs(_y0) > denorm * 1.1 && abs(_y1) > denorm * 1.1 && _y0 * _y1 > 0) {
            return 0;
        }
        if (abs(_x0) > denorm * ratio * 1.1 && abs(_x1) > denorm * ratio * 1.1 && _x0 * _x1 > 0) {
            return 0;
        }
        auto pix_y0 = _y0 * float(ny) * 0.5 / denorm;
        auto pix_x0 = _x0 * float(ny) * 0.5 / denorm;

        auto pix_y1 = _y1 * float(ny) * 0.5 / denorm;
        auto pix_x1 = _x1 * float(ny) * 0.5 / denorm;

        auto pix_0 = glm::vec2(pix_x0, pix_y0);
        auto pix_1 = glm::vec2(pix_x1, pix_y1);
        if (glm::distance(pix_0, pix_1) > factor) {
            if(in_screen(nx, ny, pix_0) && in_screen(nx, ny, pix_1)) {
                return 2;
            }
            else
                return 1;
        }
    }

    return 0;
}

int rank_edges(PrimitiveObject *prim, std::unordered_map<std::pair<int, int>, EdgeInfo, PairHash> &edge_split_mask, std::vector<EdgeInfo> &edge_flatten_info,
               CameraObject *cam, vec2i res, float factor, std::vector<float> &T, int iter) {
    auto pos = bit_cast<glm::vec3>(cam->pos);
    auto up = bit_cast<glm::vec3>(cam->up);
    auto front = bit_cast<glm::vec3>(cam->view);

    auto view = glm::lookAt(pos, pos + front, up);
    T.resize(edge_flatten_info.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, edge_flatten_info.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if(edge_flatten_info[i].edge_array_idx>=0)
                {
                    std::pair<int, int> key = {edge_flatten_info[i].v0, edge_flatten_info[i].v1};
                    //auto &value = edge_split_mask[key];
                    auto [v0_idx, v1_idx] = key;
                    auto v0 = prim->verts[v0_idx];
                    auto v1 = prim->verts[v1_idx];
                    float t = 0;
                    auto score = edge_score2(v0, v1, cam, res, factor, view, t, iter);
                    T[i] = t;
                    // Update the value directly via the iterator
                    edge_flatten_info[i].mask = score;
                }
            }
    });

    int sum = 0;
    for(int i=0;i<edge_flatten_info.size();i++)
    {
        if(edge_flatten_info[i].mask>0)
        {
            edge_flatten_info[i].index = prim->verts.size() + sum;
            sum++;
        }
    }

    return sum;
}

glm::vec3 projectToNDC(const glm::vec3& P, const glm::mat4& mvp) {
    glm::vec4 clip = mvp * glm::vec4(P, 1.0f);
    return glm::vec3(clip) / clip.w;
}

float findScreenSpaceMidpointT(
    const glm::vec3& A,
    const glm::vec3& B,
    const glm::mat4& mvp,
    float epsilon = 0.1f,
    int maxIterations = 200
) {
    glm::vec3 ndcA = projectToNDC(A, mvp);
    glm::vec3 ndcB = projectToNDC(B, mvp);
    glm::vec3 targetNDC = (ndcA + ndcB) * 0.5f;

    float t_min = 0.0f;
    float t_max = 1.0f;
    float t_mid = 0.5f;

    for (int i = 0; i < maxIterations; ++i) {
        t_mid = (t_min + t_max) * 0.5f;

        glm::vec3 P = A + t_mid * (B - A);

        glm::vec3 ndcP = projectToNDC(P, mvp);

        float current_dist = glm::distance(glm::vec2(ndcP), glm::vec2(targetNDC));

        if (current_dist < epsilon) {
            break;
        }

        if (glm::distance(glm::vec2(ndcP), glm::vec2(ndcA)) <= glm::distance(glm::vec2(ndcP), glm::vec2(ndcB))) {
            t_min = t_mid;
        } else {
            t_max = t_mid;
        }
    }

    return glm::clamp(t_mid, 0.001f, 0.999f);
}

void emit_vert(PrimitiveObject *prim, std::vector<EdgeInfo> &edge_flatten_info, int edgeSum, CameraObject* cam, double nx, double ny, std::vector<float> &T) {
    AttrVector<vec3f> prim_new_verts;
    prim_new_verts.resize(prim->verts.size() + edgeSum);
    prim->verts.forall_attr([&](auto const &key, auto &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        auto &attr = prim_new_verts.add_attr<T>(key);
        std::copy_n(arr.begin(), arr.size(), attr.data());
    });

    auto cview = glm::lookAt(bit_cast<glm::vec3>(cam->pos), bit_cast<glm::vec3>(cam->pos + cam->view), bit_cast<glm::vec3>(cam->up));
    auto cproj = glm::perspective(glm::radians(cam->fov), float(nx/ny), cam->fnear, cam->ffar);
    auto vp = cproj * cview;
    std::vector<float> temp_t(edge_flatten_info.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, edge_flatten_info.size()),
        [&](const tbb::blocked_range<size_t>& r) {

        for (size_t i = r.begin(); i != r.end(); ++i) {
            if (edge_flatten_info[i].mask) {
                auto v0 = prim->verts[edge_flatten_info[i].v0];
                auto v1 = prim->verts[edge_flatten_info[i].v1];
                float t = 0.5;
                if(edge_flatten_info[i].mask==2)
                    t = findScreenSpaceMidpointT(bit_cast<glm::vec3>(v0), bit_cast<glm::vec3>(v1), vp);
                temp_t[i] = t;
            }
        }
    });
    prim->verts.forall_attr([&](auto const &key, auto &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        auto &attr = prim_new_verts.add_attr<T>(key);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, edge_flatten_info.size()),
            [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if (edge_flatten_info[i].mask) {
                    auto v0 = arr[edge_flatten_info[i].v0];
                    auto v1 = arr[edge_flatten_info[i].v1];
                    float t = temp_t[i];
                    auto new_vert = v0 + (v1 - v0) * t;
                    attr[edge_flatten_info[i].index] = new_vert;
                }
            }
        });
    });


    prim->verts = std::move(prim_new_verts);
}
char count_tris(PrimitiveObject *prim, std::vector<EdgeInfo> &edge_flatten_info, std::vector<vec3i> &tri_edges, int idx) {
    char count = 1;
    if(tri_edges[idx][0]==-1)
        return 1;
    if (edge_flatten_info[tri_edges[idx][0]].index > -1) {
        count += 1;
    }
    if (edge_flatten_info[tri_edges[idx][1]].index > -1) {
        count += 1;
    }
    if (edge_flatten_info[tri_edges[idx][2]].index > -1) {
        count += 1;
    }
    return count;
}
void addEdge(int v0, int v1, std::vector<EdgeInfo> &edge_flatten_info, int &e0)
{
    EdgeInfo ei;
    ei.v0 = min(v0,v1);
    ei.v1 = max(v0,v1);
    ei.edge_array_idx = edge_flatten_info.size();
    edge_flatten_info.push_back(ei);

    e0 = edge_flatten_info.size()-1;
}

void split_tris(PrimitiveObject *prim, std::vector<EdgeInfo> &edge_flatten_info, std::vector<vec3i> &tri_edges, int idx, vec3i *v)
{
    auto start = prim->polys[idx][0];
    auto v0 = prim->loops[start + 0];
    auto v1 = prim->loops[start + 1];
    auto v2 = prim->loops[start + 2];
    float l0 = zeno::distance(prim->verts[v0], prim->verts[v1]);
    float l1 = zeno::distance(prim->verts[v1], prim->verts[v2]);
    float l2 = zeno::distance(prim->verts[v0], prim->verts[v2]);

    int v3;
    int v4;
    int v5;
    if(tri_edges[idx][0]>-1) {
        v3 = edge_flatten_info[tri_edges[idx][0]].index;
        v4 = edge_flatten_info[tri_edges[idx][1]].index;
        v5 = edge_flatten_info[tri_edges[idx][2]].index;
    }else{
        v3 = -1; v4 = -1; v5 = -1;
    }

    char flag = 0;
    if (v3 > -1) {
        flag |= 0b100;
    }
    if (v4 > -1) {
        flag |= 0b010;
    }
    if (v5 > -1) {
        flag |= 0b001;
    }
    if (flag == 0) {
        v[0] = {v0, v1, v2};
    }
    else if (flag == 0b111) {
        if(l0>=l1 && l0>=l2) {
            v[0] = {v0, v3, v5};
            v[1] = {v3, v4, v2};
            v[2] = {v3, v1, v4};
            v[3] = {v3, v2, v5};
        }else if(l1>=l0 && l1>=l2)
        {
            v[0] = {v3, v1, v4};
            v[1] = {v0, v3, v4};
            v[2] = {v5, v0, v4};
            v[3] = {v2, v5, v4};
        }else
        {
            v[0] = {v0, v3, v5};
            v[1] = {v3, v1, v5};
            v[2] = {v1, v4, v5};
            v[3] = {v4, v2, v5};
        }
    }
    else if (flag == 0b110) {
        v[0] = {v0, v3, v4};
        v[1] = {v3, v1, v4};
        v[2] = {v0, v4, v2};
    }
    else if (flag == 0b101) {
        v[0] = {v0, v3, v5};
        v[1] = {v3, v1, v5};
        v[2] = {v5, v1, v2};
    }
    else if (flag == 0b011) {
        v[0] = {v0, v1, v5};
        v[1] = {v1, v4, v5};
        v[2] = {v4, v2, v5};
    }
    else if (flag == 0b100) {
        v[0] = {v0, v3, v2};
        v[1] = {v3, v1, v2};
    }
    else if (flag == 0b010) {
        v[0] = {v0, v1, v4};
        v[1] = {v4, v2, v0};
    }
    else if (flag == 0b001) {
        v[0] = {v0, v1, v5};
        v[1] = {v5, v1, v2};
    }
}
void emit_tris(PrimitiveObject *prim, std::vector<EdgeInfo> &edge_flatten_info, std::vector<vec3i> &tri_edges, std::vector<int> &tri_new) {
    std::vector<zeno::vec3i> new_tris;

    std::vector<int> tri_sum;
    std::vector<int> tri_count;
    tri_count.resize(prim->polys.size());
    tri_sum.resize(prim->polys.size() + 1);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, prim->polys.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                tri_count[i] = count_tris(prim, edge_flatten_info, tri_edges, i);
            }
    });
    //todo using parallel scan
    tri_sum[0] = 0;
    for(int i=1;i<tri_sum.size();i++)
    {
        tri_sum[i] = tri_sum[i-1]+tri_count[i-1];
    }
    int total_num = tri_sum.back();
    new_tris.resize(total_num);
    tri_new.assign(total_num, 0);
    //begin tri emit
     tbb::parallel_for(tbb::blocked_range<size_t>(0, tri_count.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                int emit_start = tri_sum[i];
                int emit_count = tri_count[i];
                vec3i v[4];
                split_tris(prim, edge_flatten_info, tri_edges, i, v);
                for (int j = 0; j < emit_count; j++) {
                    new_tris[emit_start + j] = v[j];
                    if(emit_count > 1) {
                        tri_new[emit_start + j] = 1;
                    }
                }
            }
        });
    //change original tris to new_tris
    prim->loops.clear();
    prim->loops.resize(new_tris.size() * 3);
    std::copy_n((int*)new_tris.data(), prim->loops.size(), prim->loops.data());
    prim->polys.clear();
    prim->polys.resize(new_tris.size());
    for (auto i = 0; i < prim->polys.size(); i++) {
        prim->polys[i] = { 3 * i, 3 };
    }
}
struct PrimDice : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        primTriangulateIntoPolys(prim.get());
        auto camera = get_input2<CameraObject>("camera");
        auto width = get_input2<int>("width");
        auto height = get_input2<int>("height");
        auto maxIterNum = get_input2<int>("maxIterNum");
        auto factor = get_input2<float>("factor");
        int iter_count = 0;
        std::unordered_map<std::pair<int, int>, EdgeInfo, PairHash> edge_split_mask;
        EdgeInfoMap concurrent_edges;
        std::vector<EdgeInfo> e_info_vector;
        std::vector<vec3i> tri_edges;
        std::vector<int> tri_new;
        tri_new.assign(prim->polys.size(),1);
        std::vector<float> T;
        while (iter_count < maxIterNum) {
//            zeno::log_info("======={}==========", iter_count);
            form_edge(prim.get(), edge_split_mask, e_info_vector, tri_edges, tri_new);
            int new_vert_count = rank_edges(prim.get(), edge_split_mask, e_info_vector, camera.get(), {width, height}, factor,T, iter_count);

            if (new_vert_count == 0) {
                break;
            }
            emit_vert(prim.get(), e_info_vector,new_vert_count, camera.get(), width, height,T);
            emit_tris(prim.get(), e_info_vector,tri_edges, tri_new);

            iter_count += 1;
        }
        if (iter_count >= maxIterNum) {
            zeno::log_warn("iter_count:{} >= maxIterNum:{}", iter_count, maxIterNum);
        }
        set_output2("out", prim);
    }
};
ZENDEFNODE(PrimDice,
{ /* inputs: */ {
    "prim",
    "camera",
    {"int", "width", "1920"},
    {"int", "height", "1080"},
    {"int", "maxIterNum", "1"},
    {"float", "factor", "0.5"},
}, /* outputs: */ {
    "out"
}, /* params: */ {
}, /* category: */ {
    "primitive",
}});

}