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
#include "zeno/extra/TempNode.h"

namespace zeno {
struct EdgeInfo {
    //bool cutted = false;
    int v0 = 0;
    int v1 = 0;
    char mask = 0;
    int index = -1;
    int edge_array_idx = -1;
    int face_idx0 = -1;
    int face_idx1 = -1;
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
        //��ʵ�ڲ�����Ϊɶ������ô��Ҫ�� �����Ű����ĵ��Ͳ�work�ˣ� Ȼ�����ұ����������Ӧ���й�ϵ������
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
    std::vector<int> new_old_map;

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
    new_old_map.resize(total_num);
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
                    new_old_map[emit_start + j] = prim->polys.attr<int>("parentID")[i];
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
        prim->polys.attr<int>("parentID")[i] = new_old_map[i];
    }
}
static float area(zeno::vec3f v0, zeno::vec3f v1, zeno::vec3f v2)
{
    vec3f e0 = v1 - v0;
    vec3f e1 = v2 - v0;
    return 0.5 * length(cross(e0,e1));
}

//result = w.x*v0_attr + w.y*v1_attr + w.z*v2_attr
static zeno::vec3f findBarycentric(zeno::vec3f p, zeno::vec3f v0, zeno::vec3f v1, zeno::vec3f v2)
{
    float t_area = area(v0, v1, v2);
    if(t_area<0.000001)
        return zeno::vec3f (1,0,0);
    float w0 = min(area(p, v1, v2)/t_area, 1.0f);
    float w1 = min(area(p, v0, v2)/t_area, 1.0f - w0);
    float w2 = max(1.0f -  w0 - w1, 0.0f);
    return zeno::vec3f (w0, w1, w2);
}

static void prim_interp(PrimitiveObject *origin_prim,  PrimitiveObject *prim) {
    origin_prim->polys.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        auto &parentIDs = prim->polys.attr<int>("parentID");
        auto &attr = prim->polys.add_attr<T>(key);
        #pragma omp parallel for
        for (auto i = 0; i < prim->polys.size(); i++) {
            int parentID = parentIDs[i];
            attr[i] = arr[parentID];
        }
    });

    origin_prim->verts.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        auto &attr = prim->verts.add_attr<T>(key);
        auto &parentIDs = prim->polys.attr<int>("parentID");
        auto &bw = prim->loops.attr<vec3f>("bw");
        #pragma omp parallel for
        for (auto i = 0; i < prim->polys.size(); i++) {
            int parentID = parentIDs[i];
            auto vid0 = prim->loops[prim->polys[i][0]];
            auto vid1 = prim->loops[prim->polys[i][0]+1];
            auto vid2 = prim->loops[prim->polys[i][0]+2];

            auto ovid0 = origin_prim->loops[origin_prim->polys[parentID][0]];
            auto ovid1 = origin_prim->loops[origin_prim->polys[parentID][0]+1];
            auto ovid2 = origin_prim->loops[origin_prim->polys[parentID][0]+2];

            auto odata0 = arr[ovid0];
            auto odata1 = arr[ovid1];
            auto odata2 = arr[ovid2];

            auto bw0 = bw[prim->polys[i][0]];
            auto bw1 = bw[prim->polys[i][0]+1];
            auto bw2 = bw[prim->polys[i][0]+2];

            auto data0 = bw0[0] * odata0 + bw0[1] * odata1 + bw0[2] * odata2;
            auto data1 = bw1[0] * odata0 + bw1[1] * odata1 + bw1[2] * odata2;
            auto data2 = bw2[0] * odata0 + bw2[1] * odata1 + bw2[2] * odata2;

            attr[vid0] = data0;
            attr[vid1] = data1;
            attr[vid2] = data2;
        }
    });


    if (!origin_prim->loops.has_attr("uvs")) {
        return;
    }
    auto &parentIDs = prim->polys.attr<int>("parentID");
    auto &ouv = origin_prim->loops.attr<int>("uvs");
    auto &bw = prim->loops.attr<vec3f>("bw");
    auto &puv = prim->loops.add_attr<int>("uvs");
    prim->uvs.resize(prim->loops.size());
    #pragma omp parallel for
    for(auto i=0;i<prim->polys.size();i++)
    {
        size_t parentID = parentIDs[i];

        auto ovid0 = ouv[origin_prim->polys[parentID][0]];
        auto ovid1 = ouv[origin_prim->polys[parentID][0]+1];
        auto ovid2 = ouv[origin_prim->polys[parentID][0]+2];

        auto odata0 = origin_prim->uvs[ovid0];
        auto odata1 = origin_prim->uvs[ovid1];
        auto odata2 = origin_prim->uvs[ovid2];

        auto bw0 = bw[prim->polys[i][0]];
        auto bw1 = bw[prim->polys[i][0]+1];
        auto bw2 = bw[prim->polys[i][0]+2];

        auto data0 = bw0[0] * odata0 + bw0[1] * odata1 + bw0[2] * odata2;
        auto data1 = bw1[0] * odata0 + bw1[1] * odata1 + bw1[2] * odata2;
        auto data2 = bw2[0] * odata0 + bw2[1] * odata1 + bw2[2] * odata2;

        prim->uvs[i*3 + 0] = data0;
        prim->uvs[i*3 + 1] = data1;
        prim->uvs[i*3 + 2] = data2;
        puv[i*3 + 0] = i*3 + 0;
        puv[i*3 + 1] = i*3 + 1;
        puv[i*3 + 2] = i*3 + 2;
    }
}
struct PrimInterp : INode{
    void apply() override{
        auto origin_prim = get_input2<PrimitiveObject>("original_prim");
        auto prim = get_input2<PrimitiveObject>("interp_prim");
        prim_interp(origin_prim.get(), prim.get());
        set_output2("prim", prim);
    }
};

ZENDEFNODE(PrimInterp,
{ /* inputs: */ {
    "original_prim",
    "interp_prim",
}, /* outputs: */ {
    "prim"
}, /* params: */ {
}, /* category: */ {
    "primitive",
}});
struct PrimDisplacement : INode {
    void apply() override {
        auto prim_in = get_input2<PrimitiveObject>("prim_in");
        if(!prim_in->verts.has_attr("disp"))
            prim_in->verts.add_attr<vec3f>("disp");
        if(!prim_in->verts.has_attr("sample_disp"))
            prim_in->verts.add_attr<vec3f>("sample_disp");
        std::string image_file = get_input2<std::string>("image_file");
        std::string mat_name = get_input2<std::string>("mat_name");
        std::string displacement_code = get_input2<std::string>("displacement_code");

        auto img = zeno::TempNodeSimpleCaller("ReadImageFile_v2")
                .set2("path", image_file)
                .set2("srgb_to_linear", int(0))
                .call()
                .get<zeno::PrimitiveObject>("image");

        auto oPrim = zeno::TempNodeSimpleCaller("PrimSample2D")
                .set2("prim", prim_in)
                .set2("image", img)
                .set2("uvChannel", std::string("uv"))
                .set2("uvSource", std::string("loopsuv"))
                .set2("targetChannel",std::string("h"))
                .set2("remapMin",float(0))
                .set2("remapMax",float(1))
                .set2("wrap",std::string("REPEAT"))
                .set2("filter", std::string("linear"))
                .set2("borderColor", zeno::vec3f(0,0,0))
                .set2("invert U", int(0))
                .set2("invert V", int(0))
                .set2("scale", float(1))
                .set2("rotate", float(0))
                .set2("translate", zeno::vec2f(0))
                .call()
                .get<zeno::PrimitiveObject>("outPrim");


        auto disp_prim = zeno::TempNodeSimpleCaller("ParticlesWrangle")
                .set2("prim", oPrim)
                .set2("zfxCode", displacement_code)
                .call()
                .get<zeno::PrimitiveObject>("prim");

        auto &matIds = disp_prim->polys.attr<int>("matid");
        auto &sample_disp = disp_prim->verts.attr<zeno::vec3f>("sample_disp");
        auto &disp = disp_prim->verts.attr<zeno::vec3f>("disp");
        int matNum = disp_prim->userData().get2<int>("matNum");
        std::vector<int> mat_mark(matNum);
        for(int i=0;i<matNum;i++)
        {
            mat_mark[i] = 0;
            if (disp_prim->userData().get2<std::string>(format("Material_{}", i)) == mat_name)
            {
                mat_mark[i] = 1;
            }
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, disp_prim->polys.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                int mat_id = matIds[i];
                if (mat_mark[mat_id]==1) {
                    //get the three points of this poly
                    auto [start, len] = disp_prim->polys[i];
                    for (auto j = 0; j < len; j++) {
                        int vid = disp_prim->loops[start + j];
                        disp[vid] = sample_disp[vid];
                    }
                }
            }
        });


        set_output2("out", disp_prim);
    }
};

ZENDEFNODE(PrimDisplacement,
{ /* inputs: */ {
    "prim_in",
    {"readpath", "image_file"},
    {"string", "mat_name"},
    {"multiline_string", "displacement_code"},
}, /* outputs: */ {
    "out",
}, /* params: */ {
}, /* category: */ {
    "primitive",
}});
struct PrimEdgeCrease : INode {
    void apply() override {
        auto origin_prim = get_input2<PrimitiveObject>("prim");
        auto &polys = origin_prim->polys;
        auto &loops = origin_prim->loops;
        std::unordered_map<std::pair<int,int>,EdgeInfo, PairHash> edges;
        int num_lines = 0;
        auto &face_n = polys.add_attr<zeno::vec3f>("face_N");
        auto sharp_thres = get_input2<float>("sharp_threshold");
        auto corner_thres = get_input2<float>("corner_threshold");
        auto &verts = origin_prim->verts;
        auto &verts_e1 = origin_prim->verts.add_attr<vec3f>("e1");
        auto &verts_e2 = origin_prim->verts.add_attr<vec3f>("e2");
        for(int i=0;i<polys.size();i++)
        {
            zeno::vec2i poly_idx = polys[i];
            zeno::vec3f v0 = origin_prim->verts[loops[poly_idx[0] + 0]];
            zeno::vec3f v1 = origin_prim->verts[loops[poly_idx[0] + 1]];
            zeno::vec3f v2 = origin_prim->verts[loops[poly_idx[0] + 2]];
            face_n[i] = normalize(cross(v1-v0,v2-v1));
            for(int j=0;j<poly_idx[1]-1;j++)
            {
                int vert0 = loops[poly_idx[0] + j];
                int vert1 = loops[poly_idx[0] + j + 1];
                std::pair<int,int> e = { max(vert0,vert1), min(vert0,vert1) };
                if(edges.find(e)!=edges.end())
                {
                    edges[e].face_idx1 = i;
                }else{
                    num_lines++;
                    EdgeInfo ei;
                    ei.face_idx0 = i;
                    edges[e] = ei;
                }
            }
        }
        origin_prim->lines.resize(num_lines);
        auto &edge_crease_weight = origin_prim->lines.add_attr<float>("edge_crease_weight");
        auto &vert_crease_weight = origin_prim->verts.add_attr<float>("vert_crease_weight");
        int line_idx = 0;

        for(auto &key_val:edges)
        {
            float c=0.0f;
            auto e = key_val.first;
            auto ei = key_val.second;
            if(ei.face_idx0==-1 || ei.face_idx1==-1)
            {
                //this is a naked edge
                int vid0 = e.first;
                int vid1 = e.second;
                auto e = zeno::normalize(verts[vid0]-verts[vid1]);
                if(length(verts_e1[vid0])==0)
                {
                    verts_e1[vid0] = e;
                }else{
                    verts_e2[vid0] = e;
                }

                if(length(verts_e1[vid1])==0)
                {
                    verts_e1[vid1] = e;
                }else{
                    verts_e2[vid1] = e;
                }

                c = 0.0f;
            }
            else{
                int f0 = ei.face_idx0;
                int f1 = ei.face_idx1;
                auto n0 = face_n[f0];
                auto n1 = face_n[f1];
                c = dot(n1,n0);
            }
            origin_prim->lines[line_idx] = zeno::vec2i(key_val.first.first,key_val.first.second);
            edge_crease_weight[line_idx] = c<sharp_thres?10:0;
            line_idx++;
        }
        for(int i=0;i<verts.size();i++)
        {
            auto e1 = verts_e1[i];
            auto e2 = verts_e2[i];
            vert_crease_weight[i] = 0;
            if(length(e1)>0 && length(e2)>0)
                vert_crease_weight[i] = dot(e1,e2)>-corner_thres?10.0f:0;
        }
        set_output("oPrim", origin_prim);
    }
};
ZENDEFNODE(PrimEdgeCrease,
{ /* inputs: */ {
    "prim",
    {"float","sharp_threshold","0.6"},
    {"float","corner_threshold","0.1"},
}, /* outputs: */ {
    "oPrim",
}, /* params: */ {
}, /* category: */ {
    "primitive",
}});
struct PrimDice : INode {
    void apply() override {
        auto origin_prim = get_input2<PrimitiveObject>("prim");
        primTriangulateIntoPolys(origin_prim.get());
        auto prim = std::dynamic_pointer_cast<PrimitiveObject>(origin_prim->clone());
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
        prim->polys.add_attr<int>("parentID");
        for(int i=0;i<prim->polys.size();i++)
        {
            prim->polys.attr<int>("parentID")[i] = i;
        }
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
        prim->loops.add_attr<zeno::vec3f>("bw");
        tbb::parallel_for(tbb::blocked_range<size_t>(0, prim->polys.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                auto tri = prim->polys[i];
                auto new_v0_idx = prim->loops[tri[0]];
                auto new_v1_idx = prim->loops[tri[0] + 1];
                auto new_v2_idx = prim->loops[tri[0] + 2];

                int from_idx = prim->polys.attr<int>("parentID")[i];
                auto from_tri = origin_prim->polys[from_idx];
                auto from_v0_idx = origin_prim->loops[from_tri[0]];
                auto from_v1_idx = origin_prim->loops[from_tri[0] + 1];
                auto from_v2_idx = origin_prim->loops[from_tri[0] + 2];

                zeno::vec3f new_v0 = prim->verts[new_v0_idx];
                zeno::vec3f new_v1 = prim->verts[new_v1_idx];
                zeno::vec3f new_v2 = prim->verts[new_v2_idx];

                zeno::vec3f from_v0 = origin_prim->verts[from_v0_idx];
                zeno::vec3f from_v1 = origin_prim->verts[from_v1_idx];
                zeno::vec3f from_v2 = origin_prim->verts[from_v2_idx];

                zeno::vec3f bw0 = findBarycentric(new_v0, from_v0, from_v1, from_v2);
                zeno::vec3f bw1 = findBarycentric(new_v1, from_v0, from_v1, from_v2);
                zeno::vec3f bw2 = findBarycentric(new_v2, from_v0, from_v1, from_v2);

                prim->loops.attr<zeno::vec3f>("bw")[3 * i + 0] = bw0;
                prim->loops.attr<zeno::vec3f>("bw")[3 * i + 1] = bw1;
                prim->loops.attr<zeno::vec3f>("bw")[3 * i + 2] = bw2;
            }
        });
        prim_interp(origin_prim.get(), prim.get());

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
    "out",
}, /* params: */ {
}, /* category: */ {
    "primitive",
}});

}