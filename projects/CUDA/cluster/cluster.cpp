#include <cctype>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <random>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include "./kdsearch.h"

namespace zeno {
void compute_mean(int vnum, int knum, const std::vector<vec3f>& pos, const std::vector<int>& cluster, std::vector<vec3f>& center) {
    std::vector<int> count(knum, 0);
    std::vector<vec3f> sum(knum, vec3f(0.f));
    for(int i = 0; i < vnum; i++) {
        count[cluster[i]]++;
        sum[cluster[i]] += pos[i];
    }
#pragma omp parallel for
    for(int i = 0; i < knum; i++)
        center[i] = sum[i] / count[i];
}

void assign_cluster(int vnum, int knum, const std::vector<vec3f>& pos, const std::vector<vec3f>& center, std::vector<int>& cluster) {
#pragma omp parallel for
    for (int i = 0; i < vnum; i++)
        for (int j = 0; j < knum; j++)
            if (distance(pos[i], center[cluster[i]]) > distance(pos[i], center[j]))
                cluster[i] = j;
}

struct DSU {
    std::vector<int> un;
    std::vector<int> siz;

    DSU(int n) {
        un.resize(n);
        siz.resize(n);
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            un[i] = i;
            siz[i] = 1;
        }
    }

    int get_union(int u) {
        if (un[u] != u) un[u] = get_union(un[u]);
        return un[u];
    }

    int get_size(int u) {
        return siz[get_union(u)];
    }

    void merge(int u, int v) {
        int uu = get_union(u);
        int vv = get_union(v);
        if (uu != vv) {
            un[uu] = vv;
            siz[vv] += siz[uu];
            siz[uu] = 0;
        }
    }
};

struct ParticleClustering : INode {
    virtual void apply() override {
        auto pars = get_input<zeno::PrimitiveObject>("pars");
        int knum = get_input2<int>("cluster_number");
        float dmax = get_input2<float>("diameter");
        auto cluster_tag = get_input<zeno::StringObject>("cluster_tag")->get();
        bool color = get_input2<bool>("paint_color");
        auto &pos = pars->verts;
        int vnum = pos->size();
        auto &cluster = pars->add_attr<int>(cluster_tag);
        zeno::log_info("vnum: {}", vnum);
        std::vector<vec3f> center{};
        if (knum <= 0 && dmax <= 0) {
            zeno::log_warn("please enter either \"cluster_number\" or \"diameter\"");
            set_output("pars", std::move(pars));
            return;
        } else if (knum > 0 && dmax > 0) {
            zeno::log_warn("please enter only one of \"cluster_number\" and \"diameter\"");
            set_output("pars", std::move(pars));
            return;
        } else if (knum > 0) {
            center.resize(knum);
            srand(37);
            for (int i = 0; i < knum; ++i)
                center[i] = pos[rand() % vnum];
            assign_cluster(vnum, knum, pos, center, cluster);
            bool flag;
            std::vector<int> old(vnum);
            do {
                compute_mean(vnum, knum, pos, cluster, center);
    #pragma omp parallel for
                for (int i = 0; i < vnum; i++)
                    old[i] = cluster[i];
                assign_cluster(vnum, knum, pos, center, cluster);
                flag = true;
                for (int i = 0; i < vnum; i++){
                    if (old[i] != cluster[i]) {
                        flag = false;
                        break;
                    }
                }
            } while(flag);
        } else {
            KdTree* kdTree = new KdTree(pos, vnum);
            auto dsu = DSU(vnum);
            for (int i = 0; i < vnum; ++i) {
                if (dsu.get_size(i) == 1) {
                    auto neighbors = kdTree->fix_radius_search(pos[i], dmax/2);
                    for (auto &j: neighbors) {
                        if (dsu.get_size(j) == 1)
                            dsu.merge(j, i);
                    }
                }
            }
            std::map<int, int> cluster_id{};
            knum = 0;
            for (int i = 0; i < vnum; ++i) {
                int dsu_id = dsu.get_union(i);
                if (cluster_id.count(dsu_id) == 0) {
                    cluster_id[dsu_id] = knum++;
                }
                cluster[i] = cluster_id[dsu_id];
            }
            center.resize(knum);
            for (int i = 0; i < vnum; ++i) {
                int dsu_id = dsu.get_union(i);
                if (dsu_id == i)
                    center[cluster[i]] = pos[i];
            }
#pragma omp parallel for
            for (int i = 0; i < vnum; ++i)
                for (int j = 0; j < knum; ++j)
                    if (distance(pos[i], center[j]) < distance(pos[i], center[cluster[i]]))
                        cluster[i] = j;
        }
        compute_mean(vnum, knum, pos, cluster, center);
        zeno::log_info("into {} clusters", knum);

        if (color) {
            auto &clr = pars->verts.add_attr<vec3f>("clr");
#pragma omp parallel for
            for (int i = 0; i < vnum; ++i) {
                std::mt19937 rng;
                rng.seed(cluster[i]);
                unsigned int r = rng() % 256u;
                unsigned int g = rng() % 256u;
                unsigned int b = rng() % 256u;
                zeno::vec3f c{1.f * r / 256.f, 1.f * g / 256.f, 1.f * b / 256.f};
                clr[i] = c;
            }
        }

        auto cluster_center = std::make_shared<ListObject>();
        for (int i = 0; i < knum; ++i) {
            auto numeric_obj = std::make_shared<zeno::NumericObject>();
            numeric_obj->value = center[i];
            cluster_center->arr.emplace_back(numeric_obj);
        }
        set_output("pars", std::move(pars));
        set_output("cluster_center",std::move(cluster_center));
    }
};

ZENO_DEFNODE(ParticleClustering)
({
    {{"PrimitiveObject", "pars"},
    {"int", "cluster_number", "0"},
    {"float", "diameter", "0"},
    {"string", "cluster_tag", "cluster_index"},
    {"bool", "paint_color", "1"}},
    {{"PrimitiveObject", "pars"},
    {"list", "cluster_center"}},
    {},
    {"primitive"},
});
} // namespace zeno
