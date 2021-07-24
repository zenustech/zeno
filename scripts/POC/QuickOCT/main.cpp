#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include "OctreeObject.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stack>

using namespace zenbase;


struct FishYields : zeno::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto rate = std::get<int>(get_param("rate"));

    for (auto &[_, arr]: stars->m_attrs) {
        std::visit([rate](auto &arr) {
            for (int i = 0, j = 0; j < arr.size(); i++, j += rate) {
                arr[i] = arr[j];
            }
        }, arr);
    }
    size_t new_size = stars->size() / rate;
    printf("fish yields new_size = %zd\n", new_size);
    stars->resize(new_size);

    set_output("stars", get_input("stars"));
  }
};

static int defFishYields = zeno::defNodeClass<FishYields>("FishYields",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"int", "rate", "1 1"},
    }, /* category: */ {
    "NBodySolver",
    }});


struct AdvectStars : zeno::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zeno::vec3f>("pos");
    auto &vel = stars->attr<zeno::vec3f>("vel");
    auto &acc = stars->attr<zeno::vec3f>("acc");
    auto dt = get_input("dt")->as<NumericObject>()->get<float>();
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        pos[i] += vel[i] * dt + acc[i] * (dt * dt / 2);
        vel[i] += acc[i] * dt;
    }

    set_output("stars", get_input("stars"));
  }
};

static int defAdvectStars = zeno::defNodeClass<AdvectStars>("AdvectStars",
    { /* inputs: */ {
    "stars",
    "dt",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    }, /* category: */ {
    "NBodySolver",
    }});


// calc 60-bit morton code from 20-bit X,Y,Z fixed pos grid
static unsigned long morton3d(zeno::vec3f const &pos) {
    zeno::vec3L v = zeno::clamp(zeno::vec3L(zeno::floor(pos * 1048576.0f)), 0ul, 1048575ul);
    static_assert(sizeof(v[0]) == 8);

    v = (v * 0x0000000100000001ul) & 0xFFFF00000000FFFFul;
    v = (v * 0x0000000000010001ul) & 0x00FF0000FF0000FFul;
    v = (v * 0x0000000000000101ul) & 0xF00F00F00F00F00Ful;
    v = (v * 0x0000000000000011ul) & 0x30C30C30C30C30C3ul;
    v = (v * 0x0000000000000005ul) & 0x4924924949249249ul;

    return v[0] * 4 + v[1] * 2 + v[2];
}


struct LinearOctree : zeno::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &pos = stars->attr<zeno::vec3f>("pos");

    // compute boundaries
    assert(pos.size() > 0);
    zeno::vec3f bmin = pos[0];
    zeno::vec3f bmax = pos[0];
    for (int i = 1; i < stars->size(); i++) {
        bmin = zeno::min(bmin, pos[i]);
        bmax = zeno::max(bmax, pos[i]);
    }
    bmin -= 1e-6;
    bmax += 1e-6;
    auto radii = bmax - bmin;
    auto offset = -bmin / radii;
    float radius = zeno::max(radii[0], zeno::max(radii[1], radii[2]));

    // compute morton code
    std::vector<unsigned long> mc(stars->size());

    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        mc[i] = morton3d(pos[i] / radii + offset);
    }

#if 0
    // sort by morton code
    std::vector<int> indices(mc.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&mc](int pos1, int pos2) {
        return mc[pos1] < mc[pos2];
    });

    for (auto &[_, arr]: stars->m_attrs) {
        std::visit([&indices](auto &arr) {
            auto tmparr = arr;  // deep-copy
            for (int i = 0; i < arr.size(); i++) {
                arr[i] = tmparr[indices[i]];
            }
        }, arr);
    }
#endif

    // construct octree
    auto tree = zeno::IObject::make<OctreeObject>();
    tree->offset = offset;
    tree->radius = radius;

    auto &children = tree->children;
    children.resize(1);

    std::stack<int> stack;
    for (int i = 0; i < stars->size(); i++) {
        stack.push(i);
    }

    while (!stack.empty()) {
        int pid = stack.top(); stack.pop();
        int k, curr = 0;
        for (k = 60 - 3; k >= 0; k -= 3) {
            int sel = (mc[pid] >> k) & 7;
            int ch = children[curr][sel];
            if (ch == 0) {  // empty
                // directly insert a leaf node
                children[curr][sel] = -1 - pid;
                break;
            } else if (ch > 0) {  // child node
                // then visit into this node
                curr = ch;
            } else {  // leaf node
                // pop the leaf, replace with a child node, and visit later
                int oldpid = -1 - ch;
                stack.push(oldpid);
                curr = children[curr][sel] = children.size();
                children.emplace_back();
            }
        }
        if (k < 0) {  // the difference of star pos < 1 / 1048576
            printf("ERROR: particle morton code overlap 0x%lX!\n", mc[pid]);
        }
    }

    printf("%zd stars, %zd nodes, radius %f\n", stars->size(), children.size(), radius);
    set_output("tree", tree);
    set_output("stars", get_input("stars"));
  }
};

static int defLinearOctree = zeno::defNodeClass<LinearOctree>("LinearOctree",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    "tree",
    }, /* params: */ {
    }, /* category: */ {
    "NBodySolver",
    }});


struct CalcOctreeAttrs : zeno::INode {
    virtual void apply() override {
        auto stars = get_input("stars")->as<PrimitiveObject>();
        auto tree = get_input("tree")->as<OctreeObject>();
        auto &pos = stars->attr<zeno::vec3f>("pos");
        auto &mass = stars->attr<float>("mass");
        auto &children = tree->children;

        tree->mass.clear();
        tree->CoM.clear();
        tree->mass.resize(children.size());
        tree->CoM.resize(children.size());

        #pragma omp parallel for
        for (int no = 0; no < children.size(); no++) {
            std::stack<int> stack;
            stack.push(no);
            while (!stack.empty()) {
                int curr = stack.top(); stack.pop();
                for (int sel = 0; sel < 8; sel++) {
                    int ch = children[curr][sel];
                    if (ch > 0) {  // child node
                        stack.push(ch);
                    } else if (ch < 0) {  // leaf node
                        int pid = -1 - ch;
                        tree->mass[no] += mass[pid];
                        tree->CoM[no] += pos[pid] * mass[pid];
                    }
                }
            }
        }
        #pragma omp parallel for
        for (int no = 0; no < children.size(); no++) {
            if (tree->mass[no] != 0)
                tree->CoM[no] /= tree->mass[no];
        }

        printf("total mass %f at %f %f %f\n", tree->mass[0],
            tree->CoM[0][0], tree->CoM[0][1], tree->CoM[0][2]);

        set_output("tree", get_input("tree"));
    }
};

static int defCalcOctreeAttrs = zeno::defNodeClass<CalcOctreeAttrs>("CalcOctreeAttrs",
    { /* inputs: */ {
    "tree",
    "stars",
    }, /* outputs: */ {
    "tree",
    }, /* params: */ {
    }, /* category: */ {
    "NBodySolver",
    }});


float invpow3(zeno::vec3f const &a, float eps) {
    float rr = zeno::dot(a, a) + eps * eps;
    float r = zeno::sqrt(rr);
    return 1 / (r * rr);
}


struct ComputeGravity : zeno::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto tree = get_input("tree")->as<OctreeObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zeno::vec3f>("pos");
    auto &vel = stars->attr<zeno::vec3f>("vel");
    auto &acc = stars->attr<zeno::vec3f>("acc");
    auto G = std::get<float>(get_param("G"));
    auto lam = std::get<float>(get_param("lam"));
    auto eps = std::get<float>(get_param("eps"));
    printf("computing gravity...\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] = zeno::vec3f(0);
    }
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        std::stack<std::tuple<int, int>> stack;
        stack.push(std::make_tuple<int, int>(0, 0));
        while (!stack.empty()) {
            auto [curr, depth] = stack.top(); stack.pop();
            auto d2CoM = tree->CoM[curr] - pos[i];
            float node_size = tree->radius / (1 << depth);
            if (zeno::length(d2CoM) > lam * node_size) {
                printf("accel %d by far %d %f %d\n", i, curr, node_size, depth);
                acc[i] += d2CoM * tree->mass[curr] * invpow3(d2CoM, eps);
                continue;
            }
            for (int sel = 0; sel < 8; sel++) {
                int ch = tree->children[curr][sel];
                if (ch > 0) {  // child node
                    stack.push(std::make_tuple<int, int>((int)ch, depth + 1));
                } else if (ch < 0) {  // leaf node
                    int pid = -1 - ch;
                    auto d2leaf = pos[pid] - pos[i];
                    printf("accel %d by near %d\n", i, pid);
                    acc[i] += d2leaf * mass[pid] * invpow3(d2leaf, eps);
                }
            }
        }
    }
    printf("computing gravity done\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] *= G;
    }

    set_output("stars", get_input("stars"));
  }
};

static int defComputeGravity = zeno::defNodeClass<ComputeGravity>("ComputeGravity",
    { /* inputs: */ {
    "stars",
    "tree",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"float", "G", "1.0 0"},
    {"float", "eps", "0.0001 0"},
    {"float", "lam", "1.0 0"},
    }, /* category: */ {
    "NBodySolver",
    }});


struct DirectComputeGravity : zeno::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zeno::vec3f>("pos");
    auto &vel = stars->attr<zeno::vec3f>("vel");
    auto &acc = stars->attr<zeno::vec3f>("acc");
    auto G = std::get<float>(get_param("G"));
    auto eps = std::get<float>(get_param("eps"));
    printf("computing gravity...\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] = zeno::vec3f(0);
    }
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        for (int j = 0; j < stars->size(); j++) {
            auto rij = pos[j] - pos[i];
            rij *= invpow3(rij, eps);
            acc[i] += mass[j] * rij;
        }
    }
    printf("computing gravity done\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] *= G;
    }

    set_output("stars", get_input("stars"));
  }
};

static int defDirectComputeGravity = zeno::defNodeClass<DirectComputeGravity>("DirectComputeGravity",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"float", "G", "1.0 0"},
    {"float", "eps", "0.0001 0"},
    }, /* category: */ {
    "NBodySolver",
    }});
