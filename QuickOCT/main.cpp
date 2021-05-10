#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include "OctreeObject.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stack>

using namespace zenbase;


struct FishYields : zen::INode {
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

    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defFishYields = zen::defNodeClass<FishYields>("FishYields",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"int", "rate", "1 1"},
    }, /* category: */ {
    "NBodySolver",
    }});


struct AdvectStars : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zen::vec3f>("pos");
    auto &vel = stars->attr<zen::vec3f>("vel");
    auto &acc = stars->attr<zen::vec3f>("acc");
    auto dt = std::get<float>(get_param("dt"));
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        pos[i] += vel[i] * dt + acc[i] * (dt * dt / 2);
        vel[i] += acc[i] * dt;
    }

    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defAdvectStars = zen::defNodeClass<AdvectStars>("AdvectStars",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"float", "dt", "0.01 0"},
    }, /* category: */ {
    "NBodySolver",
    }});


// calc 60-bit morton code from 20-bit X,Y,Z fixed pos grid
static unsigned long morton3d(zen::vec3f const &pos) {
    zen::vec3L v = zen::clamp(zen::vec3L(zen::floor(pos * 1048576.0f)), 0ul, 1048575ul);
    static_assert(sizeof(v[0]) == 8);

    v = (v * 0x0000000100000001ul) & 0xFFFF00000000FFFFul;
    v = (v * 0x0000000000010001ul) & 0x00FF0000FF0000FFul;
    v = (v * 0x0000000000000101ul) & 0xF00F00F00F00F00Ful;
    v = (v * 0x0000000000000011ul) & 0x30C30C30C30C30C3ul;
    v = (v * 0x0000000000000005ul) & 0x4924924949249249ul;

    return v[0] * 4 + v[1] * 2 + v[2];
}


struct LinearOctree : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &pos = stars->attr<zen::vec3f>("pos");

    // compute boundaries
    assert(pos.size() > 0);
    zen::vec3f bmin = pos[0];
    zen::vec3f bmax = pos[0];
    for (int i = 1; i < stars->size(); i++) {
        bmin = zen::min(bmin, pos[i]);
        bmax = zen::max(bmax, pos[i]);
    }
    auto scale = 1 / (bmax - bmin);
    auto offset = -bmin * scale;

    // compute morton code
    std::vector<unsigned long> mc(stars->size());

    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        mc[i] = morton3d(pos[i] * scale + offset);
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
    auto tree = zen::IObject::make<OctreeObject>();
    auto &children = tree->children;
    children.resize(1);

    std::stack<int> stack;
    for (int i = 0; i < stars->size(); i++) {
        stack.push(i);
    }

    while (!stack.empty()) {
        int pid = stack.top(); stack.pop();
        int k, curr = 0;
        for (k = 27; k >= 0; k -= 3) {
            int sel = (mc[pid] >> k) & 7;
            int ch = children[curr][sel];
            if (ch == 0) {  // empty
                // directly insert a leaf node
                children[curr][sel] = -pid;
                break;
            } else if (ch > 0) {  // child node
                // then visit into this node
                curr = ch;
            } else {  // leaf node
                // pop the leaf, replace with a child node, and visit later
                stack.push(-ch);
                curr = children[curr][sel] = children.size();
                children.emplace_back();
            }
        }
        if (k < 0) {  // the difference of star pos < 1 / 1048576
            printf("ERROR: particle morton code overlap!\n");
        }
    }

    printf("LinearOctree: %zd stars -> %zd nodes\n", stars->size(), children.size());
    set_output("tree", tree);
    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defLinearOctree = zen::defNodeClass<LinearOctree>("LinearOctree",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    "tree",
    }, /* params: */ {
    }, /* category: */ {
    "NBodySolver",
    }});


struct CalcOctreeAttrs : zen::INode {
    virtual void apply() override {
        auto stars = get_input("stars")->as<PrimitiveObject>();
        auto tree = get_input("tree")->as<OctreeObject>();
        auto &pos = stars->attr<zen::vec3f>("pos");
        auto &mass = stars->attr<float>("mass");
        auto &children = tree->children;

        tree->mass.clear();
        tree->CoM.clear();
        tree->mass.resize(children.size());
        tree->CoM.resize(children.size());

        //#pragma omp parallel for
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
                        int pid = -ch;
                        tree->mass[curr] += mass[pid];
                        tree->CoM[curr] += pos[pid] * mass[pid];
                    }
                }
            }
        }
    }
};

static int defCalcOctreeAttrs = zen::defNodeClass<CalcOctreeAttrs>("CalcOctreeAttrs",
    { /* inputs: */ {
    "tree",
    "stars",
    }, /* outputs: */ {
    "tree",
    }, /* params: */ {
    }, /* category: */ {
    "NBodySolver",
    }});


struct ComputeGravity : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zen::vec3f>("pos");
    auto &vel = stars->attr<zen::vec3f>("vel");
    auto &acc = stars->attr<zen::vec3f>("acc");
    auto G = std::get<float>(get_param("G"));
    auto eps = std::get<float>(get_param("eps"));
    printf("computing gravity...\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] = zen::vec3f(0);
    }
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        for (int j = i + 1; j < stars->size(); j++) {
            auto rij = pos[j] - pos[i];
            float r = eps * eps + zen::dot(rij, rij);
            rij /= r * zen::sqrt(r);
            acc[i] += mass[j] * rij;
            acc[j] -= mass[i] * rij;
        }
    }
    printf("computing gravity done\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] *= G;
    }

    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defComputeGravity = zen::defNodeClass<ComputeGravity>("ComputeGravity",
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
