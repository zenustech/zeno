#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <cstring>
#include <cstdlib>
#include <random>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {
namespace {

struct randtype_scalar01 {
    auto operator()(wangsrng &rng) const {
        float offs{rng.next_float()};
        return offs;
    }
};

struct randtype_scalar11 {
    auto operator()(wangsrng &rng) const {
        float offs{rng.next_float()};
        return offs * 2 - 1;
    }
};

struct randtype_cube01 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float(), rng.next_float(), rng.next_float()};
        return offs;
    }
};

struct randtype_cube11 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float(), rng.next_float(), rng.next_float()};
        return offs * 2 - 1;
    }
};

struct randtype_plane01 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float(), rng.next_float(), 0};
        return offs;
    }
};

struct randtype_plane11 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float() * 2 - 1, rng.next_float() * 2 - 1, 0};
        return offs;
    }
};

struct randtype_disk {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r1 = std::sqrt(r1);
        r2 *= M_PI * 2;
        vec3f offs{r1 * std::sin(r2), r1 * std::cos(r2), 0};
        return offs;
    }
};

struct randtype_cylinder {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r1 = r1 * 2 - 1;
        r2 *= M_PI * 2;
        vec3f offs{std::sin(r2), std::cos(r2), r1};
        return offs;
    }
};

struct randtype_ball {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        float r3 = rng.next_float();
        r1 = r1 * 2 - 1;
        r2 *= M_PI * 2;
        r3 = std::cbrt(r3) * std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

struct randtype_semiball {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        float r3 = rng.next_float();
        r2 *= M_PI * 2;
        r3 = std::cbrt(r3) * std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

struct randtype_sphere {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r1 = r1 * 2 - 1;
        r2 *= M_PI * 2;
        float r3 = std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

struct randtype_semisphere {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r2 *= M_PI * 2;
        float r3 = std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

using RandTypes = std::variant
    < randtype_scalar01
    , randtype_scalar11
    , randtype_cube01
    , randtype_cube11
    , randtype_plane01
    , randtype_plane11
    , randtype_disk
    , randtype_cylinder
    , randtype_ball
    , randtype_semiball
    , randtype_sphere
    , randtype_semisphere
>;

static std::string_view lutRandTypes[] = {
    "scalar01",
    "scalar11",
    "cube01",
    "cube11",
    "plane01",
    "plane11",
    "disk",
    "cylinder",
    "ball",
    "semiball",
    "sphere",
    "semisphere",
};

}

static NumericValue numRandom(vec3f dir, std::string randType, float base, float scale, int seed) {
    auto randty = enum_variant<RandTypes>(array_index_safe(lutRandTypes, randType, "randType"));
    if (seed == -1) seed = std::random_device{}();

    wangsrng rng(seed);
    NumericValue ret;
    std::visit([&] (auto &&randty) {
        using T = std::invoke_result_t<decltype(randty), wangsrng &>;
        T offs = base + randty(rng) * scale;
        if constexpr (std::is_same_v<T, vec3f>) {
            vec3f b3 = dir, b1, b2;
            pixarONB(b3, b1, b2);
            offs = offs[0] * b1 + offs[1] * b2 + offs[2] * b3;
        }
        ret = offs;
    }, randty);
    return ret;
}

namespace {

struct NumRandom : INode {
    virtual void apply() override {
        auto dir = get_input2<vec3f>("dir");
        auto base = get_input2<float>("base");
        auto scale = get_input2<float>("scale");
        auto seed = get_input2<int>("seed");
        auto randType = get_input2<std::string>("randType");
        auto ret = objectFromLiterial(numRandom(dir, randType, base, scale, seed));
        set_output("value", std::move(ret));
    }
};

ZENDEFNODE(NumRandom, {
    {
    {"vec3f", "dir", "0,0,1"},
    {"float", "base", "0"},
    {"float", "scale", "1"},
    {"int", "seed", "-1"},
    {"enum scalar01 scalar11 cube01 cube11 plane01 plane11 disk cylinder ball semiball sphere semisphere", "randType", "scalar01"},
    },
    {
    {"int", "value"},
    },
    {
    },
    {"numeric"},
});

struct NumRandomSeedCombine : INode {
    virtual void apply() override {
        int x = get_input2<int>("x");
        int y = get_input2<int>("y");
        int z = get_input2<int>("z");
        int w = get_input2<int>("w");
        wangsrng rng(x, y, z, w);
        set_output2("seed", rng.next_int32());
    }
};

ZENDEFNODE(NumRandomSeedCombine, {
    {
    {"int", "x", "0"},
    {"int", "y", "0"},
    {"int", "z", "0"},
    {"int", "w", "0"},
    },
    {
    {"int", "seed"},
    },
    {
    },
    {"numeric"},
});

struct NumRandomInt : INode {
    virtual void apply() override {
        int valmin = get_input2<int>("valmin");
        int valmax = get_input2<int>("valmax");
        int seed = get_input2<int>("seed");
        if (seed == -1) seed = std::random_device{}();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> uni(valmin, valmax);
        int value = uni(gen);
        set_output2("value", value);
    }
};

ZENDEFNODE(NumRandomInt, {
    {
    {"int", "seed", "-1"},
    {"int", "valmin", "0"},
    {"int", "valmax", "1"},
    },
    {
    {"int", "value"},
    },
    {
    },
    {"numeric"},
});

struct NumRandomFloat : INode {
    virtual void apply() override {
        float valmin = get_input2<float>("valmin");
        float valmax = get_input2<float>("valmax");
        int seed = get_input2<int>("seed");
        if (seed == -1) seed = std::random_device{}();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> uni(valmin, valmax);
        float value = uni(gen);
        set_output2("value", value);
    }
};

ZENDEFNODE(NumRandomFloat, {
    {
    {"int", "seed", "-1"},
    {"float", "valmin", "0"},
    {"float", "valmax", "1"},
    },
    {
    {"float", "value"},
    },
    {
    },
    {"numeric"},
});

}
}
