#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/fileio.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <string_view>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <fstream>

namespace zeno {
namespace {

template <std::size_t ...Is>
static bool match_helper(char const *&it, char const *arr, std::index_sequence<Is...>) {
    if (((it[Is] == arr[Is]) && ...)) {
        it += sizeof...(Is);
        return true;
    } else {
        return false;
    }
}

template <std::size_t N>
static bool match(char const *&it, char const (&arr)[N]) {
    return match_helper(it, arr, std::make_index_sequence<N - 1>{});
}

static float takef(char const *&it) {
    char *eptr;
    float val = std::strtof(it, &eptr);
    it = eptr;
    return val;
}

static uint32_t takeu(char const *&it) {
    char *eptr;
    uint32_t val(std::strtoul(it, &eptr, 10));
    it = eptr;
    return val;
}

std::shared_ptr<PrimitiveObject> parse_obj(std::vector<char> &&bin) {
    bin.resize(bin.size() + 8, '\0');

    char const *it = bin.data();
    char const *eit = bin.data() + bin.size() - 8;

    auto prim = std::make_shared<PrimitiveObject>();

    std::vector<vec2i> uvs;

    while (it < eit) {
        auto nit = std::find(it, eit, '\n');
        if (*it == '#') {
            zeno::log_info("obj comment: {}", std::string_view(it, nit - it));
        } else if (match(it, "v ")) {
            float x = takef(it);
            float y = takef(it);
            float z = takef(it);
            zeno::log_info("v {} {} {}", x, y, z);
            prim->verts.emplace_back(x, y, z);
        } else if (match(it, "vt ")) {
            float x = takef(it);
            float y = takef(it);
            zeno::log_info("vt {} {}", x, y);
            uvs.emplace_back(x, y);
        } else if (match(it, "f ")) {
            uint32_t beg = prim->loops.size();
            uint32_t cnt{};
            while (it != nit) {
                uint32_t x = takeu(it);
                uint32_t xt{};
                if (*it == '/' && it[1] != '/') {
                    xt = takeu(it);
                }
                it = std::find(it, nit, ' ');
                --x;
                prim->loops.push_back(x);
                ++cnt;
                zeno::log_info("loop {}", x);
                it = std::find_if(it, nit, [] (char c) { return c != ' '; });
            }
            prim->polys.emplace_back(beg, cnt);
            zeno::log_info("poly {} {}", beg, cnt);
        } else if (match(it, "o ")) {
            std::string_view o_name(it, nit - it);
        }
        it = nit + 1;
    }

    return prim;
}

struct ReadObjPrim : INode {
    virtual void apply() override {
        auto path = get_input<StringObject>("path")->get();
        auto binary = file_get_binary(path);
        auto prim = parse_obj(std::move(binary));
        if (get_param<bool>("triangulate")) {
            prim_polys_to_tris(prim.get());
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadObjPrim,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {"bool", "triangulate", "1"},
        {"bool", "allow_quads", "1"},
        }, /* category: */ {
        "primitive",
        }});

}
}
