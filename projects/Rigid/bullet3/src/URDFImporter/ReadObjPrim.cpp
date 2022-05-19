#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <fstream>

using namespace zeno;

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

static int takeu(char const *&it) {
    char *eptr;
    int val(std::strtoul(it, &eptr, 10));
    it = eptr;
    return val;
}

std::shared_ptr<zeno::PrimitiveObject> parse_obj(std::vector<char> &&bin) {
    bin.resize(bin.size() + 8, '\0');

    char const *it = bin.data();
    char const *eit = bin.data() + bin.size() - 8;

    auto prim = std::make_shared<zeno::PrimitiveObject>();

    std::vector<vec2f> uvs;
    std::vector<int> loop_uvs;

    while (it < eit) {
        auto nit = std::find(it, eit, '\n');

        if (match(it, "v ")) {
            float x = takef(it);
            float y = takef(it);
            float z = takef(it);
            prim->verts.emplace_back(x, y, z);

        } else if (match(it, "vt ")) {
            float x = takef(it);
            float y = takef(it);
            uvs.emplace_back(x, y);

        } else if (match(it, "f ")) {
            int beg = prim->loops.size();
            int cnt{};
            while (it != nit) {
                int x = takeu(it) - 1;
                if (*it == '/' && it[1] != '/') {
                    ++it;
                    int xt = takeu(it) - 1;
                    loop_uvs.push_back(xt);
                }
                it = std::find(it, nit, ' ');
                prim->loops.push_back(x);
                ++cnt;
                it = std::find_if(it, nit, [] (char c) { return c != ' '; });
            }
            prim->polys.emplace_back(beg, cnt);
            prim->tris.emplace_back(prim->loops[beg], prim->loops[beg+1],  prim->loops[beg+2]);

        } else if (match(it, "l")) {
            int x = takeu(it) - 1;
            int y = takeu(it) - 1;
            prim->lines.emplace_back(x, y);

            //} else if (match(it, "o ")) {
            // todo: support tag verts to be multi components of primitive
            //std::string_view o_name(it, nit - it);

        }
        it = nit + 1;
    }

    if (!loop_uvs.empty()) {
        // todo: support vec2f in attr...
        auto &attuv = prim->loops.add_attr<vec3f>("uv");
        attuv.resize(prim->loops.size());
        for (size_t i = 0; i < loop_uvs.size(); i++) {
            auto uv = uvs[loop_uvs[i]];
            attuv[i] = vec3f(uv[0], uv[1], 0);
        }
    }

    return prim;
}

