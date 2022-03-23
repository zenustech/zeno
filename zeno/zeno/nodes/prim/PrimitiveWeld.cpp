#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <zeno/utils/value_type.h>
#include <unordered_map>

namespace zeno {

struct PrimitiveWeld : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        float dx = 0.00001f;
        float inv_dx = 1.0f / dx;

        std::unordered_map<vec3i, int, tuple_hash, tuple_equal> lut;
        lut.reserve(prim->verts.size());
        for (int i = 0; i < prim->verts.size(); i++) {
            vec3f pos = prim->verts[i];
            vec3i posi = vec3i(pos * inv_dx);
            lut.emplace(posi, i);
        }

        std::vector<int> indices;
        indices.resize(lut.size());
        int curr = 0;
        for (auto const &[key, idx]: lut) {
            indices[curr++] = idx;
        }

        prim->foreach_attr([&] (auto const &key, auto &arr) {
            using T = decltype(value_type_of(arr));
            std::vector<T> new_arr(indices.size());
#pragma omp parallel for
            for (int i = 0; i < indices.size(); i++) {
                new_arr[i] = arr[indices[i]];
            }
            arr = std::move(new_arr);
        });

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveWeld, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}

