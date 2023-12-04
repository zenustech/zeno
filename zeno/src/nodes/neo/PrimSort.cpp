#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>

namespace zeno {

struct PrimSort : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto attr = get_input2<std::string>("Attribute");
    auto mode = get_input2<std::string>("Vertex Sort");
    auto reverse = get_input2<bool>("Reverse");
    
    if (mode == "NoChange") {
        if (reverse) {
            prim->verts.forall_attr<AttrAcceptAll>([&] (auto &key, auto &arr) {
                std::reverse(arr.begin(), arr.end());
            });
            /*for (auto& tri : prim->tris) {
                std::swap(tri[0], tri[2]);
            }*/
        }
      set_output("prim", std::move(prim));
    }
    else if (mode == "ByAttribute") {
        auto &tris = prim->tris.values;
        std::vector<size_t> indices(prim->verts.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        if (prim->attr_is<float>(attr)){
            auto &tag = prim->verts.attr<float>(attr);
            std::stable_sort(indices.begin(), indices.end(), [&tag, reverse] (size_t a, size_t b) {//stable sort
                return reverse ? tag[a] > tag[b] : tag[a] < tag[b];
            });
        }
        else if (prim->attr_is<int>(attr)){
            auto &tag = prim->verts.attr<int>(attr);
            std::stable_sort(indices.begin(), indices.end(), [&tag, reverse] (size_t a, size_t b) {
                return reverse ? tag[a] > tag[b] : tag[a] < tag[b];
            });
        }
        else{
            throw std::runtime_error("Attribute type not supported");
        }

        prim->verts.forall_attr<AttrAcceptAll>([&] (auto &key, auto &arr) {
            auto oldarr = std::move(arr);
            arr.resize(oldarr.size());
            parallel_for(oldarr.size(), [&] (size_t i) {
                arr[i] = oldarr[indices[i]];
            });
        });

        // inverse mapping
        std::vector<size_t> reverse_indices(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            reverse_indices[indices[i]] = i;
        }
        for (auto& tri : tris) {
            for (auto& idx : tri) {
                idx = reverse_indices[idx];
            }
        }
        set_output("prim", std::move(prim));
    }
  }
};

ZENDEFNODE(PrimSort, {
    {
    {"PrimitiveObject", "prim"},
    {"enum NoChange ByAttribute", "Vertex Sort", "NoChange"},//Add more methods
    {"string", "Attribute", "index"},
    {"bool", "Reverse", "0"}
    //{"int", "component", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}