#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>
#include <unordered_map>

namespace zeno {
namespace {

struct PrimMarkClose : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        float distance = get_input<NumericObject>("distance")->get<float>();

        float factor = 1.0f / distance;
        std::unordered_multimap<vec3i, int, tuple_hash, tuple_equal> lut;
        lut.reserve(prim->verts.size());
        for (int i = 0; i < prim->verts.size(); i++) {
            vec3f pos = prim->verts[i];
            vec3i posi = vec3i(floor(pos * factor));
            lut.emplace(posi, i);
        }

        auto &tag = prim->verts.add_attr<int>(tagAttr);
        if (lut.size()) {
            int cnt = 0;
            auto last_key = lut.begin()->first;
            for (auto const &[key, idx]: lut) {
                if (!tuple_equal{}(last_key, key)) {
                    ++cnt;
                    last_key = key;
                }
                tag[idx] = cnt;
            }
            zeno::log_info("PrimMarkClose: collapse from {} to {}", prim->verts.size(), cnt + 1);
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkClose, {
    {
    {"PrimitiveObject", "prim"},
    {"float", "distance", "0.00001"},
    {"string", "tagAttr", "weld"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimMarkSameIf : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagValueIs = get_input2<int>("tagValueIs");
        auto tagAttrIn = get_input2<std::string>("tagAttrIn");
        auto tagAttrOut = get_input2<std::string>("tagAttrOut");
        auto const &tagArrIn = tagAttrIn != tagAttrOut ?
            prim->verts.attr<int>(tagAttrIn) :
            std::vector<int>(prim->verts.attr<int>(tagAttrIn));
        auto &tagArrOut = prim->verts.add_attr<int>(tagAttrOut);
        tagArrOut.resize(tagArrIn.size());
        int nout = 1;
        for (int i = 0; i < tagArrOut.size(); i++) {
            if (tagArrIn[i] == tagValueIs) {
                tagArrOut[i] = 0;
            } else {
                tagArrOut[i] = nout++;
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimMarkSameIf, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttrIn", "index"},
    {"int", "tagValueIs", "0"},
    {"string", "tagAttrOut", "weld"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});


struct PrimMarkIndex : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        int base = get_input<NumericObject>("base")->get<int>();
        int step = get_input<NumericObject>("step")->get<int>();
        auto type = get_input<StringObject>("type")->get();
        auto scope = get_input2<std::string>("scope");

        if (type == "float") {
            if (scope == "vert") {
                auto &tag = prim->verts.add_attr<float>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = float(base + i * step);
                });
            }
            else if (scope == "tri") {
                auto &tag = prim->tris.add_attr<float>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = float(base + i * step);
                });
            }
            else if (scope == "loop") {
                auto &tag = prim->loops.add_attr<float>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = float(base + i * step);
                });
            }
            else if (scope == "poly") {
                auto &tag = prim->polys.add_attr<float>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = float(base + i * step);
                });
            }
            else if (scope == "line") {
                auto &tag = prim->lines.add_attr<float>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = float(base + i * step);
                });
            }
        }
        else {
            if (scope == "vert") {
                auto &tag = prim->verts.add_attr<int>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = base + i * step;
                });
            }
            else if (scope == "tri") {
                auto &tag = prim->tris.add_attr<int>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = base + i * step;
                });
            }
            else if (scope == "loop") {
                auto &tag = prim->loops.add_attr<int>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = base + i * step;
                });
            }
            else if (scope == "poly") {
                auto &tag = prim->polys.add_attr<int>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = base + i * step;
                });
            }
            else if (scope == "line") {
                auto &tag = prim->lines.add_attr<int>(tagAttr);
                parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                    tag[i] = base + i * step;
                });
            }
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimMarkIndex, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "index"},
    {"enum int float", "type", "int"},
    {"int", "base", "0"},
    {"int", "step", "1"},
    {"enum vert tri loop poly line", "scope", "vert"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimCheckTagInRange : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input2<std::string>("tagAttr");
        int beg = get_input2<int>("beg");
        int end = get_input2<int>("end");
        int trueVal = get_input2<int>("trueVal");
        int falseVal = get_input2<int>("falseVal");
        int modularBy = get_input2<int>("modularBy");
        bool endExcluded = get_input2<bool>("endExcluded");
        if (endExcluded) end -= 1;

        auto &tag = prim->verts.attr<int>(tagAttr);
        if (modularBy <= 0) {
            parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                int t = tag[i];
                tag[i] = beg <= t && t <= end ? trueVal : falseVal;
            });
        } else {
            parallel_for((size_t)0, tag.size(), [&] (size_t i) {
                int t = tag[i];
                t = t < 0 ? -(-t % modularBy) : t % modularBy;
                tag[i] = beg <= t && t <= end ? trueVal : falseVal;
            });
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimCheckTagInRange, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "index"},
    {"int", "beg", "0"},
    {"int", "end", "0"},
    {"bool", "endExcluded", "0"},
    {"int", "modularBy", "0"},
    {"int", "trueVal", "1"},
    {"int", "falseVal", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
