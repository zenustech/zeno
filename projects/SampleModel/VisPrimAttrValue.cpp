#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <sstream>

namespace zeno {
namespace {

template <class T>
struct number_printer {
    void operator()(std::ostringstream &ss, T const &value) {
        ss << value;
    }
};

template <size_t N, class T>
struct number_printer<vec<N, T>> {
    void operator()(std::ostringstream &ss, vec<N, T> const &value) {
        ss << value[0];
        for (size_t i = 1; i < N; i++)
            ss << ',' << value[i];
    }
};

struct VisPrimAttrValue : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto attrName = get_input2<std::string>("attrName");
        auto scale = get_input2<float>("scale");
        auto precision = get_input2<int>("precision");
        auto includeSelf = get_input2<bool>("includeSelf");
        auto dotDecoration = get_input2<bool>("dotDecoration");
        bool textDecoration = !attrName.empty();

        std::vector<PrimitiveObject *> outprim2;
        std::vector<std::shared_ptr<PrimitiveObject>> outprim;
        if (textDecoration) {
            prim->verts.attr_visit<AttrAcceptAll>(attrName, [&] (auto const &attarr) {
                outprim.resize(attarr.size());
                outprim2.resize(attarr.size());
#pragma omp parallel for
                for (int i = 0; i < attarr.size(); i++) {
                    auto value = attarr[i];
                    auto pos = prim->verts[i];
                    std::ostringstream ss;
                    ss << std::setprecision(precision);
                    number_printer<std::decay_t<decltype(value)>>{}(ss, value);
                    auto str = ss.str();

                    auto numprim = std::static_pointer_cast<PrimitiveObject>(
                        getThisGraph()->callTempNode("LoadStringPrim", {
                            {"triangulate", std::make_shared<NumericObject>((bool)0)},
                            {"decodeUVs", std::make_shared<NumericObject>((bool)0)},
                            {"str", objectFromLiterial(str)},
                        }).at("prim"));
                    //auto numprim = std::make_shared<PrimitiveObject>();
                    for (int j = 0; j < numprim->verts.size(); j++) {
                        auto &v = numprim->verts[j];
                        v = (v + vec3f(dotDecoration ? 0.5f : 0.3f, -0.3f, 0.0f)) * scale + pos;
                    }
                    outprim2[i] = numprim.get();
                    outprim[i] = std::move(numprim);
                }
            });
        }
        if (dotDecoration) {
            int attarrsize = textDecoration ? outprim.size() : prim->verts.size();
            outprim.resize(attarrsize * (1 + (int)textDecoration));
            outprim2.resize(attarrsize * (1 + (int)textDecoration));
            auto numprim = std::static_pointer_cast<PrimitiveObject>(
                getThisGraph()->callTempNode("LoadSampleModel", {
                    {"triangulate", std::make_shared<NumericObject>((bool)0)},
                    {"decodeUVs", std::make_shared<NumericObject>((bool)0)},
                    {"name", objectFromLiterial("star")},
                }).at("prim"));
#pragma omp parallel for
            for (int i = 0; i < attarrsize; i++) {
                auto pos = prim->verts[i];
                auto offprim = std::make_shared<PrimitiveObject>(*numprim);
                for (int j = 0; j < offprim->verts.size(); j++) {
                    auto &v = offprim->verts[j];
                    v = v * (scale * 0.25f) + pos;
                }
                outprim2[i + attarrsize * (int)textDecoration] = offprim.get();
                outprim[i + attarrsize * (int)textDecoration] = std::move(offprim);
            }
        }
        if (includeSelf) {
            outprim2.push_back(prim.get());
        }

        auto retprim = primMerge(outprim2);
        set_output("outPrim", std::move(retprim));
    }
};

ZENO_DEFNODE(VisPrimAttrValue)({
    {
        {"prim"},
        {"string", "attrName", "pos"},
        {"float", "scale", "0.05"},
        {"int", "precision", "3"},
        {"bool", "includeSelf", "0"},
        {"bool", "dotDecoration", "1"},
    },
    {
        {"outPrim"},
    },
    {},
    {"visualize"},
});

}
}
