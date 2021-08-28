#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {


struct PrimitiveDuplicate : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<PrimitiveObject>("meshPrim");
        auto pars = get_input<PrimitiveObject>("particlesPrim");

        auto outm = std::make_shared<PrimitiveObject>();
        outm->resize(pars->size() * mesh->size());

        float uniScale = has_input("uniScale") ?
            get_input<NumericObject>("uniScale")->get<float>() : 1.0f;

        auto const &parspos = pars->attr<vec3f>("pos");
        auto const &meshpos = mesh->attr<vec3f>("pos");
        auto &outmpos = outm->add_attr<vec3f>("pos");

        for(int i = 0; i < parspos.size(); i++) {
            for (int j = 0; j < meshpos.size(); j++) {
                outmpos[i * parspos.size() + j] = parspos[j] + uniScale * meshpos[j];
            }
        }
        set_output("outPrim", std::move(outm));
    }
};


ZENDEFNODE(PrimitiveDuplicate, {
        {
        "meshPrim",
        "particlesPrim",
        "uniScale",
        }, {
        }, {
        }, {
        "primitive",
        }});


}
