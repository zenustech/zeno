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

        #pragma omp parallel for
        for(int i = 0; i < parspos.size(); i++) {
            for (int j = 0; j < meshpos.size(); j++) {
                outmpos[i * meshpos.size() + j] = parspos[i] + uniScale * meshpos[j];
            }
        }

        for (auto const &[key, attr]: mesh->m_attrs) {
            if (key == "pos") continue;
            std::visit([&] (auto const &attr) {
                using T = std::decay_t<decltype(attr[0])>;
                auto &outattr = outm->add_attr<T>(key);
                #pragma omp parallel for
                for(int i = 0; i < pars->size(); i++) {
                    for (int j = 0; j < attr.size(); j++) {
                        outattr[i * attr.size() + j] = attr[j];
                    }
                }
            }, attr);
        }

        outm->points.resize(pars->size() * mesh->points.size());
        #pragma omp parallel for
        for(int i = 0; i < pars->size(); i++) {
            for (int j = 0; j < mesh->points.size(); j++) {
                outm->points[i * mesh->points.size() + j]
                    = mesh->points[j] + i * meshpos.size();
            }
        }

        outm->lines.resize(pars->size() * mesh->lines.size());
        #pragma omp parallel for
        for(int i = 0; i < pars->size(); i++) {
            for (int j = 0; j < mesh->lines.size(); j++) {
                outm->lines[i * mesh->lines.size() + j]
                    = mesh->lines[j] + i * meshpos.size();
            }
        }

        outm->tris.resize(pars->size() * mesh->tris.size());
        #pragma omp parallel for
        for(int i = 0; i < pars->size(); i++) {
            for (int j = 0; j < mesh->tris.size(); j++) {
                outm->tris[i * mesh->tris.size() + j]
                    = mesh->tris[j] + i * meshpos.size();
            }
        }

        outm->quads.resize(pars->size() * mesh->quads.size());
        #pragma omp parallel for
        for(int i = 0; i < pars->size(); i++) {
            for (int j = 0; j < mesh->quads.size(); j++) {
                outm->quads[i * mesh->quads.size() + j]
                    = mesh->quads[j] + i * meshpos.size();
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
        "outPrim",
        }, {
        }, {
        "primitive",
        }});


}
