#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zeno/VDBGrid.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"
#include <zeno/StringObject.h>

namespace zeno {
namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};



template <class GridPtr>
void vdb_wrangle(zfx::x64::Executable *exec, GridPtr &grid, bool modifyActive) {
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    zeno::boolean_switch(modifyActive, [&] (auto modifyActive) {
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.modifyValue([&](auto &v) {
                auto ctx = exec->make_context();
                if constexpr (std::is_same_v<std::decay_t<decltype(v)>, openvdb::Vec3f>) {
                    ctx.channel(0)[0] = v[0];
                    ctx.channel(1)[0] = v[1];
                    ctx.channel(2)[0] = v[2];
                    ctx.execute();
                    v[0] = ctx.channel(0)[0];
                    v[1] = ctx.channel(1)[0];
                    v[2] = ctx.channel(2)[0];
    
                } else {
                    ctx.channel(0)[0] = v;
                    ctx.execute();
                    v = ctx.channel(0)[0];
                    
                }
                
            });

            if constexpr(modifyActive.value){
                float testv;
                auto v = iter.getValue();
                if constexpr (std::is_same_v<std::decay_t<decltype(v)>, openvdb::Vec3f>) {
                    testv = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
                } else {
                    testv = std::abs(v);
                }
                iter.setValueOn(testv<1e-5);
            }
        }
    };
    velman.foreach(wrangler);
    });
    openvdb::tools::prune(grid->tree());
}

struct VDBWrangle : zeno::INode {
    virtual void apply() override {
        auto grid = get_input<zeno::VDBGrid>("grid");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        if (std::dynamic_pointer_cast<zeno::VDBFloatGrid>(grid))
            opts.define_symbol("@val", 1);
        else if (std::dynamic_pointer_cast<zeno::VDBFloat3Grid>(grid))
            opts.define_symbol("@val", 3);
        else
            dbg_printf("unexpected vdb grid type");
        opts.reassign_channels = false;

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        {
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        auto const &gs = *this->getGlobalState();
        params->lut["F"] = objectFromLiterial((float)gs.frameid);
        params->lut["DT"] = objectFromLiterial(gs.frame_time);
        params->lut["T"] = objectFromLiterial(gs.frame_time * gs.frameid + gs.frame_time_elapsed);
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        for (auto const &[key, ref]: getThisGraph()->portalIns) {
            if (auto i = code.find('$' + key); i != std::string::npos) {
                i = i + key.size() + 1;
                if (code.size() <= i || !std::isalnum(code[i])) {
                    dbg_printf("ref portal %s\n", key.c_str());
                    auto res = getThisGraph()->callTempNode("PortalOut",
                          {{"name:", objectFromLiterial(key)}}).at("port");
                    params->lut[key] = std::move(res);
                }
            }
        }
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        }
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, par]: params->getLiterial<zeno::NumericValue>()) {
            auto key = '$' + key_;
            auto dim = std::visit([&] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_convertible_v<T, zeno::vec3f>) {
                    parvals.push_back(v[0]);
                    parvals.push_back(v[1]);
                    parvals.push_back(v[2]);
                    parnames.emplace_back(key, 0);
                    parnames.emplace_back(key, 1);
                    parnames.emplace_back(key, 2);
                    return 3;
                } else if constexpr (std::is_convertible_v<T, float>) {
                    parvals.push_back(v);
                    parnames.emplace_back(key, 0);
                    return 1;
                } else return 0;
            }, par);
            opts.define_param(key, dim);
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        std::vector<float> pars(prog->params.size());
        for (int i = 0; i < pars.size(); i++) {
            auto [name, dimid] = prog->params[i];
            assert(name[0] == '$');
            dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            dbg_printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }
        auto modifyActive = (get_input<zeno::StringObject>("ModifyActive")->get())=="true";
        if (auto p = std::dynamic_pointer_cast<zeno::VDBFloatGrid>(grid); p)
            vdb_wrangle(exec, p->m_grid, modifyActive);
        else if (auto p = std::dynamic_pointer_cast<zeno::VDBFloat3Grid>(grid); p)
            vdb_wrangle(exec, p->m_grid, modifyActive);

        set_output("grid", std::move(grid));
    }
};

ZENDEFNODE(VDBWrangle, {
    {{"VDBGrid", "grid"}, {"string", "zfxCode"},{"enum true false","ModifyActive","false"},
     {"DictObject:NumericObject", "params"}},
    {{"VDBGrid", "grid"}},
    {},
    {"zenofx"},
});

}
}
