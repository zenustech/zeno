#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/VDBGrid.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"

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
void vdb_wrangle(zfx::x64::Executable *exec, GridPtr &grid) {
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
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);
}

struct VDBWrangle : zeno::INode {
    virtual void apply() override {
        auto grid = get_input<zeno::VDBGrid>("grid");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        if (zeno::silent_any_cast<std::shared_ptr<zeno::VDBFloatGrid>>(grid).has_value())
            opts.define_symbol("@val", 1);
        else if (zeno::silent_any_cast<std::shared_ptr<zeno::VDBFloat3Grid>>(grid).has_value())
            opts.define_symbol("@val", 3);
        else
            dbg_printf("unexpected vdb grid type");
        opts.reassign_channels = false;

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, par]: params->getLiterial<zeno::NumericValue>()) {
            auto key = '$' + key_;
            auto dim = std::visit([&] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) {
                    parvals.push_back(v[0]);
                    parvals.push_back(v[1]);
                    parvals.push_back(v[2]);
                    parnames.emplace_back(key, 0);
                    parnames.emplace_back(key, 1);
                    parnames.emplace_back(key, 2);
                    return 3;
                } else if constexpr (std::is_same_v<T, float>) {
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

        if (auto p = zeno::silent_any_cast<std::shared_ptr<zeno::VDBFloatGrid>>(grid); p.has_value())
            vdb_wrangle(exec, p.value()->m_grid);
        else if (auto p = zeno::silent_any_cast<std::shared_ptr<zeno::VDBFloat3Grid>>(grid); p.has_value())
            vdb_wrangle(exec, p.value()->m_grid);

        set_output("grid", std::move(grid));
    }
};

ZENDEFNODE(VDBWrangle, {
    {{"VDBGrid", "grid"}, {"string", "zfxCode"},
     {"DictObject:NumericObject", "params"}},
    {{"VDBGrid", "grid"}},
    {},
    {"zenofx"},
});

}
