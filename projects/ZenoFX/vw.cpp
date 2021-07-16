#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
#include <zeno/VDBGrid.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};

template <class Tree>
void vdb_wrangle(zfx::x64::Executable *exec, Tree &tree) {
    auto wrangler = [&](openvdb::Vec3fTree::LeafNodeType &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.modifyValue([&](openvdb::Vec3f &v) {
                auto ctx = exec->make_context();
                ctx.channel(0)[0] = v[0];
                ctx.channel(1)[0] = v[1];
                ctx.channel(2)[0] = v[2];
                ctx.execute();
                v[0] = ctx.channel(0)[0];
                v[1] = ctx.channel(1)[0];
                v[2] = ctx.channel(2)[0];
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<Tree>(tree);
    velman.foreach(wrangler);
}

struct VDBWrangle : zeno::INode {
    virtual void apply() override {
        auto grid = get_input<zeno::VDBGrid>("grid");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        if (dynamic_cast<zeno::VDBFloatGrid *>(grid.get()))
            opts.define_symbol("@val", 1);
        else if (dynamic_cast<zeno::VDBFloat3Grid *>(grid.get()))
            opts.define_symbol("@val", 3);
        else
            printf("unexpected vdb grid type");
        opts.reassign_channels = false;

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            auto par = dynamic_cast<zeno::NumericObject *>(obj.get());
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
            }, par->value);
            opts.define_param(key, dim);
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        std::vector<float> pars(prog->params.size());
        for (int i = 0; i < pars.size(); i++) {
            auto [name, dimid] = prog->params[i];
            assert(name[0] == '$');
            printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        if (auto p = dynamic_cast<zeno::VDBFloatGrid *>(grid.get()))
            vdb_wrangle(exec, p->m_grid->tree());
        else if (auto p = dynamic_cast<zeno::VDBFloat3Grid *>(grid.get()))
            vdb_wrangle(exec, p->m_grid->tree());

        set_output("grid", std::move(grid));
    }
};

ZENDEFNODE(VDBWrangle, {
    {"grid", "zfxCode", "params"},
    {"grid"},
    {},
    {"zenofx"},
});
