#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
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
};

static void vectors_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<Buffer> const &chs
    ) {
    if (chs.size() == 0)
        return;
    size_t size = chs[0].count;
    for (int i = 1; i < chs.size(); i++) {
        size = std::min(chs[i].count, size);
    }

    #pragma omp parallel for
    for (int i = 0; i < size - exec->SimdWidth + 1; i += exec->SimdWidth) {
        auto ctx = exec->make_context();
        for (int j = 0; j < chs.size(); j++) {
            for (int k = 0; k < exec->SimdWidth; k++)
                ctx.channel(j)[k] = chs[j].base[chs[j].stride * (i + k)];
        }
        ctx.execute();
        for (int j = 0; j < chs.size(); j++) {
            for (int k = 0; k < exec->SimdWidth; k++)
                 chs[j].base[chs[j].stride * (i + k)] = ctx.channel(j)[k];
        }
    }
    for (int i = size / exec->SimdWidth * exec->SimdWidth; i < size; i++) {
        auto ctx = exec->make_context();
        for (int j = 0; j < chs.size(); j++) {
            ctx.channel(j)[0] = chs[j].base[chs[j].stride * i];
        }
        ctx.execute();
        for (int j = 0; j < chs.size(); j++) {
            chs[j].base[chs[j].stride * i] = ctx.channel(j)[0];
        }
    }
}

struct ParticlesWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        opts.detect_new_symbols = true;
        prim->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        });

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            auto par = zeno::safe_any_cast<zeno::NumericValue>(obj);
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
                } else {
                    printf("invalid parameter type encountered: `%s`\n",
                            typeid(T).name());
                    return 0;
                }
            }, par);
            dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
            opts.define_param(key, dim);
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        for (auto const &[name, dim]: prog->newsyms) {
            dbg_printf("auto-defined new attribute: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            auto key = name.substr(1);
            if (dim == 3) {
                prim->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                prim->add_attr<float>(key);
            } else {
                dbg_printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            dbg_printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            prim->attr_visit(name.substr(1),
            [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs[i] = iob;
        }
        vectors_wrangle(exec, chs);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {{"PrimitiveObject", "prim"},
     {"string", "zfxCode"}, {"DictObject:NumericObject", "params"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"zenofx"},
});


struct SyncPrimitiveAttributes : zeno::INode {
    virtual void apply() override {
        auto prim1 = get_input<zeno::PrimitiveObject>("prim1");
        auto prim2 = get_input<zeno::PrimitiveObject>("prim2");

        prim1->verts.foreach_attr([&] (auto const &key, auto const &attr) {
            using T = std::decay_t<decltype(attr[0])>;
            prim2->add_attr<T>(key);
        });

        prim2->verts.foreach_attr([&] (auto const &key, auto const &attr) {
            using T = std::decay_t<decltype(attr[0])>;
            prim1->add_attr<T>(key);
        });

        // prim1->resize(prim1->size());
        // prim2->resize(prim2->size());

        set_output("prim1",prim1);
        set_output("prim2",prim2);
    }
};

ZENDEFNODE(SyncPrimitiveAttributes, {
    {"prim1", "prim2"},
    {"prim1", "prim2"},
    {},
    {"zenofx"},
});


}
