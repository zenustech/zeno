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

                printf("1\n");
        zfx::Options opts(zfx::Options::for_x64);
                printf("2\n");
        opts.detect_new_symbols = true;
                printf("3\n");
        prim->foreach_attr([&] (auto const &key, auto const &attr) {
                printf("4\n");
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
                printf("5\n");
            dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
                printf("6\n");
            opts.define_symbol('@' + key, dim);
                printf("7\n");
        });

                printf("8\n");
        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
                printf("9\n");
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            auto par = zeno::smart_any_cast<std::shared_ptr<zeno::NumericObject>>(obj).get();
            auto dim = std::visit([&] (auto const &v) {
                printf("10\n");
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
            dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
            opts.define_param(key, dim);
                printf("11\n");
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);
                printf("12\n");

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
                printf("13\n");

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
                printf("14\n");

        std::vector<Buffer> chs(prog->symbols.size());
                printf("14.0\n");
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
                printf("14.1\n");
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
                printf("14.2\n");
            assert(name[0] == '@');
            Buffer iob;
                printf("14.3 %s\n", name.c_str());
            prim->attr_visit(name.substr(1),
            [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
                printf("14.5\n");
            chs[i] = iob;
        }
                printf("15\n");
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

}
