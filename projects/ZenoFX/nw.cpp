#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>

namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

static void numeric_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<float *> const &chs
    ) {
    auto ctx = exec->make_context();
    for (int j = 0; j < chs.size(); j++) {
        ctx.channel(j)[0] = *chs[j];
    }
    ctx.execute();
    for (int j = 0; j < chs.size(); j++) {
        *chs[j] = ctx.channel(j)[0];
    }
}

struct NumericWrangle : zeno::INode {
    virtual void apply() override {
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        opts.detect_new_symbols = true;

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
            printf("define param: %s dim %d\n", key.c_str(), dim);
            opts.define_param(key, dim);
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        auto result = std::make_shared<zeno::DictObject>();
        for (auto const &[name, dim]: prog->newsyms) {
            printf("output numeric value: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            auto key = name.substr(1);
            zeno::NumericValue value;
            if (dim == 4) {
                value = zeno::vec4f{};
            } else if (dim == 3) {
                value = zeno::vec3f{};
            } else if (dim == 2) {
                value = zeno::vec2f{};
            } else if (dim == 1) {
                value = float{};
            } else {
                printf("ERROR: bad output dimension for numeric: %d\n", dim);
                abort();
            }
            result->lut[name] = std::make_shared<zeno::NumericObject>(value);
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<float *> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            printf("output %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            chs[i] = std::visit([&] (auto const &value) {
                    return (float *)(void *)&value;
            }, std::static_pointer_cast<zeno::NumericObject>(
                result->lut[name])->value);
        }

        numeric_wrangle(exec, chs);

        set_output("results", std::move(result));
    }
};

ZENDEFNODE(NumericWrangle, {
    {{"dict:numeric", "params"}, {"string", "zfxCode"}},
    {{"dict:numeric", "result"}},
    {},
    {"zenofx"},
});

}
