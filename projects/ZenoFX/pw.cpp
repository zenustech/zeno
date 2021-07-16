#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
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
        for (auto const &[key, attr]: prim->m_attrs) {
            int dim = std::visit([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            }, attr);
            printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        }

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

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            auto const &attr = prim->attr(name.substr(1));
            std::visit([&] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            chs[i] = iob;
        }
        vectors_wrangle(exec, chs);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"prim", "zfxCode", "params"},
    {"prim"},
    {},
    {"zenofx"},
});
