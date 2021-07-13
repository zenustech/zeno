#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ListObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>

static zfx::Compiler<zfx::x64::Program> compiler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};

static void vectors_wrangle
    ( zfx::Program<zfx::x64::Program> *prog
    , std::vector<Buffer> const &chs
    //, std::vector<float> const &pars
    ) {
    if (chs.size() == 0)
        return;
    size_t size = chs[0].count;
    for (int i = 1; i < chs.size(); i++) {
        size = std::min(chs[i].count, size);
    }
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto ctx = prog->make_context();
        /*for (int j = 0; j < pars.size(); j++) {
            ctx.regtable[j] = pars[j];
        }*/
        for (int j = 0; j < chs.size(); j++) {
            *ctx.pointer(j) = chs[j].base[chs[j].stride * i];
        }
        ctx.execute();
        for (int j = 0; j < chs.size(); j++) {
            chs[j].base[chs[j].stride * i] = *ctx.pointer(j);
        }
    }
}

struct ParticlesWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts;
        for (auto const &[key, attr]: prim->m_attrs) {
            int dim = std::visit([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            }, attr);
            opts.define_symbol('@' + key, dim);
        }

        /*auto params = get_input<zeno::ListObject>("params");
        std::vector<float> pars;
        std::vector<std::string> parnames;
        for (int i = 0; i < params->arr.size(); i++) {
            auto const &obj = params->arr[i];
            std::ostringstream keyss; keyss << "arg" << i;
            auto key = keyss.str();
            auto par = dynamic_cast<zeno::NumericObject *>(obj.get());
            oss << "define ";
            std::visit([&] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) {
                    oss << "f3";
                    pars.push_back(v[0]);
                    pars.push_back(v[1]);
                    pars.push_back(v[2]);
                    parnames.push_back(key + ".0");
                    parnames.push_back(key + ".1");
                    parnames.push_back(key + ".2");
                } else if constexpr (std::is_same_v<T, float>) {
                    oss << "f1";
                    pars.push_back(v);
                    parnames.push_back(key + ".0");
                } else oss << "unknown";
            }, par->value);
            oss << " " << key << '\n';
        }
        for (auto const &par: parnames) {
            oss << "parname " << par << '\n';
        }*/

        auto prog = compiler.compile(code, opts);

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
        vectors_wrangle(prog, chs);//, pars);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"prim", "zfxCode", "params"},
    {"prim"},
    {},
    {"zenofx"},
});
