#include <zeno/zeno.h>
#include <zeno/zfx.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ListObject.h>
#include <cassert>

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
};

static void vectors_wrangle
    ( zfx::Program const *prog
    , std::vector<Buffer> const &chs
    ) {
    if (chs.size() == 0)
        return;
    size_t size = chs[0].count;
    for (int i = 1; i < chs.size(); i++) {
        size = std::min(chs[i].count, size);
    }
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        zfx::Context ctx;
        for (int j = 0; j < chs.size(); j++) {
            ctx.memtable[j] = chs[j].base + chs[j].stride * i;
        }
        prog->execute(&ctx);
    }
}

struct ParticlesWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");

        auto code = get_input<zeno::StringObject>("zfxCode")->get();
        std::ostringstream oss;
        for (auto const &[key, attr]: prim->m_attrs) {
            oss << "define ";
            std::visit([&oss] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) oss << "f3";
                else if constexpr (std::is_same_v<T, float>) oss << "f1";
                else oss << "unknown";
            }, attr);
            oss << " @" << key << '\n';
        }

        auto params = get_input<zeno::ListObject>("params");
        std::vector<zeno::NumericValue> pars;
        for (auto const &obj: params->arr) {
            auto par = dynamic_cast<zeno::NumericObject *>(obj.get());
            oss << "define ";
            std::visit([&oss] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) oss << "f3";
                else if constexpr (std::is_same_v<T, float>) oss << "f1";
                else oss << "unknown";
            }, par->value);
            auto key = "arg0";
            oss << " " << key << '\n';
            pars.push_back(par->value);
        }

        code = oss.str() + code;
        auto prog = zfx::compile_program(code);

        std::vector<Buffer> chs(prog->channels.size());
        for (int i = 0; i < chs.size(); i++) {
            auto chan = zfx::split_str(prog->channels[i], '.');
            assert(chan.size() == 2);
            int dimid = 0;
            std::stringstream(chan[1]) >> dimid;
            Buffer iob;
            auto const &attr = prim->attr(chan[0]);
            std::visit([&] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            chs[i] = iob;
        }
        vectors_wrangle(prog, chs);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"prim", "zfxCode", "params"},
    {"prim"},
    {},
    {"zenofx"},
});
