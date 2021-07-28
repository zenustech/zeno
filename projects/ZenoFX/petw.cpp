#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>

namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 1;
};

/*struct EdgeList {
    std::vector<std::vector<int>> edgelist;

    void build(std::vector<zeno::vec2i> const &edges) {
        for (int i = 0; i < edges.size(); i++) {
            auto uv = edges[i];
            auto size = std::max(uv[0], uv[1]) + 1;
            if (size > edgelist.size())
                edgelist.resize(size);
            edgelist[uv[0]].push_back(uv[1]);
            edgelist[uv[1]].push_back(uv[0]);
        }
    }

    std::vector<int> const &neighbors_of(int i) const {
        return edgelist[i];
    }
};*/

static void vectors_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<Buffer> const &chs
    , std::vector<zeno::vec2i> const &edges
    ) {
    if (chs.size() == 0)
        return;

    #pragma omp parallel for
    for (int i = 0; i < edges.size(); i++) {
        auto uv = edges[i];
        auto ctx = exec->make_context();
        for (int k = 0; k < chs.size(); k++) {
            ctx.channel(k)[0] = chs[k].base[chs[k].stride * uv[chs[k].which]];
        }
        ctx.execute();
        for (int k = 0; k < chs.size(); k++) {
            chs[k].base[chs[k].stride * uv[chs[k].which]] = ctx.channel(k)[0];
        }
    }
}

struct PrimitiveEdgeTopologyWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        opts.detect_new_symbols = true;
        for (auto const &[key, attr]: prim->m_attrs) {
            int dim = std::visit([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            }, attr);
            printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
            printf("define symbol: @@%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@@" + key, dim);
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

        for (auto const &[name, dim]: prog->newsyms) {
            printf("auto-defined new attribute: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            std::string key = name.substr(1);
            if (key[0] == '@') {
                key = key.substr(1);
            }
            if (dim == 3) {
                prim->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                prim->add_attr<float>(key);
            } else {
                printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
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

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            auto const &attr = prim->attr(name);
            std::visit([&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            chs[i] = iob;
        }

        vectors_wrangle(exec, chs);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveEdgeTopologyWrangle, {
    {{"primitive", "prim"},
     {"string", "zfxCode"}, {"dict:numeric", "params"}},
    {{"primitive", "prim"}},
    {},
    {"zenofx"},
});

}
