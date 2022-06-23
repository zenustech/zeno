#if 0
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"

namespace zeno {
namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 1;
    int which = 0;
};

static void vectors_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<Buffer> const &chs
    , std::vector<zeno::vec2i> const &edges
    ) {
    if (chs.size() == 0)
        return;

    //#pragma omp parallel for
    for (int i = 0; i < edges.size(); i++) {
        int uv[3] = {i, edges[i][0], edges[i][1]};
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

struct PrimitiveEdgeWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto edgePrim = get_input<zeno::PrimitiveObject>("edgePrim");
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
            dbg_printf("define symbol: @1%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@1" + key, dim);
            dbg_printf("define symbol: @2%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@2" + key, dim);
        });
        prim->foreach_attr([&] (auto const &key, auto const &attr) {
        for (auto const &[key, attr]: edgePrim->m_attrs) {
            int dim = std::visit([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            }, attr);
            dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        }

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        {
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        auto const &gs = *this->getGlobalState();
        params->lut["F"] = objectFromLiterial(gs.frameid);
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
                } else return 0;
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
            std::string key = name.substr(1);
            auto *primPtr = edgePrim.get();
            if ('1' <= key[0] && key[0] <= '9') {
                key = key.substr(1);
                primPtr = prim.get();
            }
            if (dim == 3) {
                primPtr->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                primPtr->add_attr<float>(key);
            } else {
                dbg_printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
        }

        edgePrim->resize(prim->lines.size());

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
            zeno::PrimitiveObject *primPtr;
            if ('1' <= name[1] && name[1] <= '9') {
                iob.which = name[1] - '0';
                name = name.substr(2);
                primPtr = prim.get();
            } else {
                iob.which = 0;
                name = name.substr(1);
                primPtr = edgePrim.get();
            }
            auto const &attr = primPtr->attr(name);
            std::visit([&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            chs[i] = iob;
        }

        vectors_wrangle(exec, chs, prim->lines);

        set_output("prim", std::move(prim));
        set_output("edgePrim", std::move(edgePrim));
    }
};

ZENDEFNODE(PrimitiveEdgeWrangle, {
    {{"PrimitiveObject", "prim"}, {"PrimitiveObject", "edgePrim"},
     {"string", "zfxCode"}, {"DictObject:NumericObject", "params"}},
    {{"PrimitiveObject", "prim"}, {"PrimitiveObject", "edgePrim"}},
    {},
    {"zenofx"},
});

}
}
#endif
