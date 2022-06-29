#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"

namespace {
    using namespace zeno;

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

static void numeric_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<float> &chs
    ) {
    auto ctx = exec->make_context();
    for (int j = 0; j < chs.size(); j++) {
        ctx.channel(j)[0] = chs[j];
    }
    ctx.execute();
    for (int j = 0; j < chs.size(); j++) {
        chs[j] = ctx.channel(j)[0];
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
        {
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        auto const &gs = *this->getGlobalState();
        params->lut["F"] = objectFromLiterial((float)gs.frameid);
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
            auto par = zeno::objectToLiterial<zeno::NumericValue>(obj);
            auto dim = std::visit([&] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_convertible_v<T, zeno::vec3f>) {
                    parvals.push_back(v[0]);
                    parvals.push_back(v[1]);
                    parvals.push_back(v[2]);
                    parnames.emplace_back(key, 0);
                    parnames.emplace_back(key, 1);
                    parnames.emplace_back(key, 2);
                    return 3;
                } else if constexpr (std::is_convertible_v<T, float>) {
                    parvals.push_back(float(v));
                    parnames.emplace_back(key, 0);
                    return 1;
                } else return 0;
            }, par);
            dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
            opts.define_param(key, dim);
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        auto result = std::make_shared<zeno::DictObject>();
        for (auto const &[name, dim]: prog->newsyms) {
            dbg_printf("output numeric value: %s with dim %d\n",
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
                err_printf("ERROR: bad output dimension for numeric: %d\n", dim);
            }
            result->lut[key] = std::make_shared<zeno::NumericObject>(value);
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

        std::vector<float> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("output %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
        }

        numeric_wrangle(exec, chs);

        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            float value = chs[i];
            dbg_printf("output %d: %s.%d = %f\n", i, name.c_str(), dimid, value);
            auto key = name.substr(1);
            std::visit([dimid = dimid, value] (auto &res) {
                    dimid[(float *)(void *)&res] = value;
            }, zeno::safe_dynamic_cast<zeno::NumericObject>(result->lut[key])->value);
        }

        set_output("result", std::move(result));
    }
};

ZENDEFNODE(NumericWrangle, {
    {{"DictObject:NumericObject", "params"}, {"string", "zfxCode"}},
    {{"DictObject:NumericObject", "result"}},
    {},
    {"zenofx"},
});

}
