//
// Created by admin on 2022/6/15.
//

#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>

#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include <vector>
#include "dbg_printf.h"

namespace {
    static zfx::Compiler compiler;
    static zfx::x64::Assembler assembler;

    static void numeric_eval (zfx::x64::Executable *exec,
                             std::vector<float> &chs) {
            auto ctx = exec->make_context();
            for (int j = 0; j < chs.size(); j++) {
                ctx.channel(j)[0] = chs[j];
            }
            ctx.execute();
            for (int j = 0; j < chs.size(); j++) {
                chs[j] = ctx.channel(j)[0];
            }

    }

    struct NumericEval : zeno::INode {
        virtual void apply() {
            auto code = get_input<zeno::StringObject>("zfxCode")->get();
//一个模板函数，返回一个std::shared_ptr<T>，这里对这一个智能指针调用get()返回一个裸指针
            zfx::Options opts(zfx::Options::for_x64);
            opts.detect_new_symbols = true;

            //接收参数
            auto params = has_input("params") ? get_input<zeno::DictObject>("params") :
                std::make_shared<zeno::DictObject>();

            std::vector<float> parvals;//存储参数值
            std::vector<std::pair<string, int>> parnames;
            for (auto const &[key_, obj] : params->lut) {
                //lut是DictObject中的一个map<std::string, zany>
                //zany是std::shared_ptr<IObject>的别名
                auto key = '$' + key_;
                auto par = zeno::objectToLiterial<zeno::NumericValue>(obj);
                //par 是一个NumericValue
                //ObjectToLiterial是一个模板函数由两个重载一个返回bool， 一个返回T
                //获取参数的维数
                auto dim = std::visit([&](auto const &v){
                    using T = std::decay_t<decltype(v)>;
                    //判断参数是三维数组还是，单浮点数
                    if constexpr(std::is_same_v(T, zeno::vec3f)) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        paranames.emplace_back(key, 1);
                        paranames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr(std::is_constructible_v<T, float>) {
                        parvals.push_back(float(v));
                        paranames(emplace_back(key, 0));
                        return 1;
                    } else return 0;
                }, par);
                dbg_print("define param : %s dim %d\n, key.c_str(), dim");
                opts.define_param(key, dim);
            }

            //开始编译
            auto prog = compiler.compile(code, opts);
            auto exec = assembler.assemble(prog->assembly);

            //计算输出结果
            auto result = std::make_shared<zeno::NumericObject>();
            for (auto const &[name, dim] : prog->newsyms) {
                dbg_printf("output numeric value %s with dim %d\n", name.c_str(), dim);
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
                    dbg_printf("ERROR : bad output dimension for numeric : %d\n", dim);
                    abort();
                }
            }
            result->set(value);
            //result->lut[key] = std::make_shared<zeno::NumericObject>(value);
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            dbg_printf("parameter %d: %s.%d\n", i , name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(), paranames.end(), std::pair{name , dimid});
            auto value = parvals.at(it - paranames.begin());
            dbg_printf("(value %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<float> chs(prog->symbols.size());//初始化chs的大小
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("output %d : %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
        }

        numeric_eval(exec, chs);

        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            float value = chs[i];
            dbg_printf("output %d : %s. %d = %f\n", i , name.c_str(), dimid, value);
            auto key = name.substr(1);
            std::visit([dimid = dimid, value] (auto &res) {
                dimid[(float*)(void*)&res] = value;
            }, result->get())
        }
        set_output("result", std::move(result));
    };

    ZENDEFNODE(NumericEval, {
    /* inputs*/
        {
          {"DictObject:NumericObject", "params"}, {"string", "zfxCode"}
        },
    /*OutPut*/
        {
            {"NumericObject", "result"}
        },
        {},
        {"zenofx"},
    });
}
