//
// Created by admin on 2022/6/22.
//
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GlobalState.h>

namespace {


    struct StringEval : zeno::INode {
        virtual void apply() override {
            auto code = get_input2<std::string>("zfxCode");
            for (int i = 0; i < code.size(); i++) {
                if (code[i] == '$' && code[i+1] == 'F') {
                    code.replace(i, 2, std::to_string(getGlobalState()->frameid));
                } else if (code == '$' && code[i+1] == 'N') {
                    //这里把$N替换成啥
                  //  code.replace(i, 2, std::to_string())
                } else {
                    continue;
                }
            }
            set_output2("code", code);
        }
    };

    ZENDEFNODE(StringEval, {
                               /*input*/
                            {{"string", "zfxCode"}},
                            {{"string", "code"}},
                            {},
                            {"zenofx"}
                           });
}
