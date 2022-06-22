//
// Created by admin on 2022/6/22.
//
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/assetDir.h>
#include <sstream>
#include <iomanip>

namespace zeno {
namespace {


    //
    // $F       current frame number
    // $NASLOC  the NAS path set from UI
    //
    // $F    0 1 2 3 10 20 30 100 200 300
    // $FF   00 01 02 03 10 20 30 100 200 300
    // $FFF  000 001 002 003 010 020 030 100 200 300
    //
    // for example:
    //   $NASLOC/out$FFFFFF.obj
    // will get:
    //   Z:/ZenusTech/Models/out000042.obj
    //
    struct StringEval : zeno::INode {
        virtual void apply() override {
            auto code = get_input2<std::string>("zfxCode");


            std::size_t pos0 = 0;
            while (1) {
                if (auto pos = code.find("$F", pos0); pos != std::string::npos) {
                    auto oldpos0 = pos0;
                    std::ostringstream oss;
                    pos0 = pos;
                    int w = 1;
                    while (code.size() > pos0 + 1 && code[pos0] == 'F') {
                        ++pos0;
                        ++w;
                    }
                    if (w != 1) {
                        oss << std::setfill('0') << std::setw(w);
                    }
                    oss << getGlobalState()->frameid;
                    code.replace(oldpos0, pos0 - oldpos0, oss.str());
                } else if (auto pos = code.find("$NASLOC", pos0); pos != std::string::npos) {
                    auto nasloc = zeno::getConfigVariable("NASLOC");
                    code.replace(pos0, pos - pos0, nasloc);
                    pos0 = pos;
                } else {
                    break;
                }
            }

            //for (int i = 0; i < code.size(); i++) {
                //if (code[i] == '$' && code[i+1] == 'F') {
                    //code.replace(i, 2, std::to_string(getGlobalState()->frameid));
                //} else if (code == '$' && code[i+1] == 'N') {
                    ////这里把$N替换成啥
                  ////  code.replace(i, 2, std::to_string())
                //} else {
                    //continue;
                //}
            //}
            set_output2("result", std::move(code));
        }
    };

    ZENDEFNODE(StringEval, {
                               /*input*/
                            {{"string", "zfxCode"}},
                            {{"string", "result"}},
                            {},
                            {"zenofx"}
                           });
}
}
