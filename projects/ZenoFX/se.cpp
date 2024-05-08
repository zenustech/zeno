//
// Created by admin on 2022/6/22.
//
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/TempNode.h>
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
    // {<NumericEval expression> : <precison>}
    //
    // {$F + 42}        42 43 44 45 46 47
    // {$F + 42 : 3}    042 043 044 045 046 047
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
            while (1) if (auto pos = code.find('{', pos0); pos != std::string::npos) {
                auto pos2 = code.find('}', pos + 1);
                if (pos2 == std::string::npos)
                    continue;
                auto necode = code.substr(pos + 1, pos2 - pos - 1);
                int w = 1;
                if (auto nepos = necode.find(':'); nepos != std::string::npos) {
                    w = std::stoi(necode.substr(nepos + 1));
                    necode = necode.substr(0, nepos);
                }
                int val = std::rint(temp_node("NumericEval")
                    .set2("zfxCode", necode)
                    .set2("resType", "float")
                    .get2<float>("result"));
                std::ostringstream oss;
                if (w > 1) {
                    oss << std::setfill('0') << std::setw(w);
                }
                oss << val;
                auto ost = oss.str();
                code.replace(pos, pos2 + 1 - pos, ost);
                pos0 = pos + ost.size();
            } else break;

            pos0 = 0;
            while (1) if (auto pos = code.find("$FPS", pos0); pos != std::string::npos) {
                auto fps = zeno::getConfigVariable("FPS");
                code.replace(pos, 4, fps);
                pos0 = pos + 4;
            }
            else break;

            pos0 = 0;
            while (1) if (auto pos = code.find("$F", pos0); pos != std::string::npos) {
                std::ostringstream oss;
                pos0 = pos + 2;
                int w = 1;
                while (code.size() > pos0 + 1 && code[pos0] == 'F') {
                    ++pos0;
                    ++w;
                }
                if (w != 1) {
                    oss << std::setfill('0') << std::setw(w);
                }
                oss << getGlobalState()->getFrameId();
                code.replace(pos, 2 + w - 1, oss.str());
            } else break;

            pos0 = 0;
            while (1) if (auto pos = code.find("$NASLOC", pos0); pos != std::string::npos) {
                auto nasloc = zeno::getConfigVariable("NASLOC");
                code.replace(pos, 7, nasloc);
                pos0 = pos + 7;
            } else break;

            pos0 = 0;
            while (1) if (auto pos = code.find("$ZSG", pos0); pos != std::string::npos) {
                auto zsgPath = zeno::getConfigVariable("ZSG");
                code.replace(pos, 4, zsgPath);
                pos0 = pos + 4;
            }
            else break;

            //for (int i = 0; i < code.size(); i++) {
                //if (code[i] == '$' && code[i+1] == 'F') {
                    //code.replace(i, 2, std::to_string(getGlobalState()->getFrameId()));
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
                            {{"string", "zfxCode", "", ParamSocket, CodeEditor}},
                            {{"string", "result"}},
                            {},
                            {"zenofx"}
                           });
}
}
