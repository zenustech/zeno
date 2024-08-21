#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/types/PrimitiveObject.h>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <thread>
#include <deque>

namespace zeno {
namespace {

    struct FFPlayAudioFile : zeno::INode {
        virtual void apply() override {
            auto path = get_input<StringObject>("path")->get(); // std::string
            std::ostringstream oss;
            if (get_input2<bool>("nodisp")) {
                oss << "-nodisp ";
            }
            oss << std::quoted(path);
            std::thread t([cmd = "ffplay -autoexit " + oss.str()] {
                zeno::log_info("executing command: {}", cmd);
                int ret = std::system(cmd.c_str());
                zeno::log_info("ffplay exited with {}", ret);
            });
            if (get_input2<bool>("wait")) {
                t.join();
            } else {
                t.detach();
            }
        }
    };

    ZENDEFNODE(FFPlayAudioFile, {
        {
            {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
            {gParamType_Bool, "nodisp", "0"},
            {gParamType_Bool, "wait", "0"},
        },
        {
        },
        {},
        {
            "audio"
        },
    });

    struct AudioSumPower : zeno::INode {
        virtual void apply() override {
            auto fftp = get_input<PrimitiveObject>("FFTPrim");
            float sumpower = 0;
            auto &power = fftp->attr<float>("power");
            auto &freq = fftp->attr<float>("freq");
            for (int i = 4; i < fftp->size(); i++) {
                sumpower += power[i];
            }
            set_output2("sumpower", sumpower);
        }
    };
    ZENDEFNODE(AudioSumPower, {
        {
            {gParamType_Primitive, "FFTPrim"},
        },
        {
            {gParamType_Float, "sumpower"},
        },
        {},
        {
            "audio"
        },
    });

    struct AudioPowerVariation : zeno::INode {
        std::deque<float> hist;

        virtual void apply() override {
            auto sumpower = get_input2<float>("sumpower");
            int maxhist = get_input2<int>("winwidth");
            hist.push_back(sumpower);
            if (hist.size() > maxhist) {
                hist.pop_front();
            }
            auto v = std::vector<float>(hist.begin(), hist.end());
            std::sort(v.begin(), v.end());
            auto midhist = v[v.size() / 2];
            auto powvar = std::max(0.f, sumpower - midhist);
            powvar *= get_input2<float>("scale");
            powvar = std::min(powvar, get_input2<float>("maximum"));
            set_output2("powvar", powvar);
        }
    };
    ZENDEFNODE(AudioPowerVariation, {
        {
            {gParamType_Float, "sumpower"},
            {gParamType_Int, "winwidth", "20"},
            {gParamType_Int, "scale", "1"},
            {gParamType_Int, "maximum", "1"},
        },
        {
            {gParamType_Float, "powvar"},
        },
        {},
        {
            "audio"
        },
    });

}
}
