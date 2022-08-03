#include <zeno/utils/nowarn.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include "zeno/utils/logger.h"
#include "zeno/types/UserData.h"
#include "zeno/types/StringObject.h"
#include "zeno/types/NumericObject.h"
#include "aquila/aquila/aquila.h"
#include <deque>
#include <zeno/types/ListObject.h>
#include "PBD.h"


namespace zeno {
    struct PBD : zeno::INode {
        virtual void apply() override {
            auto path = get_input<StringObject>("path")->get(); // std::string

            auto result = std::make_shared<PrimitiveObject>(); // std::shared_ptr<PrimitiveObject>
            auto &value = result->add_attr<float>("value"); //std::vector<float>
            auto &t = result->add_attr<float>("t");

            for (std::size_t i = 0; i < result->verts.size(); ++i) {
                t[i] = float(i);
            }

            result->userData().set("SampleRate", std::make_shared<zeno::NumericObject>((int)wav.getSampleRate()));
            result->userData().set("BitDepth", std::make_shared<zeno::NumericObject>((int)wav.getBitDepth()));
            result->userData().set("NumSamplesPerChannel", std::make_shared<zeno::NumericObject>((int)wav.getNumSamplesPerChannel()));
            result->userData().set("LengthInSeconds", std::make_shared<zeno::NumericObject>((float)wav.getLengthInSeconds()));

            set_output("wave",result);
        }
    };
    ZENDEFNODE(PBD, {
        {
            {"readpath", "path"},
        },
        {
            "wave",
        },
        {},
        {
            "audio"
        },
    });


} // namespace zeno
