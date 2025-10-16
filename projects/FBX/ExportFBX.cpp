#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/assetDir.h>
#include <zeno_FBX_config.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>

#include <string>

namespace {

struct ExportFBX : zeno::INode {

    virtual void apply() override {
        auto abcpath = zsString2Std(get_input2_string("abcpath"));
        auto fbxpath = zsString2Std(get_input2_string("fbxpath"));
        auto outpath = zsString2Std(get_input2_string("outpath"));
        auto extra_param = zsString2Std(get_input2_string("extra_param"));

        zeno::log_info("----- ABC Path {}", abcpath);
        zeno::log_info("----- FBX Path {}", fbxpath);
        zeno::log_info("----- OUT Path {}", outpath);

        zeno::cihouWinPath(abcpath);
        zeno::cihouWinPath(fbxpath);
        zeno::cihouWinPath(outpath);

        //system("pwd");

        auto cmd = (std::string)
                       "\"" + zeno::getAssetDir(DEM_DIR, "DemBones") + "\"" +
                   " -i=\"" + fbxpath + "\"" +
                   " -a=\"" + abcpath + "\"" +
                   " " + extra_param +
                   " -o=\"" + outpath + "\"" +
                   "";

#ifdef _WIN32
        for (auto &c: cmd) {
            if (c == '/') c = '\\';
        }
#endif

        has_input("custom_command") ? cmd = zsString2Std(get_input2_string("custom_command"))
                                    : cmd;

        int er = std::system(cmd.c_str());

        auto result = std::make_unique<zeno::NumericObject>();
        result->set(er);

        zeno::log_info("----- CMD {}", cmd);
        zeno::log_info("----- Exec Result {}", er);

        set_output("result", std::move(result));
    }
};

ZENDEFNODE(ExportFBX,
           {       /* inputs: */
            {
                {gParamType_String, "custom_command"},
                {gParamType_String, "extra_param", " -b=5"},
                {gParamType_String, "abcpath"},
                {gParamType_String, "fbxpath"},
                {gParamType_String, "outpath"}
            },  /* outputs: */
            {
                {gParamType_Int,"result"}
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

}
