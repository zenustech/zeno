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

struct ExportFBX : zeno::INode {

    virtual void apply() override {
        auto abcpath = get_input<zeno::StringObject>("abcpath")->get();
        auto fbxpath = get_input<zeno::StringObject>("fbxpath")->get();
        auto outpath = get_input<zeno::StringObject>("outpath")->get();

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
            " -b=5" +
            " -o=\"" + outpath + "\"" +
            "";

#ifdef _WIN32
        for (auto &c: cmd) {
            if (c == '/') c = '\\';
        }
#endif
        int er = std::system(cmd.c_str());
        auto result = std::make_shared<zeno::NumericObject>();
        result->set(er);

        zeno::log_info("----- CMD {}", cmd);
        zeno::log_info("----- Exec Result {}", er);

        set_output("result", std::move(result));
    }
};

ZENDEFNODE(ExportFBX,
           {       /* inputs: */
               {
                   {"string", "abcpath"},
                   {"string", "fbxpath"},
                   {"string", "outpath"}
               },  /* outputs: */
               {
                    "result"
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });
