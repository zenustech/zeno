#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
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

        //system("pwd");

        std::vector<std::string> cmds = {
            "./dem/DemBones",
            "-i=\"" + fbxpath + "\"",
            "-a=\"" + abcpath + "\"",
            "-b=5",
            "-o=\"" + outpath + "\"",
        };

        std::vector<char *> cmds_cs;
        for (auto const &cmd: cmds)
            cmds_cs.push_back((char *)cmd.c_str());
        int er = DemBones_main(cmds_cs.data());

        auto result = std::make_shared<zeno::NumericObject>();

        //zeno::log_info("----- CMD {}", cmd);
        zeno::log_info("----- DemBones Exec Result {}", er);

        result->set(er);
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
                   "primitive",
               }
           });
