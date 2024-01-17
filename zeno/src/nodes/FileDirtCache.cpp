#include <zeno/zeno.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/log.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/assetDir.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <cstdlib>

namespace zeno {
namespace {

struct EmbedZsgGraph : zeno::INode {
    virtual void apply() override {
        auto zsgPath = get_input2<std::string>("zsgPath");
        auto zsgp = std::filesystem::u8path(zsgPath).string();
        auto zslPath = (std::filesystem::temp_directory_path() / (".tmpEZG-" + std::to_string(std::random_device()()) + "-tmp.zsl")).string();
        auto zslp = std::filesystem::u8path(zslPath).string();
        auto cmd = zeno::getConfigVariable("EXECFILE") + " -invoke dumpzsg2zsl " + zsgp + " " + zslp;
        log_info("executing command: [{}]", cmd);
        std::system(cmd.c_str());
        log_info("done execution, zsl should be generated");
        std::string content;
        if (std::ifstream ifs(zslPath); !ifs) {
            throw makeError("failed to generate temporary zsl file!\n");
        } else {
            std::istreambuf_iterator<char> iit(ifs), eiit;
            std::copy(iit, eiit, std::back_inserter(content));
            ifs.close();
            /* std::filesystem::remove(zslPath); */
        }
        auto g = getThisSession()->createGraph("main");
        g->addSubnetNode("custom")->loadGraph(content.c_str());
        auto argsDict = has_input("argsDict") ? get_input<DictObject>("argsDict") : std::make_shared<DictObject>();
        auto retsDict = std::make_shared<DictObject>();
        retsDict->lut = g->callSubnetNode("custom", argsDict->lut);
        /* for (auto &[k, v]: retsDict->lut) { */
        /*     zeno::log_warn("mama [{}] [{}]", k, v); */
        /* } */
        set_output("retsDict", std::move(retsDict));
    }
};

ZENO_DEFNODE(EmbedZsgGraph)({
    {
       {"readpath", "zsgPath", ""},
       {"dict", "argsDict"},
    },
    {
       {"dict", "retsDict"},
    },
    {
    },
    {"subgraph"},
});

}
}
