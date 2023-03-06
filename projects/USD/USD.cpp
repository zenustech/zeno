#include "usdDefinition.h"
#include "usdImporter.h"

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>

struct ZUSDStage : zeno::IObjectClone<ZUSDStage> {
    EUSDStage eusdStage;
};

struct OpenUSDStage : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();

        EUsdImporter usdImporter;
        EUsdOpenStageOptions options;
        options.Identifier = path;
        usdImporter.OpenStage(options);

        auto zusdStage = std::make_shared<ZUSDStage>();
        zusdStage->eusdStage = EUSDStage{usdImporter.mStage};

        set_output("zstage", std::move(zusdStage));
    }
};
ZENDEFNODE(OpenUSDStage,
       {       /* inputs: */
        {
            {"readpath", "path"},
        },  /* outputs: */
        {
            "zstage"
        },  /* params: */
        {
        },  /* category: */
        {
            "USD",
        }
       });
