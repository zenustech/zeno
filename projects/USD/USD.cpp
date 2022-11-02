#include <iostream>

#include <boost/predef/os.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/js/json.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/tf/fileUtils.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/inherits.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/logger.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

struct USDStage : zeno::INode {
    virtual void apply() override {
        auto primObjs = get_input<zeno::ListObject>("list")->get<zeno::PrimitiveObject>();
        UsdStageRefPtr usdStage = UsdStage::CreateInMemory();

        for(auto const& prim: primObjs){
            std::string primPath = prim->userData().get2<std::string>("path");
            std::cout << "USD: Path " << primPath << std::endl;
        }

        auto result = std::make_shared<zeno::NumericObject>();
        result->set(0);
        set_output("result", std::move(result));

        std::string stageString;
        usdStage->ExportToString(&stageString);
        usdStage->Save();
        std::cout << "USD: Stage " << std::endl << stageString << std::endl;
    }
};
ZENDEFNODE(USDStage,
       {       /* inputs: */
        {
            "list",
        },  /* outputs: */
        {
            "result",
        },  /* params: */
        {
        },  /* category: */
        {
            "USD",
        }
       });
}