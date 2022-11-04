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

        for (auto const &prim : primObjs) {
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
ZENDEFNODE(USDStage, {/* inputs: */
                      {
                          "list",
                      }, /* outputs: */
                      {
                          "result",
                      },  /* params: */
                      {}, /* category: */
                      {
                          "USD",
                      }});

struct USDLight : zeno::INode {
    virtual void apply() override {
        auto translate = get_input2<zeno::vec3f>("translate");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto scale = get_input2<zeno::vec3f>("scale");
        auto intensity = get_input2<float>("intensity");
        auto exposure = get_input2<float>("exposure");
        auto color = get_input2<zeno::vec3f>("color");
        auto type = get_param<std::string>("type");
        auto path = get_input2<std::string>("path");
        std::string _type;
        if (type == "Disk") {
            _type = "UsdLuxDiskLight";
        } else if (type == "Cylinder") {
            _type = "UsdLuxCylinderLight";
        } else if (type == "Distant") {
            _type = "UsdLuxDistantLight";
        } else if (type == "Dome") {
            _type = "UsdLuxDomeLight";
        } else if (type == "Rectangle") {
            _type = "UsdLuxRectLight";
        } else if (type == "Sphere") {
            _type = "UsdLuxSphereLight";
        }

        // TODO Display the light shape
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->verts.emplace_back(translate);
        prim->userData().set2("translate", std::move(translate));
        prim->userData().set2("rotate", std::move(rotate));
        prim->userData().set2("scale", std::move(scale));
        prim->userData().set2("intensity", std::move(intensity));
        prim->userData().set2("exposure", std::move(exposure));
        prim->userData().set2("color", std::move(color));
        prim->userData().set2("P_Type", std::move(_type));
        prim->userData().set2("P_Path", std::move(path));
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(USDLight)
({
    {
        {"vec3f", "translate", "0, 0, 0"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"float", "intensity", "1"},
        {"float", "exposure", "0"},
        {"vec3f", "color", "1, 1, 1"},
        {"string", "path", "/lights/light"},
    },
    {"prim"},
    {
        {"enum Disk Distant Cylinder Dome Rectangle Sphere", "type", "Disk"},
    },
    {"USD"},
});
}