#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/pybjson.h>
#include <zeno/zeno.h>
#include <algorithm>

namespace {

enum CarModelOptionType
{
    Head = 0,
    Tail,
    Top,
    Window,
    HeadOutline,
    TopProfile,
    SideProfile
};

struct CarModelOption : zeno::IObjectClone<CarModelOption> {
    CarModelOptionType type = Head;
    bool checked = false;
    std::shared_ptr<zeno::PrimitiveObject> originalObj = {};
    std::shared_ptr<zeno::PrimitiveObject> newObj = {};
};

struct BlendShapeListParse : zeno::INode {
    virtual void apply() override {
        auto objOption = std::make_shared<CarModelOption>();

        auto lineType = get_param<std::string>("LineType");
        auto checked = get_param<bool>("Checked");
        if (checked == true) {
            objOption->checked = true;

            if (lineType == "Head") {
                objOption->type = Head;
            } else if (lineType == "Tail") {
                objOption->type = Tail;
            } else if (lineType == "Top") {
                objOption->type = Top;
            } else if (lineType == "Window") {
                objOption->type = Window;
            } else if (lineType == "HeadOutline") {
                objOption->type = HeadOutline;
            } else if (lineType == "TopProfile") {
                objOption->type = TopProfile;
            } else if (lineType == "SideProfile") {
                objOption->type = SideProfile;
            }
        }
        //auto bsPrims = get_input<zeno::PrimitiveObject>("bsPrims").get();
        auto primAndMaterial = get_input<zeno::ListObject>("list").get();
        //auto tmpPrim = dynamic_cast<zeno::PrimitiveObject *>(primAndMaterial->arr[0].get());
        auto tmpPrim1 = dynamic_cast<zeno::ListObject *>(primAndMaterial->arr[0].get());
        //objOption->originalObj = std::make_shared<zeno::PrimitiveObject>(*tmpPrim);
        if (tmpPrim1 == nullptr)
        {
            return;
        }
        auto tmpPrim = dynamic_cast<zeno::PrimitiveObject *>(tmpPrim1->arr[0].get());
        objOption->originalObj = std::make_shared<zeno::PrimitiveObject>(*tmpPrim);
        std::vector<float> x, y, z;
        for (auto p : tmpPrim->verts) {
            x.push_back(p[0]);
            y.push_back(p[1]);
            z.push_back(p[2]);
        }

        objOption->newObj = std::make_shared<zeno::PrimitiveObject>(*tmpPrim);
        set_output("prim", objOption->newObj);
        set_output("opt", objOption);

        float minl = *std::min_element(x.begin(), x.end());
        float maxl = *std::max_element(x.begin(), x.end());
        auto tmpl = std::make_shared<zeno::NumericObject>();
        zeno::log_info("length:{},  {},  {}", maxl, minl, fabs(maxl - minl));
        tmpl->set(fabs(maxl - minl));
        set_output("length", tmpl);

        float minw = *std::min_element(z.begin(), z.end());
        float maxw = *std::max_element(z.begin(), z.end());
        auto tmpw = std::make_shared<zeno::NumericObject>();
        zeno::log_info("width:{},  {},  {}", maxw, minw, fabs(maxw - minw));
        tmpw->set(fabs(maxw - minw));
        set_output("width", tmpw);

        float minh = *std::min_element(y.begin(), y.end());
        float maxh = *std::max_element(y.begin(), y.end());
        auto tmpH = std::make_shared<zeno::NumericObject>();
        zeno::log_info("height:{},  {},  {}", maxh, minh, fabs(maxh - minh));
        tmpH->set(fabs(maxh - minh));
        set_output("height", tmpH);
    }
};
ZENDEFNODE(BlendShapeListParse,{/*输入*/
                           {"list"},
                           /*输出*/
                           {"prim", "opt", "length", "width", "height"},
                           /*参数*/
                           {
                               {"bool", "Checked", "false"},
                               {"enum Head Tail Top Window HeadOutline TopProfile SideProfile", "LineType", "Head"},
                           },
                           /*类别*/
                           {"ViewUI"}});



struct CreatePrimeListInPointIndex : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto pointList = get_input<zeno::ListObject>("pointList")->get();
        auto changePntList = get_input<zeno::ListObject>("changePntList")->get<std::decay_t<zeno::NumericObject>>();
        auto outPrim = std::make_shared<zeno::PrimitiveObject>();
        if (prim != nullptr) {
            std::vector<std::vector<int>> pntIdx;
            if (pointList.size() > 0)
            {
                for (int i = 0; i < pointList.size(); i++) {
                    auto tmp1 = dynamic_cast<zeno::ListObject *>(pointList[i].get());
                    auto tmp2 = tmp1->get<std::decay_t<zeno::NumericObject>>();
                    std::vector<int> tmpPntIdx;
                    for (int i = 0; i < tmp2.size(); i++)
                    {
                        tmpPntIdx.push_back(tmp2[i]->get<int>());
                    }
                    pntIdx.push_back(tmpPntIdx);
                }
            }
            for (int i = 0; i < changePntList.size(); i++) {
                auto tmpPntOffset = changePntList[i]->get<zeno::vec3f>();
                auto tmpIdx = pntIdx[i];
                for (auto tmpIdxSub : tmpIdx) {
                    auto tmpPntOriginal = prim->verts[tmpIdxSub];
                    prim->verts[tmpIdxSub][0] = tmpPntOriginal[0] + tmpPntOffset[0];
                    prim->verts[tmpIdxSub][1] = tmpPntOriginal[1] + tmpPntOffset[1];
                }                
            }          
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(CreatePrimeListInPointIndex, {/*输入*/
                                         {"prim", "pointList", "changePntList"},
                                          /*输出*/
                                          {"prim"},
                                          /*参数*/
                                          {},
                                          /*类别*/
                                          {"ViewUI"}});


} // namespace
