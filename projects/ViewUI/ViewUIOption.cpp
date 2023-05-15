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
    Head = 2,
    Tail,
    Top,
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

struct ShowPrim : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ShowPrim,
    {/*输入*/
    {"prim"},
    /*输出*/
    {"prim"},
    /*参数*/
    {},
    /*类别*/
    {"ViewUI"}});

struct PointSpace : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto pointList = get_input<zeno::ListObject>("pointList").get();
        auto tmp1 = pointList->get<std::decay_t<zeno::NumericObject>>();
        std::vector<zeno::vec3f> tmpPnts;
        for (int n = 0; n < tmp1.size(); n++) {
            tmpPnts.push_back(prim->verts[tmp1[n]->get<int>()]);
        }
        float space = fabs(fabs(tmpPnts[1][0] - tmpPnts[0][0] + tmpPnts[0][0]) -
                           fabs(tmpPnts[2][0] + tmpPnts[3][0] - tmpPnts[2][0]));
        auto value = std::make_shared<zeno::NumericObject>();
        value->set(space);
        set_output("value", value);
    }
};

ZENDEFNODE(PointSpace, {/*输入*/
                      {"prim", "pointList"},
                      /*输出*/
                      {"value"},
                      /*参数*/
                      {},
                      /*类别*/
                      {"ViewUI"}});

struct BlendShapeListParse : zeno::INode {
    virtual void apply() override {
        auto objOption = std::make_shared<CarModelOption>();

        auto lineType = get_param<std::string>("LineType");
        auto checked = get_param<bool>("Checked");

        //auto bsPrims = get_input<zeno::PrimitiveObject>("bsPrims").get();
        auto primAndMaterial = get_input<zeno::ListObject>("list").get();
        //auto tmpPrim = dynamic_cast<zeno::PrimitiveObject *>(primAndMaterial->arr[0].get());
        auto tmpPrim1 = dynamic_cast<zeno::ListObject *>(primAndMaterial->arr[0].get());
        //objOption->originalObj = std::make_shared<zeno::PrimitiveObject>(*tmpPrim);
        if (tmpPrim1 == nullptr)
        {
            set_output("prim", objOption->originalObj);
            set_output("controlPoint", objOption->newObj);
            set_output("opt", objOption);
            set_output("length", std::make_shared<zeno::NumericObject>(0));
            set_output("width", std::make_shared<zeno::NumericObject>(0));
            set_output("height", std::make_shared<zeno::NumericObject>(0));
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

        if (checked == true) {
            objOption->checked = true;

            if (lineType == "Head") {
                objOption->type = Head;
            } else if (lineType == "Tail") {
                objOption->type = Tail;
            } else if (lineType == "Top") {
                objOption->type = Top;
            } else if (lineType == "HeadOutline") {
                objOption->type = HeadOutline;
            } else if (lineType == "TopProfile") {
                objOption->type = TopProfile;
            } else if (lineType == "SideProfile") {
                objOption->type = SideProfile;
            }
            auto tmpNewPrim = dynamic_cast<zeno::PrimitiveObject *>(primAndMaterial->arr[objOption->type].get());
            objOption->newObj = std::make_shared<zeno::PrimitiveObject>(*tmpNewPrim);
        } else {
            objOption->newObj = std::make_shared<zeno::PrimitiveObject>();
        }
        
        set_output("prim", objOption->originalObj);
        set_output("controlPoint", objOption->newObj);
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
                           {"prim", "controlPoint", "opt", "length", "width", "height"},
                           /*参数*/
                           {
                               {"bool", "Checked", "false"},
                               {"enum Head Tail Top HeadOutline TopProfile SideProfile", "LineType", "Head"},
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
                    for (int n = 0; n < tmp2.size(); n++)
                    {
                        tmpPntIdx.push_back(tmp2[n]->get<int>());
                    }
                    pntIdx.push_back(tmpPntIdx);
                }
            }
            outPrim->verts.resize(pntIdx.size());
            for (int i = 0; i < changePntList.size(); i++) {
                auto tmpIdx = pntIdx[i];
                if (changePntList[i]->value.index() == 4)
                {                    
                    auto tmpPntOffset = changePntList[i]->get<float>();
                    int tmpSize = tmpIdx.size() / 2;
                    for (int i = 0; i < tmpIdx.size(); i++)
                    {
                        if (i < tmpSize) {
                            prim->verts[tmpIdx[i]][2] += tmpPntOffset;
                        } else {
                            prim->verts[tmpIdx[i]][2] -= tmpPntOffset;
                        }
                    }
                    /*for (auto tmpIdxSub : tmpIdx) {
                    }*/

                } else {
                    auto tmpPntOffset = changePntList[i]->get<zeno::vec2f>();                    
                    for (auto tmpIdxSub : tmpIdx) {
                        prim->verts[tmpIdxSub][0] += tmpPntOffset[0];
                        prim->verts[tmpIdxSub][1] += tmpPntOffset[1];
                    }
                }
                outPrim->verts[i] = prim->verts[tmpIdx[0]];
            }
        }
        set_output("prim", std::move(prim));
        set_output("ControlPoint", std::move(outPrim));
    }
};
ZENDEFNODE(CreatePrimeListInPointIndex, {/*输入*/
                                         {"prim", "pointList", "changePntList"},
                                          /*输出*/
                                         {"prim", "ControlPoint"},
                                          /*参数*/
                                          {},
                                          /*类别*/
                                          {"ViewUI"}});


} // namespace
