#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/pybjson.h>
#include <zeno/utils/UserData.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/extra/assetDir.h>
#include <zeno/zeno.h>
#include <algorithm>
#include "assimp/scene.h"
#include "assimp/mesh.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "../FBX/assimp/code/PostProcessing/TriangulateProcess.h"
#include "../FBX/assimp/include/assimp/mesh.h"
#include "glm/vec3.hpp"
#include "glm/glm.hpp"

#include "ViewUtil.h"

#include <memory>
#include <atomic>
#include <limits>
#include <type_traits>
#include <iostream>
#include <format>

namespace {

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


struct PrimitiveEdit : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto pointIndex = get_param<int>("pointIndex");
        auto tmpUserData = prim->userData();
        std::string pIdx = get_param<std::string>("index");
        std::string sArea = get_param<std::string>("area");
        std::string sType = get_param<std::string>("Operate");
        int inRow = get_param<int>("rowCount");
        int inCol = get_param<int>("columnCount");

        int inOutLine = get_param<int>("outline");
        int inInLine = get_param<int>("inline");

		auto movePoint = get_input2<zeno::vec3f>("move");
        
        std::vector<int> movePointIdxs;
        if (has_input("points")) {
			auto movePointIdx = get_input<zeno::ListObject>("points")->get();
			for (int i = 0; i < movePointIdx.size(); i++) {
				auto tmpIdx = dynamic_cast<zeno::NumericObject*>(movePointIdx[i].get());
				movePointIdxs.push_back(tmpIdx->get<int>());
			}
        }

        auto pIdxs = split(pIdx, ',');
        auto sAreas = split(sArea, ',');
        if (sType == "Line")
        {
            if (sAreas.size() == 0) { 
                //auto outPrim = std::make_shared<zeno::PrimitiveObject>();
                set_output("prim", std::move(std::make_shared<zeno::PrimitiveObject>()));
                //set_output("outs", std::move(std::make_shared<zeno::ListObject>()));
                return;
            }
        } 
        else if (pIdxs.size() == 0 || sAreas.size() == 0)
        {
            //auto outPrim = std::make_shared<zeno::PrimitiveObject>();
            set_output("prim", std::move(std::make_shared<zeno::PrimitiveObject>()));
            //set_output("outs", std::move(std::make_shared<zeno::ListObject>()));
            return;
        }

        std::sort(pIdxs.begin(), pIdxs.end());
        #if 0
        auto outPrim = std::make_shared<zeno::PrimitiveObject>();
        outPrim->verts->push_back(zeno::vec3f(stof(sAreas[0]), stof(sAreas[1]), stof(sAreas[2])));
        outPrim->verts->push_back(zeno::vec3f(stof(sAreas[3]), stof(sAreas[4]), stof(sAreas[5])));
        set_output("prim", outPrim);
        #endif

        //outPrim->verts = prim->verts;
        std::shared_ptr<zeno::PrimitiveObject> outPrim = std::static_pointer_cast<zeno::PrimitiveObject>(prim->clone());
        if (outPrim->tris->size() == 0)
        {
			prim_triangulate(outPrim.get());
			outPrim->loops->clear();
			outPrim->polys->clear();
        }
        zeno::log_info("tris size: {}, verts size: {}", outPrim->tris.size(), outPrim->verts.size());

        auto outlist = std::make_shared<zeno::ListObject>();

        if (sType == "Face")
        {
            //获取鼠标平面
            auto tmpvx = fabs(fabs(stof(sAreas[0])) - fabs(stof(sAreas[3])));
            auto tmpvy = fabs(fabs(stof(sAreas[1])) - fabs(stof(sAreas[4])));
            auto tmpvz = fabs(fabs(stof(sAreas[2])) - fabs(stof(sAreas[5])));
            auto tmpv = tmpvx < tmpvy ? 0 : 1;
            if (tmpv == 0)
                tmpv = tmpvx < tmpvz ? 0 : 2;
            else
                tmpv = tmpvy < tmpvz ? 1 : 2;

            std::map<int, zeno::vec3f> tmpPnts;
            auto tmpMean = 0.f;

            //删除点
            int eraseIdx = 0;
            for (auto p : pIdxs) {
                int tmpIdx = std::stoi(p);
                tmpPnts.emplace(tmpIdx, prim->verts[tmpIdx]);
                auto it = outPrim->verts->begin() + tmpIdx - eraseIdx;
				outPrim->verts->erase(it);
				eraseIdx++;
            }

            #define REPEAT_DETECT                                                                                       \
            bool repeat = false;                                                                                    \
            for (auto dT : dTris) {                                                                                 \
                if (dT[0] == outPrim->tris[i][0] && dT[1] == outPrim->tris[i][1] && dT[2] == outPrim->tris[i][2]) { \
                    repeat = true;                                                                                  \
                    break;                                                                                          \
                }                                                                                                   \
            }                                                                                                       \

            std::vector<zeno::vec2i> realDTris;
            std::vector<zeno::vec3i> dTris{};
            for (int i = 0; i < outPrim->tris->size(); i++) 
            {
                for (auto tmpP : tmpPnts) {
                    if (tmpP.first == outPrim->tris[i][0]) {

                        REPEAT_DETECT
                        if (!repeat) {
                            dTris.push_back(outPrim->tris[i]);
                            realDTris.push_back({outPrim->tris[i][1], outPrim->tris[i][2]});
                            zeno::log_info("{} {} {}", outPrim->tris[i][0], outPrim->tris[i][1], outPrim->tris[i][2]);
                        }

                    } else if (tmpP.first == outPrim->tris[i][1]) {
                        REPEAT_DETECT
                        if (!repeat) {
                            dTris.push_back(outPrim->tris[i]);
                            realDTris.push_back({outPrim->tris[i][0], outPrim->tris[i][2]});
                            zeno::log_info("{} {} {}", outPrim->tris[i][0], outPrim->tris[i][1], outPrim->tris[i][2]);
                        }
                    } else if (tmpP.first == outPrim->tris[i][2]) {
                        REPEAT_DETECT
                        if (!repeat) {
                            dTris.push_back(outPrim->tris[i]);
                            realDTris.push_back({outPrim->tris[i][0], outPrim->tris[i][1]});
                            zeno::log_info("{} {} {}", outPrim->tris[i][0], outPrim->tris[i][1], outPrim->tris[i][2]);
                        }
                    }
                }
            }


            // 删除三角形
            for (int i = 0; i < outPrim->tris->size(); i++) {
                auto itp1 = tmpPnts.find(outPrim->tris[i][0]);
                auto itp2 = tmpPnts.find(outPrim->tris[i][1]);
                auto itp3 = tmpPnts.find(outPrim->tris[i][2]);
                if (itp1 != tmpPnts.end() || itp2 != tmpPnts.end() || itp3 != tmpPnts.end()) {
                    auto itt = outPrim->tris->begin() + i;
                    outPrim->tris->erase(itt);
                    i--;
                }
            }

            // 修正三角形索引
            for (int i = 0; i < outPrim->tris->size(); i++) 
            {
                int tmpCurIdx1 = outPrim->tris[i][0];
                int tmpCurIdx2 = outPrim->tris[i][1];
                int tmpCurIdx3 = outPrim->tris[i][2];
                for (auto itp = tmpPnts.begin(); itp != tmpPnts.end(); itp++) {
                    if (tmpCurIdx1 >= itp->first) {
                        outPrim->tris[i][0] -= 1;
                    }
                    if (tmpCurIdx2 >= itp->first) {
                        outPrim->tris[i][1] -= 1;
                    }
                    if (tmpCurIdx3 >= itp->first) {
                        outPrim->tris[i][2] -= 1;
                    }
                }
                tmpCurIdx1 = outPrim->tris[i][0];
                tmpCurIdx2 = outPrim->tris[i][1];
                tmpCurIdx3 = outPrim->tris[i][2];
                int tmpVecSize = outPrim->verts.size();
                if (tmpCurIdx1 >= tmpVecSize || 
                    tmpCurIdx2 >= tmpVecSize || 
                    tmpCurIdx3 >= tmpVecSize) {
                    zeno::log_info("error face point index: {},{},{}", tmpCurIdx1, tmpCurIdx2, tmpCurIdx3);
                }                
            }

            std::vector<int> outline{};
            int startIndice = realDTris[0][0];
            int firstIndice = startIndice;
            int foundIdx = 0;
            outline.push_back(startIndice);
            startIndice = realDTris[0][1];
            outline.push_back(startIndice);
            for (int i = 0; i < realDTris.size(); ++i) {
                int nextIndice = -1;
                for (int j = 0; j < realDTris.size(); ++j) {
                    if (j == foundIdx)
                        continue;

                    auto tI2 = realDTris[j];
                    if (startIndice == tI2[0] || startIndice == tI2[1]) {
                        if (startIndice == tI2[0])
                            nextIndice = tI2[1];
                        if (startIndice == tI2[1])
                            nextIndice = tI2[0];

                        foundIdx = j;
                        break;
                    }
                }

                if (nextIndice == -1) {
                    break;
                } else {

                    if (nextIndice == firstIndice)
                        break;

                    outline.push_back(nextIndice);
                    startIndice = nextIndice;
                }
            }

            std::vector<zeno::vec3f> innPoints;
            innPoints.resize(4);
            //
            // 0  --------  1
            //   |        |
            //   |        |
            // 3  --------  2
            //
            innPoints[0] = zeno::vec3f(stof(sAreas[0]), stof(sAreas[1]), stof(sAreas[2]));
            innPoints[1] = zeno::vec3f(stof(sAreas[3]), stof(sAreas[1]), stof(sAreas[5]));
            innPoints[2] = zeno::vec3f(stof(sAreas[3]), stof(sAreas[4]), stof(sAreas[5]));
            innPoints[3] = zeno::vec3f(stof(sAreas[0]), stof(sAreas[4]), stof(sAreas[2]));
            
            for (auto i : movePointIdxs)
            {
                if (i >= innPoints.size())
                    break;
                innPoints[i] += movePoint;
            }

            auto innP = innPoints[0];
            float minnimal = 9999999999.0f;
            std::map<float, int> minValues;
            for (int outIdx = 0; outIdx < outline.size(); ++outIdx) {
                auto outP = outPrim->verts[outline[outIdx]];
                float len = glm::length(glm::vec3(outP[0], outP[1], outP[2]) - glm::vec3(innP[0], innP[1], innP[2]));
                minValues[len] = outIdx;
            }
            
            int minValueIdx = minValues.begin()->second;

            int inOutLine = get_param<int>("outline");
            int inInLine = get_param<int>("inline");
            std::vector<int> holeIndices;
            std::vector<zeno::vec3f> holePositions;
            for (int idx = 0; idx < outline.size(); ++idx) {
                int orderIdx = outline[(idx + minValueIdx) % outline.size()];
                holeIndices.push_back(orderIdx);
            }
            holeIndices.push_back(outline[minValueIdx]);
            if (inOutLine == 1)
            {
                std::reverse(holeIndices.begin(), holeIndices.end());
            }
            if (inInLine == 1)
            {
                holeIndices.push_back(outPrim->verts->size());
                holeIndices.push_back(outPrim->verts->size() + 1);
                holeIndices.push_back(outPrim->verts->size() + 2);
                holeIndices.push_back(outPrim->verts->size() + 3);
                holeIndices.push_back(outPrim->verts->size());
            } else {
                holeIndices.push_back(outPrim->verts->size());
                holeIndices.push_back(outPrim->verts->size() + 3);
                holeIndices.push_back(outPrim->verts->size() + 2);
                holeIndices.push_back(outPrim->verts->size() + 1);
                holeIndices.push_back(outPrim->verts->size());
            }
            
            zeno::log_info("tris size: {}, verts size: {}", outPrim->tris.size(), outPrim->verts.size());
            outPrim->verts->push_back(innPoints[0]);
            outPrim->verts->push_back(innPoints[1]);
            outPrim->verts->push_back(innPoints[2]);
            outPrim->verts->push_back(innPoints[3]);

            auto triProcess = Assimp::TriangulateProcess();
            aiMesh *mesh = new aiMesh;
            mesh->mVertices = new aiVector3D[outPrim->verts->size()];
            mesh->mFaces = new aiFace[outPrim->tris->size() + 1];

            for (int vIdx = 0; vIdx < outPrim->verts->size(); ++vIdx) {
                auto p = outPrim->verts[vIdx];
                mesh->mVertices[vIdx] = aiVector3D{p[0], p[1], p[2]};
            }
            for (int fIdx = 0; fIdx < outPrim->tris->size(); ++fIdx) {
                auto f = outPrim->tris[fIdx];
                aiFace face;
                unsigned int count = 3;
                face.mNumIndices = count;
                unsigned int *mIndices = new unsigned int[3];
                mIndices[0] = (unsigned int)f[0];
                mIndices[1] = (unsigned int)f[1];
                mIndices[2] = (unsigned int)f[2];
                face.mIndices = mIndices;
                mesh->mFaces[fIdx] = face;
            }

            aiFace holeFace;
            holeFace.mNumIndices = holeIndices.size();
            unsigned int* mIndices = new unsigned int[holeIndices.size()];
            for (int hIdx = 0; hIdx < holeIndices.size(); ++hIdx) {
                mIndices[hIdx] = holeIndices[hIdx];
            }
            holeFace.mIndices = mIndices;
            mesh->mFaces[outPrim->tris->size()] = holeFace;

            mesh->mNumVertices = outPrim->verts->size();
            mesh->mNumFaces = outPrim->tris->size()+1;

            triProcess.TriangulateMesh(mesh);

            outPrim->verts->clear();
            outPrim->tris->clear();
            outPrim->lines->clear();

            for (int fIdx = 0; fIdx < mesh->mNumFaces; ++fIdx) {
                aiFace aFace = mesh->mFaces[fIdx];
                zeno::vec3i tri;
                tri[0] = aFace.mIndices[0];
                tri[1] = aFace.mIndices[1];
                tri[2] = aFace.mIndices[2];
                outPrim->tris->emplace_back(tri);

            }

            for (int vIdx = 0; vIdx < mesh->mNumVertices; ++vIdx) {
                aiVector3D pos = mesh->mVertices[vIdx];
                outPrim->verts->emplace_back(zeno::vec3f(pos[0], pos[1], pos[2]));
            }
          
            //保存点到文件
            #if 0
            FILE *fp = fopen("test.txt", "w");
            if (fp) {
                for (auto p : outPrim->verts) {
                    fprintf(fp, "%f %f %f\n", p[0], p[1], p[2]);
                }
            }
            fclose(fp);
            #endif

#if 0
            //添加显示鼠标点
            auto vSize = outPrim->verts->size();
            auto outPrim1 = std::make_shared<zeno::PrimitiveObject>();
            outPrim1->verts->emplace_back(outPrim->verts[vSize - 1]);
            auto &pIdxArrt1 = outPrim1->add_attr<int>("pIdx");
            pIdxArrt1[0] = vSize - 1;

            auto outPrim2 = std::make_shared<zeno::PrimitiveObject>();
            outPrim2->verts->emplace_back(outPrim->verts[vSize - 2]);
            auto &pIdxArrt2 = outPrim2->add_attr<int>("pIdx");
            pIdxArrt2[0] = vSize - 2;

            auto outPrim3 = std::make_shared<zeno::PrimitiveObject>();
            outPrim3->verts->emplace_back(outPrim->verts[vSize - 3]);
            auto &pIdxArrt3 = outPrim3->add_attr<int>("pIdx");
            pIdxArrt3[0] = vSize - 3;

            auto outPrim4 = std::make_shared<zeno::PrimitiveObject>();
            outPrim4->verts->emplace_back(outPrim->verts[vSize - 4]);
            auto &pIdxArrt4 = outPrim4->add_attr<int>("pIdx");
            pIdxArrt4[0] = vSize - 4;

            outlist->arr.push_back(outPrim1);
            outlist->arr.push_back(outPrim2);
            outlist->arr.push_back(outPrim3);
            outlist->arr.push_back(outPrim4);
#endif
            zeno::log_info("tris size: {}, verts size: {}", outPrim->tris.size(), outPrim->verts.size());

#if 1
			//outlist->arr.push_back(outPrim);
			set_output("prim", outPrim);
#else
			set_output("prim", outPrim);
			set_output("outs", outlist);
#endif

        } 
        else if (sType == "Line")
        {
            // 判断鼠标平面
            auto tmpvx = fabs(fabs(stof(sAreas[0])) - fabs(stof(sAreas[3])));
            auto tmpvy = fabs(fabs(stof(sAreas[1])) - fabs(stof(sAreas[4])));
            auto tmpvz = fabs(fabs(stof(sAreas[2])) - fabs(stof(sAreas[5])));
            auto tmpv = tmpvx < tmpvy ? 0 : 1;
            if (tmpv == 0)
                tmpv = tmpvx < tmpvz ? 0 : 2;
            else
                tmpv = tmpvy < tmpvz ? 1 : 2;

            //鼠标划的起始点和终点
            zeno::vec3f p1 = {stof(sAreas[0]), stof(sAreas[1]), stof(sAreas[2])};
            zeno::vec3f p2 = {stof(sAreas[3]), stof(sAreas[4]), stof(sAreas[5])};
            //p1[tmpv] += p1[tmpv] > 0 ? 0.1 : -0.1;
            //判断点x,y,z 间隔
            auto spacx = (p1[0] - p2[0]) / inCol;
            spacx = p1[0] < p1[0] ? -fabs(spacx) : fabs(spacx);            
            auto spacy = (p1[1] - p2[1]) / inCol;
            spacy = p1[1] < p1[1] ? -fabs(spacy) : fabs(spacy);
            auto spacz = (p1[2] - p2[2]) / inCol;
            spacz = p1[2] < p1[2] ? -fabs(spacz) : fabs(spacz);

            std::vector<zeno::vec3f> tmpPnts;
            //int tmpAddPntIdx = outPrim->verts.size();
            // 添加鼠标划线中间的点            
            for (int addIdx = 0; addIdx < inCol; addIdx++)
            {
                zeno::vec3f tmpPnt;
                if (addIdx == 0) {
                    tmpPnt = zeno::vec3f(p1[0] + spacx, p1[1] + spacy, p1[2] + spacz);
                    tmpPnts.push_back(tmpPnt);
                } else {
                    tmpPnt = tmpPnts.back();
                    tmpPnt[0] += spacx;
                    tmpPnt[1] += spacy;
                    //tmpPnt[2] += spacz;
                }
                tmpPnts.push_back(tmpPnt);
            }

            // 判断点是否在面中来确定要添加的点
            // 所在三角型的索引，点位置
			// 判断在三角形中的点
			// 判断在三角形中的点
			std::map<int, zeno::vec3f> addPnts;
			for (int i = 0; i < outPrim->verts->size(); i++) {
				zeno::vec3f A = outPrim->verts[outPrim->tris[i][0]];
				zeno::vec3f B = outPrim->verts[outPrim->tris[i][1]];
				zeno::vec3f C = outPrim->verts[outPrim->tris[i][2]];
				std::vector<zeno::vec3f> tmpTri = { A, B, C };
				for (int tmpPIdx = 0; tmpPIdx < tmpPnts.size(); tmpPIdx++)
				{
					zeno::vec3f D = tmpPnts[tmpPIdx];
					if (fabs(A[tmpv] - D[tmpv]) > 0.1)
					{
						continue;
					}
#if 1
					auto a1 = calculateArea(A, B, C);
					auto a2 = calculateArea(A, B, D);
					auto a3 = calculateArea(A, D, C);
					auto a4 = calculateArea(D, B, C);
					float ret = fabs(a1 - (a2 + a3 + a4));
					if (ret < 0.001)
					{
						std::cout << "tri idx: " << i << "  ==========In Plane Pnt: "
							<< tmpPnts[tmpPIdx][0] << " " << tmpPnts[tmpPIdx][1] << " " << tmpPnts[tmpPIdx][2] << std::endl;
						addPnts[i] = tmpPnts[tmpPIdx];
					}
#else


					//if (dot_inplane_in(A, B, C, tmpPnts[tmpPIdx]))//判点是否在空间三角形上,包括边界,三点共线无意义
					//if (dot_inplane_ex(A, B, C, tmpPnts[tmpPIdx]))//判点是否在空间三角形上,不包括边界,三点共线无意义
					//if (dots_onplane(A, B, C, tmpPnts[tmpPIdx]) == true)//判四点共面
					//if (calculatePntInTril(A, B, C, tmpPnts[tmpPIdx]))
					//if (calculatePntInTrilEx(A, B, C, tmpPnts[tmpPIdx]))
					//if (is_in_3d_polygon(tmpTri, tmpPnts[tmpPIdx]))
					{
						char format_str[128] = { 0 };
						snprintf(format_str, sizeof(format_str) - 1, "A:%f, %f, %f : B:%f, %f, %f : C:%f, %f, %f : P:%f, %f, %f",
							A[0], A[1], A[2],
							B[0], B[1], B[2],
							C[0], C[1], C[2],
							D[0], D[1], D[2]);
						std::cout << format_str << std::endl;
						//std::cout << "In Plane" << std::endl;
						//tmpP[tmpv] = calNewPointPosition(A, B, C)[tmpv];
						addPnts[i] = tmpPnts[tmpPIdx];
					}
#endif
				}
			}

			std::vector<zeno::vec3f> tmpVerts = outPrim->verts;
			std::vector<zeno::vec3i> tmpTris = outPrim->tris;
            std::map<int ,zeno::vec3i> tmpDTris;

            //先减面
			int forIndex = 0;
			for (auto&& [ikey, pnt] : addPnts)
			{
                //记录被删除面在 addPnts中的索引和顶点索引
                tmpDTris[ikey] = tmpTris[ikey - forIndex];
                tmpTris.erase(tmpTris.begin() + ikey - forIndex);
                forIndex++;
				
			}
            
			// 找三角形共边
			std::vector<TrisSharedEdge> tmpDTrisEdges;
			std::vector<TriangleData> tmpDTrisData;
			for (auto&& [ikey, pnt] : tmpDTris)
			{
				tmpDTrisData.push_back({ ikey ,pnt });
			}

			int i1 = tmpDTris.size() - 1;
			int i2 = tmpDTrisEdges.size();
            int tmpWhileCount = 0;
            while ((i2 < i1) && (tmpWhileCount < i1))
			{
				auto tv = tmpDTrisData.back();
				tmpDTrisData.pop_back();
				tmpDTrisData.insert(tmpDTrisData.begin(), tv);

				//std::iter_swap(tmpDTrisData.begin(), tmpDTrisData.end()-1);
				tmpDTrisEdges = findSharedEdgeTriOrd(tmpDTrisData);
				i2 = tmpDTrisEdges.size();
                tmpWhileCount++;
			}

            forIndex = 0;
			bool addOnce = true;
			for (auto p : tmpDTrisEdges)
			{
				int tmpCommEdgeTrisidx = addOnce == true ? p.tris1 : p.tris2;
				auto tmpAddPnt = addPnts[tmpCommEdgeTrisidx];
				
				int tmpPntIdx = tmpVerts.size();
				auto iter = std::find(movePointIdxs.begin(), movePointIdxs.end(), forIndex);
				if (iter != movePointIdxs.end())
				{
					tmpAddPnt += movePoint;
                    forIndex++;
				}
				//添加点
				tmpVerts.push_back(tmpAddPnt);

				std::vector<int> tmpPreEdge, tmpCurEdge, tmpNextEdge;
				for (auto tmpperp : p.previous)
				{
					if (tmpperp > 0)tmpPreEdge.push_back(tmpperp);
				}
				for (auto tmpperp : p.edge)
				{
					if (tmpperp > 0)tmpCurEdge.push_back(tmpperp);
				}
				for (auto tmpperp : p.next)
				{
					if (tmpperp > 0)tmpNextEdge.push_back(tmpperp);
				}
				zeno::vec3i tmpTri = tmpDTris[tmpCommEdgeTrisidx];
				auto retTris = addtri(tmpTri, tmpPreEdge, tmpCurEdge, tmpNextEdge, tmpPntIdx, tmpPntIdx - 1, tmpPntIdx + 1);
				for (auto retTri : retTris)
				{
					tmpTris.push_back(retTri);
				}

				if (addOnce == true)
				{
					tmpCommEdgeTrisidx = p.tris2;
					tmpAddPnt = addPnts[tmpCommEdgeTrisidx];
					tmpPntIdx = tmpVerts.size();
					iter = std::find(movePointIdxs.begin(), movePointIdxs.end(), forIndex);
					if (iter != movePointIdxs.end())
					{
						tmpAddPnt += movePoint;
                        forIndex++;
					}
					//添加点
					tmpVerts.push_back(tmpAddPnt);

					zeno::vec3i tmpTri = tmpDTris[tmpCommEdgeTrisidx];

					int v0 = tmpTri[0];
					int v1 = tmpTri[1];
					int v2 = tmpTri[2];
					if (v0 < v1) std::swap(v0, v1);
					if (v0 < v2) std::swap(v0, v2);
					if (v1 < v2) std::swap(v1, v2);

					auto tv10 = p.edge[0];
					auto tv11 = p.edge[1];
					auto tv20 = p.next[0];
					auto tv21 = p.next[1];
					if (tv10 == tv20)
					{
						tmpTris.push_back({ tv11, tmpPntIdx, tv21 });
					}
					else if (tv10 == tv21)
					{
						tmpTris.push_back({ tv11, tmpPntIdx, tv20 });
					}
					else if (tv11 == tv20)
					{
						tmpTris.push_back({ tv10, tmpPntIdx, tv21 });
					}
					else if (tv11 == tv21)
					{
						tmpTris.push_back({ tv10, tmpPntIdx, tv20 });
					}
				}
				addOnce = false;
			}

			if (addPnts.size() > 0)
			{
				outPrim = std::make_shared<zeno::PrimitiveObject>();
				outPrim->verts->resize(tmpVerts.size());
				outPrim->tris->resize(tmpTris.size());
				for (int i = 0; i < tmpVerts.size(); i++)
				{
					outPrim->verts[i] = tmpVerts[i];
				}
				for (int i = 0; i < tmpTris.size(); i++) {
					outPrim->tris[i] = tmpTris[i];
				}
			}

			for (auto&& [ikey, pnt] : addPnts)
			{
				auto outPrim1 = std::make_shared<zeno::PrimitiveObject>();
				outPrim1->verts->push_back(pnt);
				outlist->arr.push_back(outPrim1);

			}
            
            zeno::log_info("tris size: {}, verts size: {}", outPrim->tris.size(), outPrim->verts.size());
           
            //outPrim->tris->clear();
            #if 1
            outlist->arr.push_back(outPrim);
            set_output("prim", outlist);
            //set_output("outs", std::make_shared<zeno::ListObject>());
            #else
            set_output("prim", outPrim);
            set_output("outs", outlist);
            #endif
        }
    }
};

ZENDEFNODE(PrimitiveEdit, {/*输入*/
                           {"prim","points", "move", "movePointIdx"},
                           /*输出*/
                           {"prim"},
                           /*参数*/
                           {/*{"bool", "Default", "false"},*/
                               {"int", "pointIndex", "0"},
                               {"int", "rowCount", "3"},
                               {"int", "columnCount", "5"},
                               {"enum Line Face", "Operate", "Face"},
                               {"string", "index", ""},
                               {"string", "area", ""},
                               {"int", "outline", "0"},
                               {"int", "inline", "0"},
                           },
                           /*类别*/
                           {"ViewUI"}});



struct PrimInsertionPoint : zeno::INode {
    virtual void apply() override {
        auto inPrim = get_input<zeno::PrimitiveObject>("prim");
		auto movePoint = get_input2<zeno::vec3f>("move");
		auto movePointIdx = get_input<zeno::ListObject>("movePointIdx")->get();
        auto pointList = get_input<zeno::ListObject>("pointList")->get();

        auto outPrim = std::make_shared<zeno::PrimitiveObject>();
        auto outlist = std::make_shared<zeno::ListObject>();

		std::vector<int> movePointIdxs;
		for (int i = 0; i < movePointIdx.size(); i++) {
			auto tmpIdx = dynamic_cast<zeno::NumericObject*>(movePointIdx[i].get());
			movePointIdxs.push_back(tmpIdx->get<int>());
		}

        if (inPrim->polys.size() > 0)
        {
			prim_triangulate(inPrim.get());
			inPrim->loops->clear();
			inPrim->polys->clear();
			inPrim->uvs->clear();
        }

        if (inPrim != nullptr) {
            // 获取所有点
            std::vector<zeno::vec3f> tmpPnts;
			for (int i = 0; i < pointList.size(); i++) {
				auto tmp1 = dynamic_cast<zeno::PrimitiveObject*>(pointList[i].get());
				for (int n = 0; n < tmp1->verts.size(); n++)
				{
                    tmpPnts.push_back(tmp1->verts[n]);
				}
			}

			// 判断鼠标平面
			auto tmpvx = fabs(tmpPnts[0][0] - tmpPnts[1][0]);
			auto tmpvy = fabs(tmpPnts[0][1] - tmpPnts[1][1]);
			auto tmpvz = fabs(tmpPnts[0][2] - tmpPnts[1][2]);
			auto tmpv = tmpvx < tmpvy ? 0 : 1;
			if (tmpv == 0)
				tmpv = tmpvx < tmpvz ? 0 : 2;
			else
				tmpv = tmpvy < tmpvz ? 1 : 2;

            // 判断在三角形中的点
			std::map<int, zeno::vec3f> addPnts;
			for (int i = 0; i < inPrim->verts->size(); i++) {
				zeno::vec3f A = inPrim->verts[inPrim->tris[i][0]];
				zeno::vec3f B = inPrim->verts[inPrim->tris[i][1]];
				zeno::vec3f C = inPrim->verts[inPrim->tris[i][2]];
                std::vector<zeno::vec3f> tmpTri = { A, B, C };
				for (int tmpPIdx = 0; tmpPIdx < tmpPnts.size(); tmpPIdx++)
				{
                    zeno::vec3f D = tmpPnts[tmpPIdx];
                    if (fabs(A[tmpv] - D[tmpv]) > 0.1)
                    {
                        continue;
                    }
#if 1
                    auto a1 = calculateArea(A, B, C);
                    auto a2 = calculateArea(A, B, D);
                    auto a3 = calculateArea(A, D, C);
                    auto a4 = calculateArea(D, B, C);
                    float ret = fabs(a1 - (a2 + a3 + a4));
                    if (ret < 0.001)
					{
                        std::cout << "tri idx: " << i << "  ==========In Plane Pnt: " 
                            << tmpPnts[tmpPIdx][0] << " " << tmpPnts[tmpPIdx][1] << " " << tmpPnts[tmpPIdx][2] << std::endl;
                        addPnts[i] = tmpPnts[tmpPIdx];
                    }
#else
					

					//if (dot_inplane_in(A, B, C, tmpPnts[tmpPIdx]))//判点是否在空间三角形上,包括边界,三点共线无意义
                    //if (dot_inplane_ex(A, B, C, tmpPnts[tmpPIdx]))//判点是否在空间三角形上,不包括边界,三点共线无意义
                    //if (dots_onplane(A, B, C, tmpPnts[tmpPIdx]) == true)//判四点共面
                    //if (calculatePntInTril(A, B, C, tmpPnts[tmpPIdx]))
                    //if (calculatePntInTrilEx(A, B, C, tmpPnts[tmpPIdx]))
					//if (is_in_3d_polygon(tmpTri, tmpPnts[tmpPIdx]))
					{
						char format_str[128] = { 0 };
						snprintf(format_str, sizeof(format_str) - 1, "A:%f, %f, %f : B:%f, %f, %f : C:%f, %f, %f : P:%f, %f, %f",
							A[0], A[1], A[2],
							B[0], B[1], B[2],
							C[0], C[1], C[2],
							D[0], D[1], D[2]);
						std::cout << format_str << std::endl;
                        //std::cout << "In Plane" << std::endl;
						//tmpP[tmpv] = calNewPointPosition(A, B, C)[tmpv];
						addPnts[i] = tmpPnts[tmpPIdx];
					}
#endif
				}
			}

			std::vector<zeno::vec3f> tmpVerts = inPrim->verts;
			std::vector<zeno::vec3i> tmpTris = inPrim->tris;
			std::map<int, zeno::vec3i> tmpDTris;

			//先减面
			int forIndex = 0;
			for (auto&& [ikey, pnt] : addPnts)
			{
				//记录被删除面在 addPnts中的索引和顶点索引
				tmpDTris[ikey] = tmpTris[ikey - forIndex];
				tmpTris.erase(tmpTris.begin() + ikey - forIndex);
				forIndex++;

			}

            // 找三角形共边
            std::vector<TrisSharedEdge> tmpDTrisEdges;
            std::vector<TriangleData> tmpDTrisData;
            for (auto&& [ikey, pnt] : tmpDTris)
            {
                tmpDTrisData.push_back({ ikey ,pnt });
            }

            int i1 = tmpDTris.size() - 1;
            int i2 = tmpDTrisEdges.size();
            while (i2 < i1)
            {
                auto tv = tmpDTrisData.back();
                tmpDTrisData.pop_back();
                tmpDTrisData.insert(tmpDTrisData.begin(), tv);

                //std::iter_swap(tmpDTrisData.begin(), tmpDTrisData.end()-1);
                tmpDTrisEdges = findSharedEdgeTriOrd(tmpDTrisData);
                i2 = tmpDTrisEdges.size();
            }

            bool addOnce = true;
            std::vector<int> addPntIdxs;
            for (auto p : tmpDTrisEdges)
            {
				int tmpCommEdgeTrisidx = addOnce == true ? p.tris1 : p.tris2;
                auto tmpAddPnt = addPnts[tmpCommEdgeTrisidx];

                int tmpPntIdx = tmpVerts.size();
                addPntIdxs.push_back(tmpPntIdx);
				auto iter = std::find(movePointIdxs.begin(), movePointIdxs.end(), forIndex);
				if (iter != movePointIdxs.end())
				{
					tmpAddPnt += movePoint;
				}
				//添加点
				tmpVerts.push_back(tmpAddPnt);

                std::vector<int> tmpPreEdge, tmpCurEdge, tmpNextEdge;
                for (auto tmpperp : p.previous)
                {
                    if (tmpperp > 0)tmpPreEdge.push_back(tmpperp);
                }
				for (auto tmpperp : p.edge)
				{
					if (tmpperp > 0)tmpCurEdge.push_back(tmpperp);
				}
				for (auto tmpperp : p.next)
				{
					if (tmpperp > 0)tmpNextEdge.push_back(tmpperp);
				}
                zeno::vec3i tmpTri = tmpDTris[tmpCommEdgeTrisidx];
                auto retTris = addtri(tmpTri, tmpPreEdge, tmpCurEdge, tmpNextEdge, tmpPntIdx, tmpPntIdx - 1, tmpPntIdx + 1);
				for (auto retTri : retTris)
				{
					tmpTris.push_back(retTri);
				}

                if (addOnce == true)
                {
					tmpCommEdgeTrisidx = p.tris2;
					tmpAddPnt = addPnts[tmpCommEdgeTrisidx];
					tmpPntIdx = tmpVerts.size();
                    addPntIdxs.push_back(tmpPntIdx);
					iter = std::find(movePointIdxs.begin(), movePointIdxs.end(), forIndex);
					if (iter != movePointIdxs.end())
					{
						tmpAddPnt += movePoint;
					}
					//添加点
					tmpVerts.push_back(tmpAddPnt);

					zeno::vec3i tmpTri = tmpDTris[tmpCommEdgeTrisidx];

					int v0 = tmpTri[0];
					int v1 = tmpTri[1];
					int v2 = tmpTri[2];
					if (v0 < v1) std::swap(v0, v1);
					if (v0 < v2) std::swap(v0, v2);
					if (v1 < v2) std::swap(v1, v2);

                    auto tv10 = p.edge[0];
                    auto tv11 = p.edge[1];
					auto tv20 = p.next[0];
					auto tv21 = p.next[1]; 
                    if (tv10 == tv20)
                    {
                        tmpTris.push_back({ tv11, tv21 , tmpPntIdx });
                    }
                    else if (tv10 == tv21)
                    {
                        tmpTris.push_back({ tv20, tv11, tmpPntIdx });
                    }
                    else if (tv11 == tv20)
                    {
                        tmpTris.push_back({ tv10, tv21 , tmpPntIdx});
					}
					else if (tv11 == tv21)
					{
						tmpTris.push_back({ tv10, tv20 , tmpPntIdx});
					}
                }
                addOnce = false;                
            }

			if (addPnts.size() > 0)
			{
				outPrim = std::make_shared<zeno::PrimitiveObject>();
				outPrim->verts->resize(tmpVerts.size());
				outPrim->tris->resize(tmpTris.size());
				for (int i = 0; i < tmpVerts.size(); i++)
				{
					outPrim->verts[i] = tmpVerts[i];
				}
				for (int i = 0; i < tmpTris.size(); i++) {
					outPrim->tris[i] = tmpTris[i];
				}
                std::vector<zeno::vec2i> tmpAddLines;
                for (int i = 1; i < addPntIdxs.size(); i++) {
                    int line1 = addPntIdxs[i - 1];
                    int line2 = addPntIdxs[i];
					if (line1 < line2) std::swap(line1, line2);
                    tmpAddLines.push_back({ line1, line2 });
				}

				

				std::vector<zeno::vec2i> edges;
                std::vector<int> fixedEdges;
				edges.clear();
				std::set<std::pair<int, int>> edge_set;
				int n_tri = outPrim->tris.size();
				for (int i = 0; i < n_tri; ++i) {

					int v0 = outPrim->tris[i][0];
					int v1 = outPrim->tris[i][1];
					int v2 = outPrim->tris[i][2];

					if (v0 < v1) std::swap(v0, v1);
					if (v0 < v2) std::swap(v0, v2);
					if (v1 < v2) std::swap(v1, v2);

					auto p01 = std::make_pair(v0, v1);
					auto p02 = std::make_pair(v0, v2);
					auto p12 = std::make_pair(v1, v2);

					if (!edge_set.count(p01)) {
						edge_set.insert(p01);
                        edges.push_back({ v0, v1 });                        
						for (auto p : tmpAddLines) {
							if ((p[0] == v0) && (p[1] == v1))
							{
                                fixedEdges.push_back(i);
                                break;
							}							
						}
					}
					if (!edge_set.count(p02)) {
						edge_set.insert(p02);
                        edges.push_back({ v0, v2 });
						for (auto p : tmpAddLines) {
							if ((p[0] == v0) && (p[1] == v2))
							{
                                fixedEdges.push_back(i);
								break;
							}
						}
					}
					if (!edge_set.count(p12)) {
						edge_set.insert(p12);
                        edges.push_back({ v1, v2 });
						for (auto p : tmpAddLines) {
							if ((p[0] == v1) && (p[1] == v2))
							{
                                fixedEdges.push_back(i);
								break;
							}
						}
					}
				}
				outPrim->lines = edges;
				outPrim->lines.add_attr<float>("edge");
				auto& attr_arr = outPrim->lines.attr<float>("edge");
                for (auto p: fixedEdges)
                {
                    attr_arr[p] = 1;
                }
				zeno::log_info("{}", edges.size());
			}

			for (auto&& [ikey, pnt] : addPnts)
			{
				auto outPrim1 = std::make_shared<zeno::PrimitiveObject>();
				outPrim1->verts->push_back(pnt);
				outlist->arr.push_back(outPrim1);

			}

			zeno::log_info("tris size: {}, verts size: {}", outPrim->tris.size(), outPrim->verts.size());
        }
        set_output("points", outlist);
        set_output("prim", outPrim);
    }
};

ZENDEFNODE(PrimInsertionPoint, {/*输入*/
                                  {"pointList","prim","move","movePointIdx"},
                                  /*输出*/
                                  {"prim","points"},
                                  /*参数*/
                                  {{"int", "calcount", "0"}},
                                  /*类别*/
                                  {"ViewUI"} });


struct ChangeSelectPointPos : zeno::INode {
    virtual void apply() override {
        auto inPrim1 = get_input<zeno::PrimitiveObject>("prim1");
        auto inPrim2 = get_input<zeno::PrimitiveObject>("prim2");

        int pIdx = inPrim2->attr<int>("pIdx")[0];
        auto pos = inPrim2->verts[0];

        inPrim1->verts[pIdx] = pos;

        set_output("prim", std::move(inPrim1));
    }
};

ZENDEFNODE(ChangeSelectPointPos, {/*输入*/
                                  {"prim1", "prim2"},
                                  /*输出*/
                                  {"prim"},
                                  /*参数*/
                                  {},
                                  /*类别*/
                                  {"ViewUI"}});


// Set Attr
struct PrimSetAttrEx : zeno::INode {
	void apply() override {

		auto prim = get_input<zeno::PrimitiveObject>("prim");
		auto value = get_input<zeno::NumericObject>("value");
		auto name = get_input2<std::string>("name");
		auto type = get_input2<std::string>("type");
		auto index = get_input<zeno::ListObject>("index")->get();
		auto method = get_input<zeno::StringObject>("method")->get();
        
		std::visit(
			[&](auto ty) {
				using T = decltype(ty);

				std::map<int, T> idxs;
				for (auto p : index)
				{
					auto tmpIdx = dynamic_cast<zeno::NumericObject*>(p.get());
                    idxs[tmpIdx->get<int>()] = value->get<T>();
				}

				if (method == "vert") {
					if (!prim->has_attr(name)) {
						prim->add_attr<T>(name);
					}
					auto& attr_arr = prim->attr<T>(name);

					auto val = value->get<T>();
					if (idxs.size() <= attr_arr.size()) {						
						for (auto&& [ikey, iVal] : idxs)
						{
							attr_arr[ikey] = iVal;
						}
					}
				}
				else if (method == "tri") {
					if (!prim->tris.has_attr(name)) {
						prim->tris.add_attr<T>(name);
					}
					auto& attr_arr = prim->tris.attr<T>(name);
					auto val = value->get<T>();
					if (idxs.size() <= attr_arr.size()) {
						for (auto&& [ikey, iVal] : idxs)
						{
							attr_arr[ikey] = iVal;
						}
					}
				}
				else if (method == "line") {
					if (!prim->lines.has_attr(name)) {
						prim->lines.add_attr<T>(name);
					}
					auto& attr_arr = prim->lines.attr<T>(name);
					auto val = value->get<T>();
					if (idxs.size() <= attr_arr.size()) {
						for (auto&& [ikey, iVal] : idxs)
						{
							attr_arr[ikey] = iVal;
						}
					}
				}
				else {
					throw zeno::Exception("bad type: " + method);
				}
			},
            zeno::enum_variant<std::variant<float, zeno::vec2f, zeno::vec3f, zeno::vec4f, int, zeno::vec2i, zeno::vec3i, zeno::vec4i>>(
                zeno::array_index({ "float", "vec2f", "vec3f", "vec4f", "int", "vec2i", "vec3i", "vec4i" }, type)));

		set_output("prim", std::move(prim));
	}
};
ZENDEFNODE(PrimSetAttrEx,
	{ /* inputs: */ {
			"prim",
			{"int", "value", "0"},
			{"string", "name", "edge"},
			{"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
			{"enum vert tri line", "method", "tri"},
			{"list", "index"},
		}, /* outputs: */ {
			"prim",
		}, /* params: */ {
		}, /* category: */ {
			"erode",
		} });


struct CMDZenoMeshes : zeno::INode {
	virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        //'.\Zeno Meshes.exe' C://Users//zs//Desktop//car111.obj  -t 24 -b

		for (auto c : path) {
			if (c == '/') c = '\\';
		}

		auto cmd = (std::string)"\"Zeno Meshes.exe\" " + path + " -t 24 -b";
		int er = std::system(cmd.c_str());

		auto result = std::make_shared<zeno::NumericObject>();
		result->set(er);

		zeno::log_info("----- CMD {}", cmd);
		zeno::log_info("----- Exec Result {}", er);

		set_output("result", std::move(result));
	}
};

ZENDEFNODE(CMDZenoMeshes, {/*输入*/
								  {{"writepath", "path"}},
								  /*输出*/
								  {"result"},
								  /*参数*/
								  {},
								  /*类别*/
								  {"ViewUI"} });


} // namespace