#include "apphelper.h"
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>
#include "util/log.h"
#include <zenomodel/include/uihelper.h>
#include "common_def.h"
#include "../startup/zstartup.h"
#include "variantptr.h"
#include "viewport/displaywidget.h"


QModelIndexList AppHelper::getSubInOutNode(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& sockName, bool bInput)
{
    //get SubInput/SubOutput Node by socket of a subnet node.
    const QModelIndexList& indices = pModel->searchInSubgraph(bInput ? "SubInput" : "SubOutput", subgIdx);
    QModelIndexList result;
    for (const QModelIndex &idx_ : indices)
    {
        const QString& subInputId = idx_.data(ROLE_OBJID).toString();
        if ((sockName == "DST" && !bInput) || (sockName == "SRC" && bInput))
        {
            result.append(idx_);
            continue;
        }
        const PARAMS_INFO &params = idx_.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        if (params["name"].value == sockName)
        {
            result.append(idx_);
            // there muse be a unique SubOutput for specific name.
            return result;
        }
    }
    return result;
}

QLinearGradient AppHelper::colorString2Grad(const QString& colorStr)
{
    QLinearGradient grad;
    QStringList L = colorStr.split("\n", QtSkipEmptyParts);
    ZASSERT_EXIT(!L.isEmpty(), grad);

    bool bOk = false;
    int n = L[0].toInt(&bOk);
    ZASSERT_EXIT(bOk && n == L.size() - 1, grad);
    for (int i = 1; i <= n; i++)
    {
        QStringList color_info = L[i].split(" ", QtSkipEmptyParts);
        ZASSERT_EXIT(color_info.size() == 4, grad);

        float f = color_info[0].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);
        float r = color_info[1].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);
        float g = color_info[2].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);
        float b = color_info[3].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);

        QColor clr;
        clr.setRgbF(r, g, b);
        grad.setColorAt(f, clr);
    }
    return grad;
}

INPUT_SOCKET AppHelper::getInputSocket(const QPersistentModelIndex& index, const QString& inSock, bool& exist)
{
    INPUT_SOCKETS inputs = index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    //assuming inSock is valid...
    INPUT_SOCKET _inSocket;
    if (inputs.find(inSock) == inputs.end())
    {
        exist = false;
        return _inSocket;
    }
    _inSocket = inputs[inSock];
    exist = true;
    return _inSocket;
}

void AppHelper::ensureSRCDSTlastKey(INPUT_SOCKETS& inputs, OUTPUT_SOCKETS& outputs)
{
    if (inputs.lastKey() != "SRC")
    {
        //ensure that the "SRC" is the last key in sockets.
        INPUT_SOCKET srcSocket = inputs["SRC"];
        inputs.remove("SRC");
        inputs.insert("SRC", srcSocket);
    }
    if (outputs.lastKey() != "DST")
    {
        //ensure that the "DST" is the last key in sockets.
        OUTPUT_SOCKET dstSocket = outputs["DST"];
        outputs.remove("DST");
        outputs.insert("DST", dstSocket);
    }
}

QString AppHelper::nativeWindowTitle(const QString& currentFilePath)
{
    QString ver = QString::fromStdString(getZenoVersion());
    if (currentFilePath.isEmpty())
    {
        QString title = QString("Zeno Editor (%1)").arg(ver);
        return title;
    }
    else
    {
        QString title = QString::fromUtf8("%1 - Zeno Editor (%2)").arg(currentFilePath).arg(ver);
        return title;
    }
}

void AppHelper::socketEditFinished(QVariant newValue, QPersistentModelIndex nodeIdx, QPersistentModelIndex paramIdx) {
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZenoMainWindow *main = zenoApp->getMainWindow();
    if (!pModel || !main)
        return;
    if (nodeIdx.data(ROLE_OBJNAME).toString() == "LightNode" &&
        nodeIdx.data(ROLE_OPTIONS).toInt() == OPT_VIEW &&
        (main->isAlways() || main->isAlwaysLightCamera()))
    {
        //only update nodes.
        zeno::scope_exit sp([=] { pModel->setApiRunningEnable(true);});
        pModel->setApiRunningEnable(false);

        QAbstractItemModel *paramsModel = const_cast<QAbstractItemModel *>(paramIdx.model());
        ZASSERT_EXIT(paramsModel);
        paramsModel->setData(paramIdx, newValue, ROLE_PARAM_VALUE);
        modifyLightData(nodeIdx);
    } else {
        int ret = pModel->ModelSetData(paramIdx, newValue, ROLE_PARAM_VALUE);
    }
}

void AppHelper::modifyLightData(QPersistentModelIndex nodeIdx) {
    QStandardItemModel *viewParams = QVariantPtr<QStandardItemModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams);
    QStandardItem *inv_root = viewParams->invisibleRootItem();
    ZASSERT_EXIT(inv_root);
    QStandardItem *inputsItem = inv_root->child(0);
    std::string name = nodeIdx.data(ROLE_OBJID).toString().toStdString();
    const UI_VECTYPE posVec = inputsItem->child(0)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    const UI_VECTYPE scaleVec = inputsItem->child(1)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    const UI_VECTYPE rotateVec = inputsItem->child(2)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    const UI_VECTYPE colorVec = inputsItem->child(4)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    float posX = posVec[0];
    float posY = posVec[1];
    float posZ = posVec[2];
    float scaleX = scaleVec[0];
    float scaleY = scaleVec[1];
    float scaleZ = scaleVec[2];
    float rotateX = rotateVec[0];
    float rotateY = rotateVec[1];
    float rotateZ = rotateVec[2];
    float r = colorVec[0];
    float g = colorVec[1];
    float b = colorVec[2];

    float intensity = inputsItem->child(5)->index().data(ROLE_PARAM_VALUE).value<float>();
    zeno::vec3f pos = zeno::vec3f(posX, posY, posZ);
    zeno::vec3f scale = zeno::vec3f(scaleX, scaleY, scaleZ);
    zeno::vec3f rotate = zeno::vec3f(rotateX, rotateY, rotateZ);
    auto verts = ZenoLights::computeLightPrim(pos, rotate, scale);

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget *> views = pWin->viewports();
    for (auto pDisplay : views)
    {
        Zenovis* pZenovis = pDisplay->getZenoVis();
        ZASSERT_EXIT(pZenovis);

        auto scene = pZenovis->getSession()->get_scene();
        ZASSERT_EXIT(scene);

        std::shared_ptr<zeno::IObject> obj;
        for (auto const &[key, ptr] : scene->objectsMan->lightObjects) {
            if (key.find(name) != std::string::npos) {
                obj = ptr;
                name = key;
            }
        }
        auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj.get());

        if (prim_in) {
            auto &prim_verts = prim_in->verts;
            prim_verts[0] = verts[0];
            prim_verts[1] = verts[1];
            prim_verts[2] = verts[2];
            prim_verts[3] = verts[3];
            prim_in->verts.attr<zeno::vec3f>("clr")[0] = zeno::vec3f(r, g, b) * intensity;

            prim_in->userData().setLiterial<zeno::vec3f>("pos", zeno::vec3f(posX, posY, posZ));
            prim_in->userData().setLiterial<zeno::vec3f>("scale", zeno::vec3f(scaleX, scaleY, scaleZ));
            prim_in->userData().setLiterial<zeno::vec3f>("rotate", zeno::vec3f(rotateX, rotateY, rotateZ));
            if (prim_in->userData().has("intensity")) {
                prim_in->userData().setLiterial<zeno::vec3f>("color", zeno::vec3f(r, g, b));
                prim_in->userData().setLiterial<float>("intensity", std::move(intensity));
            }

            scene->objectsMan->needUpdateLight = true;
            pDisplay->setSimpleRenderOption();
            zenoApp->getMainWindow()->updateViewport();
        } else {
            zeno::log_info("modifyLightData not found {}", name);
        }
    }
}

void AppHelper::initLaunchCacheParam(LAUNCH_PARAM& param)
{
    QSettings settings(zsCompanyName, zsEditor);
    param.enableCache = settings.value("zencache-enable").isValid() ? settings.value("zencache-enable").toBool() : true;
    param.tempDir = settings.value("zencache-autoremove", true).isValid() ? settings.value("zencache-autoremove", true).toBool() : false;
    param.cacheDir = settings.value("zencachedir").isValid() ? settings.value("zencachedir").toString() : "";
    param.cacheNum = settings.value("zencachenum").isValid() ? settings.value("zencachenum").toInt() : 1;
}
