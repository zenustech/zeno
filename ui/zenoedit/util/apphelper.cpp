#ifdef ZENO_WITH_PYTHON3
#include <Python.h>
#endif
#include "apphelper.h"
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>
#include "util/log.h"
#include <zenomodel/include/uihelper.h>
#include "common_def.h"
#include "../startup/zstartup.h"
#include "variantptr.h"
#include "viewport/displaywidget.h"
#include "common.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include "viewport/zoptixviewport.h"


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
    int ret = pModel->ModelSetData(paramIdx, newValue, ROLE_PARAM_VALUE);
}

void AppHelper::modifyOptixObjDirectly(QVariant newValue, QPersistentModelIndex nodeIdx, QPersistentModelIndex paramIdx, bool editByPropPanel)
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZenoMainWindow* main = zenoApp->getMainWindow();
    if (nodeIdx.data(ROLE_OBJNAME).toString() == "LightNode" &&
        nodeIdx.data(ROLE_OPTIONS).toInt() == OPT_VIEW &&
        (main->isAlways() && (main->runtype() == RunALL || main->runtype() == RunLightCamera) || editByPropPanel))
    {
        modifyLightData(newValue, nodeIdx, paramIdx);
    }
    else if ((nodeIdx.data(ROLE_OBJNAME).toString() == "CameraNode" ||
        nodeIdx.data(ROLE_OBJNAME).toString() == "MakeCamera" ||
        nodeIdx.data(ROLE_OBJNAME).toString() == "TargetCamera") &&
        ((nodeIdx.data(ROLE_OPTIONS).toInt() == OPT_VIEW && (main->isAlways() && (main->runtype() == RunALL || main->runtype() == RunLightCamera))) || editByPropPanel)
        )
    {
        modifyOptixCameraPropDirectly(newValue, nodeIdx, paramIdx);
    }
    else if (( main->isAlways() || editByPropPanel))
    {
        socketEditFinished(newValue, nodeIdx, paramIdx);
    }
}

void AppHelper::modifyOptixCameraPropDirectly(QVariant newValue, QPersistentModelIndex nodeIdx, QPersistentModelIndex paramIdx)
{
    int apertureSocketIdx = 0, distancePlaneSocketIdx = 0;
    if (nodeIdx.data(ROLE_OBJNAME).toString() == "CameraNode")
    {
        apertureSocketIdx = 4; distancePlaneSocketIdx = 5;
    }else if (nodeIdx.data(ROLE_OBJNAME).toString() == "MakeCamera" || nodeIdx.data(ROLE_OBJNAME).toString() == "TargetCamera")
    {
        apertureSocketIdx = 6; distancePlaneSocketIdx = 7;
    }
    QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams);
    QStandardItem* inv_root = viewParams->invisibleRootItem();
    ZASSERT_EXIT(inv_root);
    QStandardItem* inputsItem = inv_root->child(0);
    float cameraAperture = inputsItem->child(apertureSocketIdx)->index().data(ROLE_PARAM_VALUE).value<float>();
    float cameraDistancePlane = inputsItem->child(distancePlaneSocketIdx)->index().data(ROLE_PARAM_VALUE).value<float>();
    QString paramName = paramIdx.data(ROLE_PARAM_NAME).toString();
    if (paramName == "aperture")
        cameraAperture = newValue.value<float>();
    else if (paramName == "focalPlaneDistance")
        cameraDistancePlane = newValue.value<float>();

    UI_VECTYPE skipParam(2, 0);
    if (!inputsItem->child(apertureSocketIdx)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[0] = 1;
    if (!inputsItem->child(distancePlaneSocketIdx)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[1] = 1;

    QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
    for (auto pDisplay : views) {
        if (pDisplay->isGLViewport())
            continue;
        ZASSERT_EXIT(pDisplay);
        if (ZOptixViewport* optixViewport = pDisplay->optixViewport())
            optixViewport->updateCameraProp(cameraAperture, cameraDistancePlane, skipParam);
    }
}

VideoRecInfo AppHelper::getRecordInfo(const ZENO_RECORD_RUN_INITPARAM& param)
{
    VideoRecInfo recInfo;
    recInfo.bitrate = param.iBitrate;
    recInfo.fps = param.iFps;
    recInfo.frameRange = { param.iSFrame, param.iSFrame + param.iFrame - 1 };
    recInfo.numMSAA = 0;
    recInfo.numOptix = param.iSample;
    recInfo.audioPath = param.audioPath;
    recInfo.record_path = param.sPath;
    recInfo.videoname = param.videoName;
    recInfo.bExportVideo = param.isExportVideo;
    recInfo.needDenoise = param.needDenoise;
    recInfo.exitWhenRecordFinish = param.exitWhenRecordFinish;

    if (!param.sPixel.isEmpty())
    {
        QStringList tmpsPix = param.sPixel.split("x");
        int pixw = tmpsPix.at(0).toInt();
        int pixh = tmpsPix.at(1).toInt();
        recInfo.res = { (float)pixw, (float)pixh };

        //viewWidget->setFixedSize(pixw, pixh);
        //viewWidget->setCameraRes(QVector2D(pixw, pixh));
        //viewWidget->updatePerspective();
    }
    else {
        recInfo.res = { (float)1000, (float)680 };
        //viewWidget->setMinimumSize(1000, 680);
    }

    return recInfo;
}

void AppHelper::modifyLightData(QVariant newValue, QPersistentModelIndex nodeIdx, QPersistentModelIndex paramIdx) {
    QStandardItemModel *viewParams = QVariantPtr<QStandardItemModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams);
    QStandardItem *inv_root = viewParams->invisibleRootItem();
    ZASSERT_EXIT(inv_root);
    QStandardItem *inputsItem = inv_root->child(0);
    QString name = nodeIdx.data(ROLE_OBJID).toString();
    UI_VECTYPE posVec = inputsItem->child(0)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    UI_VECTYPE scaleVec = inputsItem->child(1)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    UI_VECTYPE rotateVec = inputsItem->child(2)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    UI_VECTYPE colorVec = inputsItem->child(4)->index().data(ROLE_PARAM_VALUE).value<UI_VECTYPE>();
    float intensity = inputsItem->child(6)->index().data(ROLE_PARAM_VALUE).value<float>();
    QString paramName = paramIdx.data(ROLE_PARAM_NAME).toString();
    if (paramName == "position")
        posVec = newValue.value<UI_VECTYPE>();
    else if (paramName == "scale")
        scaleVec = newValue.value<UI_VECTYPE>();
    else if (paramName == "rotate")
        rotateVec = newValue.value<UI_VECTYPE>();
    else if (paramName == "color")
        colorVec = newValue.value<UI_VECTYPE>();
    else if (paramName == "intensity")
        intensity = newValue.value<float>();
    if (posVec.size() == 0 || scaleVec.size() == 0 || rotateVec.size() == 0 || colorVec.size() == 0)
        return;

    UI_VECTYPE skipParam(5, 0);
    if (!inputsItem->child(0)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[0] = 1;
    if (!inputsItem->child(1)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[1] = 1;
    if (!inputsItem->child(2)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[2] = 1;
    if (!inputsItem->child(4)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[3] = 1;
    if (!inputsItem->child(6)->index().data(ROLE_PARAM_LINKS).value<PARAM_LINKS>().isEmpty())
        skipParam[4] = 1;

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget *> views = pWin->viewports();
    for (auto pDisplay : views)
    {
        if (pDisplay->isGLViewport())
            continue;
        if (ZOptixViewport* optixViewport = pDisplay->optixViewport())
            optixViewport->modifyLightData(posVec, scaleVec, rotateVec, colorVec, intensity, name, skipParam);
    }
}

QVector<QString> AppHelper::getKeyFrameProperty(const QVariant& val)
{
    QVector<QString> ret;
    if (val.canConvert<CURVES_DATA>())
    {
        CURVES_DATA curves = val.value<CURVES_DATA>();
        ret.resize(curves.size());
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, ret);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, ret);
        for (int i = 0; i < ret.size(); i++)
        {
            QString property = "null";
            const QString& key = curve_util::getCurveKey(i);
            if (curves.contains(key))
            {
                CURVE_DATA data = curves[key];

                if (data.visible)
                {
                    property = "false";
                    int x = timeline->value();
                    for (const auto& p : data.points) {
                        int px = p.point.x();
                        if (px == x) {
                            property = "true";
                            break;
                        }
                    }
                }

            }
            ret[i] = property;
        }
    }
    if (ret.isEmpty())
        ret << "null";
    return ret;
}

bool AppHelper::getCurveValue(QVariant& val)
{
    if (val.canConvert<CURVES_DATA>())
    {
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, false);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, false);
        int nFrame = timeline->value();
        CURVES_DATA curves = val.value<CURVES_DATA>();
        if (curves.size() > 1)
        {
            UI_VECTYPE newVal;
            newVal.resize(curves.size());
            for (int i = 0; i < newVal.size(); i++)
            {
                QString key = curve_util::getCurveKey(i);
                if (curves.contains(key))
                {
                    newVal[i] = curves[key].eval(nFrame);
                }
            }
            val = QVariant::fromValue(newVal);
        }
        else if (curves.contains("x"))
        {
            val = QVariant::fromValue(curves["x"].eval(nFrame));
        }
        return true;
    }
    return false;
}

bool AppHelper::updateCurve(QVariant oldVal, QVariant& newValue)
{
    if (oldVal.canConvert<CURVES_DATA>())
    {
        bool bUpdate = false;
        CURVES_DATA curves = oldVal.value<CURVES_DATA>();
        UI_VECTYPE datas;
        //vec
        if (newValue.canConvert<UI_VECTYPE>())
        {
            datas = newValue.value<UI_VECTYPE>();
        }
        //float
        else
        {
            datas << newValue.toFloat();
        }
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, false);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, false);
        int nFrame = timeline->value();
        for (int i = 0; i < datas.size(); i++) {
            QString key = curve_util::getCurveKey(i);
            if (curves.contains(key))
            {
                CURVE_DATA& curve = curves[key];
                QPointF pos(nFrame, datas.at(i));
                if (curve_util::updateCurve(pos, curve))
                    bUpdate = true;
            }
        }
        newValue = QVariant::fromValue(curves);
        return bUpdate;
    }
    return true;
}

void AppHelper::initLaunchCacheParam(LAUNCH_PARAM& param)
{
    QSettings settings(zsCompanyName, zsEditor);
    param.enableCache = settings.value("zencache-enable").isValid() ? settings.value("zencache-enable").toBool() : true;
    param.tempDir = settings.value("zencache-autoremove").isValid() ? settings.value("zencache-autoremove").toBool() : false;
    param.cacheDir = settings.value("zencachedir").isValid() ? settings.value("zencachedir").toString() : "";
    param.cacheNum = settings.value("zencachenum").isValid() ? settings.value("zencachenum").toInt() : 1;
    param.autoCleanCacheInCacheRoot = settings.value("zencache-autoclean").isValid() ? settings.value("zencache-autoclean").toBool() : true;
}

bool AppHelper::openZsgAndRun(const ZENO_RECORD_RUN_INITPARAM& param, LAUNCH_PARAM launchParam)
{
    auto pGraphs = zenoApp->graphsManagment();
    IGraphsModel* pModel = pGraphs->openZsgFile(param.sZsgPath);
    if (!pModel)
        return false;

    if (!param.subZsg.isEmpty())
    {
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        for (auto subgFilepath : param.subZsg.split(","))
        {
            zenoApp->graphsManagment()->importGraph(subgFilepath);
        }
        QModelIndex mainGraphIdx = pGraphsModel->index("main");

        for (QModelIndex subgIdx : pGraphsModel->subgraphsIndice())
        {
            QString subgName = subgIdx.data(ROLE_OBJNAME).toString();
            if (subgName == "main" || subgName.isEmpty())
            {
                continue;
            }
            QString subgNodeId = NodesMgr::createNewNode(pGraphsModel, mainGraphIdx, subgName, QPointF(500, 500));
            QModelIndex subgNodeIdx = pGraphsModel->index(subgNodeId, mainGraphIdx);
            STATUS_UPDATE_INFO info;
            info.role = ROLE_OPTIONS;
            info.oldValue = subgNodeIdx.data(ROLE_OPTIONS).toInt();
            info.newValue = subgNodeIdx.data(ROLE_OPTIONS).toInt() | OPT_VIEW;
            pGraphsModel->updateNodeStatus(subgNodeId, info, mainGraphIdx, true);
        }
    }
    zeno::getSession().globalComm->clearState();
    launchParam.projectFps = pGraphs->timeInfo().timelinefps;

    //set userdata from zsg
    auto& ud = zeno::getSession().userData();
    ud.set2("optix_show_background", pGraphs->userdataInfo().optix_show_background);

    if (!param.paramsJson.isEmpty())
    {
        //parse paramsJson
        rapidjson::Document configDoc;
        configDoc.Parse(param.paramsJson.toUtf8());
        if (!configDoc.IsObject())
        {
            zeno::log_error("config file is corrupted");
        }
        FuckQMap<QString, CommandParam> commands = pGraphs->currentModel()->commandParams();
        for (auto& [key, param] : commands)
        {
            if (configDoc.HasMember(param.name.toUtf8()))
            {
                param.value = UiHelper::parseJson(configDoc[param.name.toStdString().c_str()], nullptr);
                param.bIsCommand = true;
            }
            pGraphs->currentModel()->updateCommandParam(key, param);
        }
    }

    launchProgram(pModel, launchParam);
    return true;
}

void AppHelper::dumpTabsToZsg(QDockWidget* dockWidget, RAPIDJSON_WRITER& writer) {
    if (ZTabDockWidget* tabDockwidget = qobject_cast<ZTabDockWidget*>(dockWidget))
    {
        for (int i = 0; i < tabDockwidget->count(); i++)
        {
            QWidget* wid = tabDockwidget->widget(i);
            if (qobject_cast<DockContent_Parameter*>(wid)) {
                writer.String("Parameter");
            }
            else if (qobject_cast<DockContent_Editor*>(wid)) {
                writer.String("Editor");
            }
            else if (qobject_cast<DockContent_View*>(wid)) {
                DockContent_View* pView = qobject_cast<DockContent_View*>(wid);
                auto dpwid = pView->getDisplayWid();
                ZASSERT_EXIT(dpwid);
                auto vis = dpwid->getZenoVis();
                ZASSERT_EXIT(vis);
                auto session = vis->getSession();
                ZASSERT_EXIT(session);
                writer.StartObject();
                if (pView->isGLView()) {
                    auto [r, g, b] = session->get_background_color();
                    writer.Key("type");
                    writer.String("View");
                    writer.Key("backgroundcolor");
                    writer.StartArray();
                    writer.Double(r);
                    writer.Double(g);
                    writer.Double(b);
                    writer.EndArray();
                }
                else {
                    writer.Key("type");
                    writer.String("Optix");
                }
                std::tuple<int, int, bool> displayinfo = pView->getOriginWindowSizeInfo();
                writer.Key("blockwindow");
                writer.Bool(std::get<2>(displayinfo));
                writer.Key("resolutionX");
                writer.Int(std::get<0>(displayinfo));
                writer.Key("resolutionY");
                writer.Int(std::get<1>(displayinfo));
                writer.Key("resolution-combobox-index");
                writer.Int(pView->curResComboBoxIndex());
                writer.EndObject();
            }
            else if (qobject_cast<ZenoSpreadsheet*>(wid)) {
                writer.String("Data");
            }
            else if (qobject_cast<DockContent_Log*>(wid)) {
                writer.String("Logger");
            }
            else if (qobject_cast<ZenoLights*>(wid)) {
                writer.String("Light");
            }
        }
    }
}

void AppHelper::pythonExcute(const QString& code)
{
#ifdef ZENO_WITH_PYTHON3
    std::string stdOutErr =
        "import sys\n\
class CatchOutErr:\n\
    def __init__(self):\n\
        self.value = ''\n\
    def write(self, txt):\n\
        self.value += txt\n\
    def flush(self):\n\
        pass\n\
catchOutErr = CatchOutErr()\n\
sys.stdout = catchOutErr\n\
sys.stderr = catchOutErr\n\
"; //this is python code to redirect stdouts/stderr

    //Py_Initialize();
    PyObject* pModule = PyImport_AddModule("__main__"); //create main module
    PyRun_SimpleString(stdOutErr.c_str()); //invoke code to redirect

    //Py_Initialize();
    if (PyRun_SimpleString(code.toUtf8()) < 0) {
        zeno::log_warn("Python Script run failed");
    }
    PyObject* catcher = PyObject_GetAttrString(pModule, "catchOutErr"); //get our catchOutErr created above
    PyObject* output = PyObject_GetAttrString(catcher, "value"); //get the stdout and stderr from our catchOutErr object
    if (output != Py_None)
    {
        QString str = QString::fromStdString(_PyUnicode_AsString(output));
        QStringList lst = str.split('\n');
        for (const auto& line : lst)
        {
            if (!line.isEmpty())
            {
                if (zenoApp->isUIApplication())
                    ZWidgetErrStream::appendFormatMsg(line.toStdString());
            }
        }
    }
#else
    zeno::log_warn("The option 'ZENO_WITH_PYTHON3' should be ON");
#endif
}

void AppHelper::generatePython(const QString& id)
{
    auto main = zenoApp->getMainWindow();
    ZASSERT_EXIT(main);

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();

    TIMELINE_INFO tinfo = main->timelineInfo();

    LAUNCH_PARAM launchParam;
    launchParam.beginFrame = tinfo.beginFrame;
    launchParam.endFrame = tinfo.endFrame;
    QString path = pModel->filePath();
    path = path.left(path.lastIndexOf("/"));
    launchParam.zsgPath = path;

    launchParam.projectFps = main->timelineInfo().timelinefps;
    launchParam.generator = id;

    AppHelper::initLaunchCacheParam(launchParam);
    launchProgram(pModel, launchParam);
}
