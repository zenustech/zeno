#include "apphelper.h"
#include "util/log.h"
#include "uicommon.h"
#include "../startup/zstartup.h"
#include "variantptr.h"
#include "viewport/displaywidget.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include "viewport/zoptixviewport.h"
#include "viewport/zenovis.h"
#include "widgets/ztimeline.h"
#include "util/curveutil.h"
#include "layout/docktabcontent.h"
#include "layout/zdockwidget.h"
#include "util/uihelper.h"


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

QVector<QString> AppHelper::getKeyFrameProperty(const QVariant& val)
{
    QVector<QString> ret;
    bool bValid = false;
    zeno::CurvesData curves = UiHelper::getCurvesFromQVar(val, &bValid);
    if (curves.empty() || !bValid)
        return ret;

    ret.resize(curves.size());

    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, ret);
    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline, ret);
    for (int i = 0; i < ret.size(); i++)
    {
        QString property = "null";
        const QString& key = curve_util::getCurveKey(i);
        const std::string& skey = key.toStdString();
        if (curves.contains(skey))
        {
            CURVE_DATA data = curve_util::toLegacyCurve(curves[skey]);
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
            curves[skey] = curve_util::fromLegacyCurve(data);
        }
        ret[i] = property;
    }

    if (ret.isEmpty())
        ret << "null";
    return ret;
}

bool AppHelper::getCurveValue(QVariant& val)
{
    bool bValid = false;
    zeno::CurvesData curves = UiHelper::getCurvesFromQVar(val, &bValid);
    if (curves.empty() || !bValid)
        return false;

    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, false);
    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline, false);
    int nFrame = timeline->value();
    if (val.canConvert<UI_VECSTRING>())
    {
        auto strVec = val.value<UI_VECSTRING>();
        UI_VECTYPE newVal;
        newVal.resize(strVec.size());
        for (int i = 0; i < newVal.size(); i++)
        {
            const auto& key = curve_util::getCurveKey(i).toStdString();
            if (curves.contains(key))
            {
                newVal[i] = curves[key].eval(nFrame);
            }
            else
            {
                newVal[i] = strVec[i].toFloat();
            }
        }
        val = QVariant::fromValue(newVal);
        return true;
    }
    else if (val.type() == QVariant::String && curves.contains("x"))
    {
        val = QVariant::fromValue(curves["x"].eval(nFrame));
        return true;
    }
    return false;
}

bool AppHelper::updateCurve(QVariant oldVal, QVariant& newValue)
{
    bool bValid = false;
    zeno::CurvesData curves = UiHelper::getCurvesFromQVar(oldVal, &bValid);
    if (curves.empty() || !bValid)
        return true;

    bool bUpdate = false;
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
    UI_VECSTRING oldVec;
    if (oldVal.canConvert<UI_VECSTRING>())
    {
        oldVec = oldVal.value<UI_VECSTRING>();
    }
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, false);
    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline, false);
    int nFrame = timeline->value();
    for (int i = 0; i < datas.size(); i++) {
        auto key = curve_util::getCurveKey(i).toStdString();
        if (curves.contains(key))
        {
            QPointF pos(nFrame, datas.at(i));
            CURVE_DATA _curve = curve_util::toLegacyCurve(curves[key]);
            if (curve_util::updateCurve(pos, _curve)) {
                curves[key] = curve_util::fromLegacyCurve(_curve);
                bUpdate = true;
            }
        }
        else if (oldVec.size() > i && oldVec[i].toFloat() != datas[i])
        {
            bUpdate = true;
            oldVec[i] = QString::number(datas[i]);
            oldVal = QVariant::fromValue(oldVec);
        }
    }
    if (bUpdate)
    {
        const auto& anyVal = zeno::reflect::make_any<zeno::CurvesData>(curves);
        newValue = QVariant::fromValue(anyVal);
    }
    return bUpdate;
}

void AppHelper::dumpTabsToZsg(QDockWidget* dockWidget, RAPIDJSON_WRITER& writer)
{
    //not QDockWidget but ads::CDockWidget
    //TODO: refacror.
#if 0
    if (ZDockWidget* tabDockwidget = qobject_cast<ZDockWidget*>(dockWidget))
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
#endif
}