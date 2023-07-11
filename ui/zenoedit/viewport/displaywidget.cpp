#include "displaywidget.h"
#include "viewportwidget.h"
#include "optixviewport.h"
#include <zenovis/RenderEngine.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/Camera.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/CameraObject.h>
#include <zenomodel/include/uihelper.h>
#include "zenomainwindow.h"
#include "camerakeyframe.h"
#include <zenoui/style/zenostyle.h>
#include <zeno/core/Session.h>
#include "timeline/ztimeline.h"
#include "dialog/zrecorddlg.h"
#include "dialog/zrecprogressdlg.h"
#include "dialog/zrecframeselectdlg.h"
#include "util/apphelper.h"


using std::string;
using std::unordered_set;
using std::unordered_map;


DisplayWidget::DisplayWidget(bool bGLView, QWidget *parent)
    : QWidget(parent)
    , m_glView(nullptr)
    , m_pTimer(nullptr)
    , m_bRecordRun(false)
    , m_bGLView(bGLView)
    , m_optixView(nullptr)
{
    QVBoxLayout *pLayout = new QVBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    initRecordMgr();

    if (m_bGLView)
    {
        m_glView = new ViewportWidget;
        pLayout->addWidget(m_glView);
    }
    else
    {
        m_optixView = new ZOptixViewport;
        pLayout->addWidget(m_optixView);
        connect(this, &DisplayWidget::frameRunFinished, m_optixView, &ZOptixViewport::onFrameRunFinished);
    }

    setLayout(pLayout);

    m_camera_keyframe = new CameraKeyframeWidget;
    Zenovis *pZenovis = getZenoVis();
    if (pZenovis) {
        pZenovis->m_camera_keyframe = m_camera_keyframe;
    }
    //connect(m_view, SIGNAL(sig_Draw()), this, SLOT(onRun()));

    //it seems there is no need to use timer, because optix is seperated from GL and update by a thread.
    m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

DisplayWidget::~DisplayWidget()
{
}

void DisplayWidget::initRecordMgr()
{
    m_recordMgr.setParent(this);
    connect(&m_recordMgr, &RecordVideoMgr::frameFinished, this,
            [=](int frameid) { zeno::log_info("frame {} has been recorded", frameid); });
}

void DisplayWidget::testCleanUp()
{
    if (m_glView)
        m_glView->testCleanUp();
}

void DisplayWidget::init()
{
    //m_camera->installEventFilter(this);
}

Zenovis* DisplayWidget::getZenoVis() const
{
    if (m_bGLView)
    {
        ZASSERT_EXIT(m_glView, nullptr);
        return m_glView->getZenoVis();
    }
    else
    {
        ZASSERT_EXIT(m_optixView, nullptr);
        return m_optixView->getZenoVis();
    }
}

QSize DisplayWidget::sizeHint() const
{
    return ZenoStyle::dpiScaledSize(QSize(12, 400));
}

QSize DisplayWidget::viewportSize() const
{
    if (m_bGLView) {
        return m_glView->size();
    } else {
        return m_optixView->size();
    }
}

void DisplayWidget::resizeViewport(QSize sz)
{
    if (m_bGLView) {
        m_glView->resizeGL(sz.width(), sz.height());
        m_glView->updateGL();
    } else {
        //todo: for optix view
    }
}

std::shared_ptr<zeno::Picker> DisplayWidget::picker() const
{
    return m_bGLView ? m_glView->picker() : nullptr;
}

void DisplayWidget::updateCameraProp(float aperture, float disPlane)
{
    if (m_glView) {
        m_glView->updateCameraProp(aperture, disPlane);
    } else {
        m_optixView->updateCameraProp(aperture, disPlane);
    }
}

void DisplayWidget::updatePerspective()
{
    if (m_glView) {
        m_glView->updatePerspective();
    }
    else {
        m_optixView->updatePerspective();
    }
}

void DisplayWidget::setNumSamples(int samples)
{
    if (m_glView) {
        m_glView->setNumSamples(samples);
    }
    else {
        m_optixView->setNumSamples(samples);
    }
}

void DisplayWidget::setCameraRes(const QVector2D& res)
{
    if (m_glView) {
        m_glView->setCameraRes(res);
    }
    else {
        m_optixView->setCameraRes(res);
    }
}

void DisplayWidget::setSafeFrames(bool bLock, int nx, int ny)
{
    if (m_glView) {
        m_glView->setSafeFrames(bLock, nx, ny);
    }
    else {
        m_optixView->setSafeFrames(bLock, nx, ny);
    }
}

void DisplayWidget::setSimpleRenderOption()
{
    if (m_glView)
        m_glView->setSimpleRenderOption();
}

void DisplayWidget::setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly) {
    if (m_optixView)
    {
        m_optixView->setRenderSeparately(updateLightCameraOnly, updateMatlOnly);
    }
}

bool DisplayWidget::isCameraMoving() const
{
    if (m_glView)
        return m_glView->m_bMovingCamera;
    else
        return m_optixView->isCameraMoving();
}

bool DisplayWidget::isPlaying() const
{
    auto zenoVis = getZenoVis();
    ZASSERT_EXIT(zenoVis, false);
    return zenoVis->isPlaying();
}

bool DisplayWidget::isGLViewport() const
{
    return m_bGLView;
}

ZOptixViewport* DisplayWidget::optixViewport() const
{
    return m_optixView;
}

void DisplayWidget::killOptix()
{
    if (m_optixView)
        m_optixView->killThread();
}

void DisplayWidget::mouseReleaseEvent(QMouseEvent* event)
{
    ZenoMainWindow* main = zenoApp->getMainWindow();
    ZASSERT_EXIT(main);
    QVector<DisplayWidget*> views = main->viewports();
    for (auto view : views)
    {
        view->setIsCurrent(false);
    }
    setIsCurrent(true);
}

void DisplayWidget::setIsCurrent(bool isCurrent)
{
    bIsCurrent = isCurrent;
}

bool DisplayWidget::isCurrent()
{
    return bIsCurrent;
}

void DisplayWidget::onPlayClicked(bool bChecked)
{
    if (m_bGLView)
    {
        if (bChecked)
        {
            m_pTimer->start(m_sliderFeq);
        }
        else
        {
            m_pTimer->stop();
        }
        if (getZenoVis())
            getZenoVis()->startPlay(bChecked);
    }
    else
    {
        emit m_optixView->sig_togglePlayButton(bChecked);
    }
}

void DisplayWidget::updateFrame(const QString &action) // cihou optix
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    if (mainWin && mainWin->inDlgEventLoop())
        return;

    if (action == "newFrame")
    {
        m_pTimer->stop();
        //zeno::log_warn("stop");
        return;
    }
    else if (action == "finishFrame")
    {
        if (isPlaying())
        {
            //restore the timer, because it will be stopped by signal of new frame.
            m_pTimer->start(m_sliderFeq);
        }
        int frame = zeno::getSession().globalComm->maxPlayFrames() - 1;
        frame = std::max(frame, 0);
        emit frameRunFinished(frame);
    }
    else if (!action.isEmpty())
    {
        //unknown signal, stop it.
        m_pTimer->stop();
        return;
    }
    if (m_bGLView)
    {
        m_glView->update();
    }
    else
    {
        m_optixView->update();
    }
}

void DisplayWidget::onCommandDispatched(int actionType, bool bChecked)
{
    if (actionType == ZenoMainWindow::ACTION_SMOOTH_SHADING)
    {
        if (m_glView)
            m_glView->getSession()->set_smooth_shading(bChecked);
        updateFrame();
    }
    else if (actionType == ZenoMainWindow::ACTION_NORMAL_CHECK)
    {
        if (m_glView)
            m_glView->getSession()->set_normal_check(bChecked);
        updateFrame();
    }
    else if (actionType == ZenoMainWindow::ACTION_WIRE_FRAME)
    {
        if (m_glView)
            m_glView->getSession()->set_render_wireframe(bChecked);
        updateFrame();
    }
    else if (actionType == ZenoMainWindow::ACTION_SHOW_GRID)
    {
        if (m_glView)
            m_glView->getSession()->set_show_grid(bChecked);
        //todo: need a notify mechanism from zenovis/session.
        updateFrame();
    }
    else if (actionType == ZenoMainWindow::ACTION_BACKGROUND_COLOR)
    {
        if (m_glView)
        {
            auto [r, g, b] = m_glView->getSession()->get_background_color();
            auto c = QColor::fromRgbF(r, g, b);
            c = QColorDialog::getColor(c);
            if (c.isValid()) {
                m_glView->getSession()->set_background_color(c.redF(), c.greenF(), c.blueF());
                updateFrame();
            }
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_SOLID)
    {
        const char *e = "bate";
        if (m_glView)
            m_glView->getSession()->set_render_engine(e);
        updateFrame(QString::fromUtf8(e));
    }
    else if (actionType == ZenoMainWindow::ACTION_SHADING)
    {
        const char *e = "zhxx";
        if (m_glView)
            m_glView->getSession()->set_render_engine(e);
        //m_view->getSession()->set_enable_gi(false);
        updateFrame(QString::fromUtf8(e));
    }
    else if (actionType == ZenoMainWindow::ACTION_OPTIX)
    {
        const char *e = "optx";
        //now we have ZOptixWidget, and don't use gl as backend of optix anymore.
        //m_glView->getSession()->set_render_engine(e);
        //updateFrame(QString::fromUtf8(e));
    }
    else if (actionType == ZenoMainWindow::ACTION_NODE_CAMERA)
    {
        if (m_glView)
        {
            int frameid = m_glView->getSession()->get_curr_frameid();
            auto *scene = m_glView->getSession()->get_scene();
            for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
                if (key.find("MakeCamera") != std::string::npos &&
                    key.find(zeno::format(":{}:", frameid)) != std::string::npos) {
                    auto cam = dynamic_cast<zeno::CameraObject *>(ptr)->get();
                    scene->camera->setCamera(cam);
                    updateFrame();
                }
            }
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_RECORD_VIDEO)
    {
        onRecord();
    }
    else if (actionType == ZenoMainWindow::ACTION_SCREEN_SHOOT)
    {
        onScreenShoot();
    } 
    else if (actionType == ZenoMainWindow::ACTION_BLACK_WHITE || 
             actionType == ZenoMainWindow::ACTION_GREEK ||
             actionType == ZenoMainWindow::ACTION_DAY_LIGHT ||
             actionType == ZenoMainWindow::ACTION_DEFAULT ||
             actionType == ZenoMainWindow::ACTION_FOOTBALL_FIELD ||
             actionType == ZenoMainWindow::ACTION_FOREST ||
             actionType == ZenoMainWindow::ACTION_LAKE ||
             actionType == ZenoMainWindow::ACTION_SEA)
    {
        //todo: no implementation from master.
    }
}

void DisplayWidget::onRunFinished()
{
}

bool DisplayWidget::isOptxRendering() const
{
    return !m_bGLView;
}

void DisplayWidget::onSliderValueChanged(int frame)
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    mainWin->clearErrorMark();

    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    if (mainWin->isAlways())
    {
        auto pGraphsMgr = zenoApp->graphsManagment();
        IGraphsModel *pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        LAUNCH_PARAM launchParam;
        launchParam.beginFrame = frame;
        launchParam.endFrame = frame;
        AppHelper::initLaunchCacheParam(launchParam);
        launchProgram(pModel, launchParam);
    }
    else
    {
        if (m_bGLView)
        {
            Zenovis *pZenoVis = getZenoVis();
            ZASSERT_EXIT(pZenoVis);
            pZenoVis->setCurrentFrameId(frame);
            updateFrame();
            onPlayClicked(false);
        }
        else
        {
            ZASSERT_EXIT(m_optixView);
            emit m_optixView->sig_switchTimeFrame(frame);
        }
        BlockSignalScope scope(timeline);
        timeline->setPlayButtonChecked(false);
    }
    if (m_glView)
    {
        m_glView->clearTransformer();
    }
}

void DisplayWidget::changeTransformOperation(const QString& node)
{
    if (m_glView)
        m_glView->changeTransformOperation(node);
}

void DisplayWidget::changeTransformOperation(int mode)
{
    if (m_glView)
        m_glView->changeTransformOperation(mode);
}

void DisplayWidget::beforeRun()
{
    if (m_glView)
    {
        m_glView->clearTransformer();
    }
    Zenovis *pZenoVis = getZenoVis();
    ZASSERT_EXIT(pZenoVis);
    pZenoVis->getSession()->get_scene()->selected.clear();
}

void DisplayWidget::afterRun()
{
    if (m_glView)
        m_glView->updateLightOnce = true;

    Zenovis *pZenoVis = getZenoVis();
    ZASSERT_EXIT(pZenoVis);
    auto scene = pZenoVis->getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::onRun(LAUNCH_PARAM launchParam)
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);

    auto pGraphsMgr = zenoApp->graphsManagment();
    ZASSERT_EXIT(pGraphsMgr);

    IGraphsModel *pModel = pGraphsMgr->currentModel();
    ZASSERT_EXIT(pModel);

    mainWin->clearErrorMark();

    if (m_glView)
    {
        m_glView->clearTransformer();
        m_glView->getSession()->get_scene()->selected.clear();
    }
    launchProgram(pModel, launchParam);

    if (m_glView)
        m_glView->updateLightOnce = true;

    Zenovis* pZenoVis = getZenoVis();
    ZASSERT_EXIT(pZenoVis);
    auto scene = pZenoVis->getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::onRun() {
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    mainWin->clearErrorMark();

    if (m_glView)
    {
        m_glView->clearTransformer();
        m_glView->getSession()->get_scene()->selected.clear();
    }

    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    QPair<int, int> fromTo = timeline->fromTo();
    int beginFrame = fromTo.first;
    int endFrame = fromTo.second;
    if (endFrame >= beginFrame && beginFrame >= 0) {
        auto pGraphsMgr = zenoApp->graphsManagment();
        IGraphsModel *pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        LAUNCH_PARAM launchParam;
        launchParam.beginFrame = beginFrame;
        launchParam.endFrame = endFrame;
        AppHelper::initLaunchCacheParam(launchParam);
        launchProgram(pModel, launchParam);
    } else {
    }

    if (m_glView)
        m_glView->updateLightOnce = true;

    Zenovis* pZenoVis = getZenoVis();
    ZASSERT_EXIT(pZenoVis);
    auto scene = pZenoVis->getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::runAndRecord(const VideoRecInfo &recInfo) {
    //reset the record info first.
    m_bRecordRun = true;
    m_recordMgr.setRecordInfo(recInfo);

    Zenovis* pZenoVis = getZenoVis();
    ZASSERT_EXIT(pZenoVis);
    pZenoVis->startPlay(true);

    //and then play.
    onPlayClicked(true);

    //run first.
    onRun();

    if (recInfo.exitWhenRecordFinish) {
        connect(&m_recordMgr, &RecordVideoMgr::recordFinished, this, [=]() { zenoApp->quit(); });
    }
}

void DisplayWidget::onScreenShoot() {
    QString path = QFileDialog::getSaveFileName(
        nullptr, tr("Path to Save"), "",
        tr("PNG images(*.png);;JPEG images(*.jpg);;BMP images(*.bmp);;EXR images(*.exr);;HDR images(*.hdr);;"));
    QString ext = QFileInfo(path).suffix();
    if (ext.isEmpty()) {
        //qt bug: won't fill extension automatically.
        ext = "png";
        path.append(".png");
    }
    if (!path.isEmpty())
    {
        Zenovis* pZenoVis = getZenoVis();
        ZASSERT_EXIT(pZenoVis);
        pZenoVis->getSession()->do_screenshot(path.toStdString(), ext.toStdString());
    }
}

void DisplayWidget::onRecord()
{
    auto &pGlobalComm = zeno::getSession().globalComm;
    ZASSERT_EXIT(pGlobalComm);

    //based on timeline value directory.
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);

    ZRecordVideoDlg dlg(this);
    if (QDialog::Accepted == dlg.exec())
    {
        VideoRecInfo recInfo;
        if (!dlg.getInfo(recInfo))
        {
            QMessageBox::warning(nullptr, tr("Record"), tr("The output path is invalid, please choose another path."));
            return;
        }
        //validation.

        ZRecFrameSelectDlg frameDlg(this);
        int ret = frameDlg.exec();
        if (QDialog::Rejected == ret) {
            return;
        }

        bool bRunBeforeRecord = false;
        recInfo.frameRange = frameDlg.recordFrameRange(bRunBeforeRecord);

        if (bRunBeforeRecord)
        {
            //clear cached objs.
            zeno::getSession().globalComm->clearState();
            LAUNCH_PARAM launchParam;
            launchParam.beginFrame = recInfo.frameRange.first;
            launchParam.endFrame = recInfo.frameRange.second;
            launchParam.autoRmCurcache = recInfo.bAutoRemoveCache;
            AppHelper::initLaunchCacheParam(launchParam);
            onRun(launchParam);
        }

        //setup signals issues.
        m_recordMgr.setRecordInfo(recInfo);

        ZRecordProgressDlg dlgProc(recInfo, this);
        connect(&m_recordMgr, SIGNAL(frameFinished(int)), &dlgProc, SLOT(onFrameFinished(int)));
        connect(&m_recordMgr, SIGNAL(recordFinished(QString)), &dlgProc, SLOT(onRecordFinished(QString)));
        connect(&m_recordMgr, SIGNAL(recordFailed(QString)), &dlgProc, SLOT(onRecordFailed(QString)));
        connect(&dlgProc, SIGNAL(cancelTriggered()), &m_recordMgr, SLOT(cancelRecord()));
        connect(&dlgProc, &ZRecordProgressDlg::pauseTriggered, this, [=]() { mainWin->toggleTimelinePlay(false); });
        connect(&dlgProc, &ZRecordProgressDlg::continueTriggered, this, [=]() { mainWin->toggleTimelinePlay(true); });

        if (!m_bGLView)
        {
            ZASSERT_EXIT(m_optixView);
            m_optixView->recordVideo(recInfo);
        }
        else
        {
            moveToFrame(recInfo.frameRange.first);      // first, set the time frame start end.
            mainWin->toggleTimelinePlay(true);          // and then play.
            //the recording implementation is RecordVideoMgr::onFrameDrawn.
        }

        if (QDialog::Accepted == dlgProc.exec()) {

        } else {
            m_recordMgr.cancelRecord();
        }
    }
}

void DisplayWidget::onRecord_slient(const VideoRecInfo& recInfo)
{
    m_recordMgr.setRecordInfo(recInfo);

    if (!m_bGLView)
    {
        ZASSERT_EXIT(m_optixView);
        m_optixView->recordVideo(recInfo);
    }
    else
    {
        moveToFrame(recInfo.frameRange.first);      // first, set the time frame start end.
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);
        mainWin->toggleTimelinePlay(true);          // and then play.
        //the recording implementation is RecordVideoMgr::onFrameDrawn.
    }

    connect(&m_recordMgr, &RecordVideoMgr::recordFinished, this, [=](QString msg) {
        zeno::log_info("process exited with {} successfully", 0);
        QApplication::exit(0);
    });

    connect(&m_recordMgr, &RecordVideoMgr::recordFailed, this, [=](QString msg) {
        zeno::log_info("process exited with {} failed", -1);
        QApplication::exit(-1);
    });
}

void DisplayWidget::moveToFrame(int frame) {
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);

    Zenovis *pZenoVis = getZenoVis();
    ZASSERT_EXIT(pZenoVis);
    pZenoVis->setCurrentFrameId(frame);
    updateFrame();
    onPlayClicked(false);
    {
        BlockSignalScope scope(timeline);
        timeline->setPlayButtonChecked(false);
        timeline->setSliderValue(frame);
    }
}

void DisplayWidget::onKill() {
    killProgram();
}

void DisplayWidget::onNodeSelected(const QModelIndex &subgIdx, const QModelIndexList &nodes, bool select) {
    // tmp code for Primitive Filter Node interaction
    if (nodes.size() > 1 || !m_bGLView)
        return;

    ZASSERT_EXIT(m_glView);
    auto node_id = nodes[0].data(ROLE_OBJNAME).toString();
    if (node_id == "PrimitiveAttrPicker") {
        auto scene = m_glView->getSession()->get_scene();
        ZASSERT_EXIT(scene);
        auto picker = m_glView->picker();
        ZASSERT_EXIT(picker);
        if (select) {
            // check input nodes
            auto input_nodes = zeno::NodeSyncMgr::GetInstance().getInputNodes(nodes[0], "prim");
            if (input_nodes.size() != 1)
                return;
            // find prim in object manager
            auto input_node_id = input_nodes[0].get_node_id();
            string prim_name;
            for (const auto &[k, v] : scene->objectsMan->pairsShared()) {
                if (k.find(input_node_id.toStdString()) != string::npos)
                    prim_name = k;
            }
            if (prim_name.empty())
                return;

            zeno::NodeLocation node_location(nodes[0], subgIdx);
            // set callback to picker
            auto callback = [node_location,
                             prim_name](unordered_map<string, unordered_set<int>> &picked_elems) -> void {
                std::string picked_elems_str;
                auto &picked_prim_elems = picked_elems[prim_name];
                for (auto elem : picked_prim_elems)
                    picked_elems_str += std::to_string(elem) + ",";
                zeno::NodeSyncMgr::GetInstance().updateNodeParamString(node_location, "selected", picked_elems_str);
            };
            if (picker) {
                picker->set_picked_elems_callback(callback);
                // ----- enter node context
                picker->save_context();
            }
            // read selected mode
            auto select_mode_str = zeno::NodeSyncMgr::GetInstance().getInputValString(nodes[0], "mode");
            if (select_mode_str == "triangle")
                scene->select_mode = zenovis::PICK_MESH;
            else if (select_mode_str == "line")
                scene->select_mode = zenovis::PICK_LINE;
            else
                scene->select_mode = zenovis::PICK_VERTEX;
            // read selected elements
            string node_context;
            auto node_selected_str = zeno::NodeSyncMgr::GetInstance().getParamValString(nodes[0], "selected");
            if (!node_selected_str.empty()) {
                auto node_selected_qstr = QString(node_selected_str.c_str());
                auto elements = node_selected_qstr.split(',');
                for (auto &e : elements)
                    if (e.size() > 0)
                        node_context += prim_name + ":" + e.toStdString() + " ";

                if (picker)
                    picker->load_from_str(node_context, scene->select_mode);
            }
            if (picker) {
                picker->sync_to_scene();
                picker->focus(prim_name);
            }
        } else {
            if (picker) {
                picker->load_context();
                picker->sync_to_scene();
                picker->focus("");
                picker->set_picked_elems_callback({});
            }
        }
        zenoApp->getMainWindow()->updateViewport();
    }
    if (node_id == "MakePrimitive") {
        auto picker = m_glView->picker();
        ZASSERT_EXIT(picker);
        if (select) {
            picker->switch_draw_mode();
            zeno::NodeLocation node_location(nodes[0], subgIdx);
            auto pick_callback = [nodes, node_location, this](float depth, int x, int y) {
                Zenovis *pZenovis = m_glView->getZenoVis();
                ZASSERT_EXIT(pZenovis && pZenovis->getSession());
                auto scene = pZenovis->getSession()->get_scene();
                auto _near = scene->camera->m_near;
                auto _far = scene->camera->m_far;
                auto fov = scene->camera->m_fov;
                auto cz = glm::length(scene->camera->m_lodcenter);
                if (depth != 0) {
//                    depth = depth * 2 - 1;
//                    cz = 2 * _near * _far / ((_far + _near) - depth * (_far - _near));
                    glm::vec4 ndc = {0, 0, depth, 1};
                    glm::vec4 clip_c = glm::inverse(scene->camera->m_proj) * ndc;
                    clip_c /= clip_c.w;
                    cz = -clip_c.z;
                }
                auto w = scene->camera->m_nx;
                auto h = scene->camera->m_ny;
                // zeno::log_info("fov: {}", fov);
                // zeno::log_info("w: {}, h: {}", w, h);
                auto u = (2.0 * x / w) - 1;
                auto v = 1 - (2.0 * y / h);
                // zeno::log_info("u: {}, v: {}", u, v);
                auto cy = v * tan(glm::radians(fov) / 2) * cz;
                auto cx = u * tan(glm::radians(fov) / 2) * w / h * cz;
                // zeno::log_info("cx: {}, cy: {}, cz: {}", cx, cy, -cz);
                glm::vec4 cc = {cx, cy, -cz, 1};
                auto wc = glm::inverse(scene->camera->m_view) * cc;
                wc /= wc.w;
                // zeno::log_info("wx: {}, wy: {}, wz: {}", word_coord.x, word_coord.y, word_coord.z);
                auto points = zeno::NodeSyncMgr::GetInstance().getInputValString(nodes[0], "points");
                zeno::log_info("fetch {}", points.c_str());
                points += std::to_string(wc.x) + " " + std::to_string(wc.y) + " " + std::to_string(wc.z) + " ";
                zeno::NodeSyncMgr::GetInstance().updateNodeInputString(node_location, "points", points);
            };
            picker->set_picked_depth_callback(pick_callback);
        } else {
            picker->switch_draw_mode();
        }
    }
}