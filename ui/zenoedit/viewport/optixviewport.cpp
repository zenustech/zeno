#include "optixviewport.h"
#include "zenovis.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "cameracontrol.h"
#include <zenovis/DrawOptions.h>
#include <zeno/extra/GlobalComm.h>
#include "settings/zenosettingsmanager.h"
#include "launch/corelaunch.h"
#include <zeno/core/Session.h>
#include <zenovis/Camera.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "viewport/displaywidget.h"
#include <zenovis/RenderEngine.h>
#include <zenomodel/include/api.h>
#include <zenomodel/include/nodesmgr.h>
#include <zeno/utils/string.h>

#include "viewport/zoptixviewport.h"
#include "nodesview/zenographseditor.h"
#include "nodesys/zenosubgraphscene.h"
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>


#include "tinygltf/json.hpp"
using Json = nlohmann::json;

static void add_set_node_xform(
        ZENO_HANDLE hGraph
        , std::optional<std::string> &cur_output_uuid
        , std::string const &outline_node_name
        , Json const &mat
        , std::unordered_map<std::string, std::string> &outline_node_to_uuid
) {
//    zeno::log_info("outline_node_name: {}", outline_node_name);
    if (mat["Mode"] == "Set") {
        if (outline_node_to_uuid.count(outline_node_name) == 0) {
            auto node = Zeno_GetNode(hGraph, cur_output_uuid.value());
            if (!node.has_value()) {
                return;
            }

            auto new_node_handle = Zeno_AddNode(hGraph, "SetNodeXform");
//            Zeno_SetView(hGraph, new_node_handle, true);
//            auto& node_sync = zeno::NodeSyncMgr::GetInstance();
//            auto node_loc = node_sync.searchNode(cur_output_uuid.value());
//            if (node_loc.has_value()) {
//                node_sync.updateNodeVisibility(node_loc.value());
//                zeno::log_info("node_loc.has_value");
//            }
            auto pos = Zeno_GetPos(hGraph, node.value());
            if (pos.has_value()) {
                zeno::vec2f npos = pos.value();
                npos += zeno::vec2f(500, 0);
                Zeno_SetPos(hGraph, new_node_handle, {npos[0], npos[1]});
            }
            std::string node_uuid;
            auto err = Zeno_GetNodeUuid(hGraph, new_node_handle, node_uuid);
            if (err != 0) {
                return;
            }
            outline_node_to_uuid[outline_node_name] = node_uuid;
            cur_output_uuid = node_uuid;
            err = Zeno_SetInputDefl(hGraph, new_node_handle, "node", outline_node_name);
            if (err != 0) {
                return;
            }
            Zeno_AddLink(hGraph, node.value(), "scene", new_node_handle, "scene");
//            Zeno_SetView(hGraph, new_node_handle, true);
            Zeno_SetView(hGraph, node.value(), false);
        }
        auto output_uuid = outline_node_to_uuid[outline_node_name];
        auto node = Zeno_GetNode(hGraph, output_uuid);
        if (!node.has_value()) {
            return;
        }
        auto const &r0 = mat["r0"];
        auto const &r1 = mat["r1"];
        auto const &r2 = mat["r2"];
        auto const &t  = mat["t"];
        Zeno_SetInputDefl(hGraph, node.value(), "r0", zeno::vec3f(r0[0], r0[1], r0[2]));
        Zeno_SetInputDefl(hGraph, node.value(), "r1", zeno::vec3f(r1[0], r1[1], r1[2]));
        Zeno_SetInputDefl(hGraph, node.value(), "r2", zeno::vec3f(r2[0], r2[1], r2[2]));
        Zeno_SetInputDefl(hGraph, node.value(),  "t", zeno::vec3f( t[0],  t[1],  t[2]));
    }
}

OptixWorker::OptixWorker(Zenovis *pzenoVis)
    : QObject(nullptr)
    , m_zenoVis(pzenoVis)
    , m_bRecording(false)
    , m_pTimer(nullptr)
{
    m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));

    ZASSERT_EXIT(m_zenoVis);
    auto session = m_zenoVis->getSession();
    ZASSERT_EXIT(session);
    auto scene = session->get_scene();
    ZASSERT_EXIT(scene);
    auto engin = scene->renderMan->getEngine("optx");
    engin->fun = [this](std::string content) {
        Json json = Json::parse(content);
        if (json["MessageType"] == "SetNodeXform") {
            emit sig_sendToXformPanel(QString::fromStdString(content));
//            ZENO_HANDLE hGraph = Zeno_GetGraph("main");
//            auto outline_node_name = std::string(json["NodeName"]);
//            if (!this->cur_node_uuid.has_value()) {
//                auto node_key = std::string(json["NodeKey"]);
//                cur_node_uuid = zeno::split_str(node_key, ':')[0];
//            }
//            if (this->cur_node_uuid.has_value()) {
//                add_set_node_xform(hGraph, this->cur_node_uuid, outline_node_name, json, this->outline_node_to_uuid);
//            }
        }
        else if (json["MessageType"] == "XformPanelInitFeedback") {
            emit sig_sendToXformPanel(QString::fromStdString(content));
        }
        else if (json["MessageType"] == "SetSceneXform") {
			emit sig_sendToXformPanel(QString::fromStdString(content));
        }
        else if (json["MessageType"] == "CleanupAssets") {
            emit sig_sendToOptixViewport(QString::fromStdString(content));
            emit sig_sendToOutline(QString::fromStdString(content));
        }
        else if (json["MessageType"] == "SetGizmoAxis") {
            emit sig_sendToOptixViewport(QString::fromStdString(content));
        }
        else {
            emit sig_sendToOutline(QString::fromStdString(content));
        }
    };

}

OptixWorker::~OptixWorker()
{
}

OptixWorker::OptixWorker(QObject* parent)
    : QObject(parent)
    , m_zenoVis(nullptr)
    , m_pTimer(nullptr)
    , m_bRecording(false)
{
    //used by offline worker.
    m_pTimer = new QTimer(this);
    m_zenoVis = new Zenovis(this);

    //fake GL
    m_zenoVis->initializeGL();
    m_zenoVis->setCurrentFrameId(0);    //correct frame automatically.

    m_zenoVis->m_camera_control = new CameraControl(m_zenoVis, nullptr, nullptr, this);
    m_zenoVis->getSession()->set_render_engine("optx");
}

void OptixWorker::updateFrame()
{
    //avoid conflict.
    if (m_bRecording)
        return;

    m_zenoVis->paintGL();
    int w = 0, h = 0;
    void *data = m_zenoVis->getSession()->get_scene()->getOptixImg(w, h);
    int scale = zeno::getSession().userData().has("optix_image_path")?1:m_zenoVis->getSession()->get_scene()->camera->zOptixCameraSettingInfo.renderRatio;
    int scale2 = m_zenoVis->getSession()->get_scene()->drawOptions->simpleRender?scale:1;
    if(scale2 == 1 && (w != m_zenoVis->getSession()->get_scene()->camera->m_nx || h!= m_zenoVis->getSession()->get_scene()->camera->m_ny) )
        scale = scale;
    else
        scale = scale2;
    //m_renderImg = QImage((uchar *)data, w, h, QImage::Format_RGBA8888);
    std::vector<int32_t> img;
    img.resize(w*h*scale*scale);
    for(int j=0;j<h*scale;j++)
        for(int i=0;i<w*scale;i++)
        {
            int jj = j/scale;
            int ii = i/scale;
            if(ii<w && jj<h)
            {
                img[j*w*scale + i]  = ((int32_t *)data)[jj*w + ii];
            }
        }
    m_renderImg = QImage((uchar *)img.data(), w*scale, h*scale, QImage::Format_RGBA8888);
    m_renderImg = m_renderImg.mirrored(false, true);

    emit renderIterate(m_renderImg);
}

void OptixWorker::onPlayToggled(bool bToggled)
{
    m_zenoVis->startPlay(bToggled);
    if (bToggled) {
        m_pTimer->start(m_slidFeq);
    }
    else {
        m_pTimer->start(m_sampleFeq);
    }
    //setRenderSeparately(RunALL);
}

void OptixWorker::onSetSlidFeq(int feq)
{
    m_slidFeq = feq;
}

void OptixWorker::onModifyLightData(UI_VECTYPE posvec, UI_VECTYPE scalevec, UI_VECTYPE rotatevec, UI_VECTYPE colorvec, float intensity, QString nodename, UI_VECTYPE skipParam)
{
    std::string name = nodename.toStdString();
    zeno::vec3f pos = zeno::vec3f(posvec[0], posvec[1], posvec[2]);
    zeno::vec3f scale = zeno::vec3f(scalevec[0], scalevec[1], scalevec[2]);
    zeno::vec3f rotate = zeno::vec3f(rotatevec[0], rotatevec[1], rotatevec[2]);
    zeno::vec3f color = zeno::vec3f(colorvec[0], colorvec[1], colorvec[2]);
    auto verts = ZenoLights::computeLightPrim(pos, rotate, scale);

    auto scene = m_zenoVis->getSession()->get_scene();
    ZASSERT_EXIT(scene);

    std::shared_ptr<zeno::IObject> obj;
    for (auto const& [key, ptr] : scene->objectsMan->lightObjects) {
        if (key.find(name) != std::string::npos) {
            obj = ptr;
            name = key;
        }
    }
    auto prim_in = dynamic_cast<zeno::PrimitiveObject*>(obj.get());

    if (prim_in) {
        auto& prim_verts = prim_in->verts;
        prim_verts[0] = verts[0];
        prim_verts[1] = verts[1];
        prim_verts[2] = verts[2];
        prim_verts[3] = verts[3];

        if (skipParam[0])
            pos = prim_in->userData().get2<zeno::vec3f>("pos");
        if (skipParam[1])
            scale = prim_in->userData().get2<zeno::vec3f>("scale");
        if (skipParam[2])
            rotate = prim_in->userData().get2<zeno::vec3f>("rotate");
        if (skipParam[3])
            color = prim_in->userData().get2<zeno::vec3f>("color");
        if (skipParam[4])
            intensity = prim_in->userData().get2<float>("intensity");

        prim_in->verts.attr<zeno::vec3f>("clr")[0] = color * intensity;

        prim_in->userData().setLiterial<zeno::vec3f>("pos", std::move(pos));
        prim_in->userData().setLiterial<zeno::vec3f>("scale", std::move(scale));
        prim_in->userData().setLiterial<zeno::vec3f>("rotate", std::move(rotate));
        if (prim_in->userData().has("intensity")) {
            prim_in->userData().setLiterial<zeno::vec3f>("color", std::move(color));
            prim_in->userData().setLiterial<float>("intensity", std::move(intensity));
        }

        scene->objectsMan->needUpdateLight = true;
        //pDisplay->setSimpleRenderOption();
    }
    else {
        zeno::log_info("modifyLightData not found {}", name);
    }
}

void OptixWorker::onUpdateCameraProp(float aperture, float disPlane, UI_VECTYPE skipParam)
{
    if (skipParam.size() == 0 || !skipParam[0])
        m_zenoVis->m_camera_control->setAperture(aperture);
    if (skipParam.size() == 0 || !skipParam[1])
        m_zenoVis->m_camera_control->setDisPlane(disPlane);
    m_zenoVis->m_camera_control->updatePerspective();
}

void OptixWorker::onFrameSwitched(int frame)
{
    //ui switch.
    m_zenoVis->setCurrentFrameId(frame);
    m_zenoVis->startPlay(false);
    m_pTimer->start(m_sampleFeq);
}

void OptixWorker::cancelRecording()
{
    m_bRecording = false;
}

void OptixWorker::setRenderSeparately(int runtype) {
    auto scene = m_zenoVis->getSession()->get_scene();
    scene->drawOptions->updateLightCameraOnly = (runType)runtype == RunLightCamera;
    scene->drawOptions->updateMatlOnly = (runType)runtype == RunMaterial;
    scene->drawOptions->updateMatrixOnly = (runType)runtype == RunMatrix;
}

void OptixWorker::onSetSafeFrames(bool bLock, int nx, int ny) {
    auto scene = m_zenoVis->getSession()->get_scene();
    scene->camera->set_safe_frames(bLock, nx, ny);
}

void OptixWorker::recordVideo(VideoRecInfo recInfo)
{
    //for the case about recording after run.
    zeno::scope_exit sp([=] {
        m_bRecording = false;
        m_pTimer->start(m_sampleFeq);
    });

    m_bRecording = true;
    m_pTimer->stop();

    for (int frame = recInfo.frameRange.first; frame <= recInfo.frameRange.second;)
    {
        if (!m_bRecording)
        {
            emit sig_recordCanceled();
            return;
        }
#ifdef ZENO_OPTIX_PROC
        QString cachePath = QString::fromStdString(zeno::getSession().globalComm->cachePath());
        QString frameDir = cachePath + "/" + QString::number(1000000 + frame).right(6);
        if (!QDir(frameDir).exists())
        {
            QThread::sleep(0);
            continue;
        }
        QString sLockFile = QString("%1/%2%3.lock").arg(cachePath).arg(zeno::iotags::sZencache_lockfile_prefix).arg(frame);
        QLockFile lckFile(sLockFile);
        bool ret = lckFile.tryLock();
        if (!ret)
        {
            QThread::sleep(0);
            continue;
        }
        lckFile.unlock();
#endif
        bool bSucceed = recordFrame_impl(recInfo, frame);
        if (bSucceed)
        {
            frame++;
        }
        else
        {
            QThread::sleep(0);
        }
    }
    emit sig_recordFinished();
}

void OptixWorker::screenShoot(QString path, QString type, int resx, int resy)
{
    bool exr = zeno::getSession().userData().has("output_exr") ? zeno::getSession().userData().get2<bool>("output_exr") : false;
    zeno::scope_exit sp([=]() {
        zeno::getSession().userData().set2("output_exr", exr);
        });
    zeno::getSession().userData().set2("output_exr", path.right(4) == ".exr" ? true : false);
    auto [x, y] = m_zenoVis->getSession()->get_window_size();
    if (!m_zenoVis->getSession()->is_lock_window())
        resx = x, resy = y;
    m_zenoVis->getSession()->set_window_size(resx, resy);
    m_zenoVis->getSession()->do_screenshot(path.toStdString(), type.toStdString(), true);
    m_zenoVis->getSession()->set_window_size(x, y);
}

bool OptixWorker::recordFrame_impl(VideoRecInfo recInfo, int frame)
{
    auto record_file = zeno::format("{}/P/{:07d}.jpg", recInfo.record_path.toStdString(), frame);
    auto extname = QFileInfo(QString::fromStdString(record_file)).suffix().toStdString();

    auto scene = m_zenoVis->getSession()->get_scene();
    auto old_num_samples = scene->drawOptions->num_samples;
    scene->drawOptions->num_samples = recInfo.numOptix;
    scene->drawOptions->denoise = recInfo.needDenoise;

    zeno::scope_exit sp([=]() {scene->drawOptions->num_samples = old_num_samples;});
    //it seems that msaa is used by opengl, but opengl has been removed from optix.
    scene->drawOptions->msaa_samples = recInfo.numMSAA;

    auto [x, y] = m_zenoVis->getSession()->get_window_size();

    auto &globalComm = zeno::getSession().globalComm;
    int numOfFrames = globalComm->numOfFinishedFrame();
    if (numOfFrames == 0)
        return false;

    std::pair<int, int> frameRg = globalComm->frameRange();
    int beginFrame = frameRg.first;
    int endFrame = frameRg.first + numOfFrames - 1;
    if (frame < beginFrame || frame > endFrame)
        return false;

    if (globalComm->isFrameBroken(frame))
    {
        /*
        QImage img(QSize((int)recInfo.res.x(), (int)recInfo.res.y()), QImage::Format_RGBA8888);
        img.fill(Qt::black);
        QPainter painter(&img);
        painter.setPen(Qt::white);
        QFont fnt = zenoApp->font();
        fnt.setPointSize(16);
        painter.setFont(fnt);
        painter.drawText(img.rect(), Qt::AlignCenter, QString(tr("the zencache of this frame has been removed")));
        img.save(QString::fromStdString(record_file), "JPG");
        */
        zeno::log_warn("The zencache of frame {} has been removed.", frame);
        return true;
    }

    int actualFrame = m_zenoVis->setCurrentFrameId(frame);
    m_zenoVis->doFrameUpdate();
    if (recInfo.bAutoRemoveCache)
        zeno::getSession().globalComm->removeCache(frame);

    m_zenoVis->getSession()->set_window_size((int)recInfo.res.x(), (int)recInfo.res.y());
    m_zenoVis->getSession()->do_screenshot(record_file, extname, true);
    m_zenoVis->getSession()->set_window_size(x, y);

    //todo: emit some signal to main thread(ui)
    emit sig_frameRecordFinished(frame);

    if (1) {
        //update ui.
        int w = 0, h = 0;
        void *data = m_zenoVis->getSession()->get_scene()->getOptixImg(w, h);
        m_renderImg = QImage((uchar *)data, w, h, QImage::Format_RGBA8888);
        m_renderImg = m_renderImg.mirrored(false, true);
        emit renderIterate(m_renderImg);
    }
    return true;
}

void OptixWorker::onSetLoopPlaying(bool enbale)
{
    m_zenoVis->setLoopPlaying(enbale);
}

void OptixWorker::stop()
{
    m_pTimer->stop();
}

void OptixWorker::work()
{
    m_pTimer->start(m_sampleFeq);
}

QImage OptixWorker::renderImage() const
{
    return m_renderImg;
}

void OptixWorker::needUpdateCamera()
{
    //todo: update reason.
    //m_zenoVis->getSession()->get_scene()->drawOptions->needUpdateGeo = false;	//just for teset.
    m_zenoVis->getSession()->get_scene()->drawOptions->needRefresh = true;
    m_pTimer->start(m_sampleFeq);
}

void OptixWorker::onCleanUpScene()
{
    m_zenoVis->cleanUpScene();
}

void OptixWorker::onCleanUpView()
{
    m_zenoVis->cleanupView();
}

void OptixWorker::onSetBackground(bool bShowBg)
{
    auto& ud = zeno::getSession().userData();
    ud.set2("optix_show_background", bShowBg);

    ZASSERT_EXIT(m_zenoVis);
    auto session = m_zenoVis->getSession();
    ZASSERT_EXIT(session);
    auto scene = session->get_scene();
    ZASSERT_EXIT(scene);
    //scene->objectsMan->needUpdateLight = true;
    //scene->drawOptions->simpleRender = true;

    if (auto engine = scene->renderMan->getEngine("optx")) {
		engine->showBackground(bShowBg);
	}

    //updateFrame();
}

void OptixWorker::onSetSampleNumber(int sample_number) {
    ZASSERT_EXIT(m_zenoVis);
    auto session = m_zenoVis->getSession();
    ZASSERT_EXIT(session);
    auto scene = session->get_scene();
    ZASSERT_EXIT(scene);
    scene->drawOptions->num_samples = sample_number;
    updateFrame();
}

void OptixWorker::onSendOptixMessage(QString msg_str) {
    ZASSERT_EXIT(m_zenoVis);
    auto session = m_zenoVis->getSession();
    ZASSERT_EXIT(session);
    auto scene = session->get_scene();
    ZASSERT_EXIT(scene);
    if(auto engine = scene->renderMan->getEngine("optx")) {
        std::string msg_std_str = msg_str.toStdString();
        Json msg = Json::parse(msg_std_str);
        if (msg["MessageType"] == "Xform") {
            auto res = m_zenoVis->m_camera_control->res();
            msg["Resolution"] = {res.x(), res.y()};
        }
        engine->outlineInit(msg);
    }
}

void OptixWorker::onSetData(
    float aperture,
    float shutter_speed,
    float iso,
    int renderRatio,
    bool aces,
    bool exposure,
    bool panorama_camera,
    bool panorama_vr180,
    float pupillary_distance
) {
//    zeno::log_info("I am in optix thread, now I want to set value {}", iso);
    auto scene = m_zenoVis->getSession()->get_scene();
    scene->camera->zOptixCameraSettingInfo.aperture = aperture;
    scene->camera->zOptixCameraSettingInfo.shutter_speed = shutter_speed;
    scene->camera->zOptixCameraSettingInfo.iso = iso;
    scene->camera->zOptixCameraSettingInfo.renderRatio = renderRatio;
    scene->camera->zOptixCameraSettingInfo.aces = aces;
    scene->camera->zOptixCameraSettingInfo.exposure = exposure;
    scene->camera->zOptixCameraSettingInfo.panorama_camera = panorama_camera;
    scene->camera->zOptixCameraSettingInfo.panorama_vr180 = panorama_vr180;
    scene->camera->zOptixCameraSettingInfo.pupillary_distance = pupillary_distance;
    scene->drawOptions->needRefresh = true;
}

ZOptixViewport::ZOptixViewport(QWidget* parent)
    : QWidget(parent)
    , m_zenovis(nullptr)
    , m_camera(nullptr)
    , updateLightOnce(false)
    , m_bMovingCamera(false)
    , m_pauseRenderDally(new QTimer)
{
    setMouseTracking(true);
    m_zenovis = new Zenovis(this);

    setFocusPolicy(Qt::ClickFocus);

    connect(m_zenovis, &Zenovis::objectsUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        //if (mainWin)
        //    emit mainWin->visObjectsUpdated(this, frameid);
    });

    connect(m_zenovis, &Zenovis::frameUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        if (mainWin) {
            bool hasglViewport = false;
            for (auto view: mainWin->viewports()) {
                if (view->isGLViewport() && view->isVisible()) {
                    hasglViewport = true;
                    break;
                }
            }
            if (!hasglViewport) {//有visible的gl窗口，则不更新timeline
                emit mainWin->visFrameUpdated(false, frameid);
            }
        }
    }, Qt::BlockingQueuedConnection);
    
    //初始化timeline置为起始帧
    auto mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    mainWin->onSetTimelineValue();

    //fake GL
    m_zenovis->initializeGL();
    m_zenovis->setCurrentFrameId(0);    //correct frame automatically.

    m_camera = new CameraControl(m_zenovis, nullptr, nullptr, this);
    m_zenovis->m_camera_control = m_camera;

    const char *e = "optx";
    m_zenovis->getSession()->set_render_engine(e);

    auto scene = m_zenovis->getSession()->get_scene();

    m_worker = new OptixWorker(m_zenovis);
    m_worker->moveToThread(&m_thdOptix);
    connect(&m_thdOptix, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(&m_thdOptix, &QThread::started, m_worker, &OptixWorker::work);
    connect(m_worker, &OptixWorker::renderIterate, this, [=](QImage img) {
        m_renderImage = img;
        drawAxis(m_renderImage);
        update();
    });
    connect(m_worker, &OptixWorker::sig_sendToOptixViewport, this, [=](QString const &content) {
        Json message = Json::parse(content.toStdString());
        if (message["MessageType"] == "SetGizmoAxis") {
            glm::mat4 mat = glm::mat4(1);
            auto const &r0 = message["r0"];
            auto const &r1 = message["r1"];
            auto const &r2 = message["r2"];
            auto const &t  = message["t"];
            mat[0] = {r0[0], r0[1], r0[2], 0.0f};
            mat[1] = {r1[0], r1[1], r1[2], 0.0f};
            mat[2] = {r2[0], r2[1], r2[2], 0.0f};
            mat[3] = { t[0],  t[1],  t[2], 1.0f};
            this->axis_coord = mat;
        }
        else if (message["MessageType"] == "CleanupAssets") {
            this->mode = "";
            this->axis = "";
            this->try_axis = "";
            this->local_space = true;
            this->axis_coord = std::nullopt;
        }
    });

    connect(m_pauseRenderDally, &QTimer::timeout, [&](){
//        zeno::log_info("time out\n");
        auto scene = m_zenovis->getSession()->get_scene();
        scene->drawOptions->simpleRender = false;
        scene->drawOptions->needRefresh = true;
        m_pauseRenderDally->stop();
        //std::cout << "SR: SimpleRender false, Active " << m_pauseRenderDally->isActive() << "\n";
    });

    connect(this, &ZOptixViewport::cameraAboutToRefresh, m_worker, &OptixWorker::needUpdateCamera);
    connect(this, &ZOptixViewport::stopRenderOptix, m_worker, &OptixWorker::stop, Qt::BlockingQueuedConnection);
    connect(this, &ZOptixViewport::resumeWork, m_worker, &OptixWorker::work, Qt::BlockingQueuedConnection);
    connect(this, &ZOptixViewport::sigRecordVideo, m_worker, &OptixWorker::recordVideo, Qt::QueuedConnection);
    connect(this, &ZOptixViewport::sigscreenshoot, m_worker, &OptixWorker::screenShoot, Qt::QueuedConnection);
    connect(this, &ZOptixViewport::sig_setSafeFrames, m_worker, &OptixWorker::onSetSafeFrames);

    connect(m_worker, &OptixWorker::sig_recordFinished, this, &ZOptixViewport::sig_recordFinished);
    connect(m_worker, &OptixWorker::sig_frameRecordFinished, this, &ZOptixViewport::sig_frameRecordFinished);
    connect(m_worker, &OptixWorker::sig_sendToOutline, this, &ZOptixViewport::sig_viewportSendToOutline);
    connect(m_worker, &OptixWorker::sig_sendToNodeEditor, this, &ZOptixViewport::sig_viewportSendToOutline);
    connect(m_worker, &OptixWorker::sig_sendToXformPanel, this, &ZOptixViewport::sig_viewportSendToXformPanel);

    connect(this, &ZOptixViewport::sig_switchTimeFrame, m_worker, &OptixWorker::onFrameSwitched);
    connect(this, &ZOptixViewport::sig_togglePlayButton, m_worker, &OptixWorker::onPlayToggled);
    connect(this, &ZOptixViewport::sig_setRunType, m_worker, &OptixWorker::setRenderSeparately);
    connect(this, &ZOptixViewport::sig_setLoopPlaying, m_worker, &OptixWorker::onSetLoopPlaying);
    connect(this, &ZOptixViewport::sig_setSlidFeq, m_worker, &OptixWorker::onSetSlidFeq);
    connect(this, &ZOptixViewport::sig_modifyLightData, m_worker, &OptixWorker::onModifyLightData);
    connect(this, &ZOptixViewport::sig_updateCameraProp, m_worker, &OptixWorker::onUpdateCameraProp);
    connect(this, &ZOptixViewport::sig_cleanUpScene, m_worker, &OptixWorker::onCleanUpScene);
    connect(this, &ZOptixViewport::sig_cleanUpView, m_worker, &OptixWorker::onCleanUpView);
    connect(this, &ZOptixViewport::sig_setBackground, m_worker, &OptixWorker::onSetBackground);
    connect(this, &ZOptixViewport::sig_setSampleNumber, m_worker, &OptixWorker::onSetSampleNumber);
    connect(this, &ZOptixViewport::sig_setdata_on_optix_thread, m_worker, &OptixWorker::onSetData);

    connect(this, &ZOptixViewport::sig_sendOptixMessage, m_worker, &OptixWorker::onSendOptixMessage, Qt::QueuedConnection);

    setRenderSeparately(RunALL);
    m_thdOptix.start();
}

ZOptixViewport::~ZOptixViewport()
{
    m_thdOptix.quit();
    m_thdOptix.wait();
}

zenovis::ZOptixCameraSettingInfo ZOptixViewport::getdata_from_optix_thread()
{
    auto scene = m_zenovis->getSession()->get_scene();
    return scene->camera->zOptixCameraSettingInfo;
}

void ZOptixViewport::setdata_on_optix_thread(zenovis::ZOptixCameraSettingInfo value)
{
//    zeno::log_info("setdata_on_optix_thread {}", value.iso);
    emit sig_setdata_on_optix_thread(
            value.aperture,
            value.shutter_speed,
            value.iso,
            value.renderRatio,
            value.aces,
            value.exposure,
            value.panorama_camera,
            value.panorama_vr180,
            value.pupillary_distance
    );
}

void ZOptixViewport::setSimpleRenderOption()
{
    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
    m_pauseRenderDally->stop();
    m_pauseRenderDally->start(3*1000);  // Second to millisecond
}

void ZOptixViewport::setRenderSeparately(runType runtype) {
    emit sig_setRunType((int)runtype);
}

void ZOptixViewport::cameraLookTo(zenovis::CameraLookToDir dir)
{
    m_camera->lookTo(dir);
}

Zenovis* ZOptixViewport::getZenoVis() const
{
    return m_zenovis;
}

bool ZOptixViewport::isCameraMoving() const
{
    return m_bMovingCamera;
}

void ZOptixViewport::updateCamera()
{
    emit cameraAboutToRefresh();
}

void ZOptixViewport::killThread()
{
    stopRender();
    m_thdOptix.quit();
    m_thdOptix.wait();
}

void ZOptixViewport::setSlidFeq(int feq)
{
    emit sig_setSlidFeq(feq);
}

void ZOptixViewport::cleanUpScene()
{
    emit sig_cleanUpScene();
}

void ZOptixViewport::cleanupView()
{
    emit sig_cleanUpView();
}

void ZOptixViewport::modifyLightData(UI_VECTYPE pos, UI_VECTYPE scale, UI_VECTYPE rotate, UI_VECTYPE color, float intensity, QString name, UI_VECTYPE skipParam)
{
    emit sig_modifyLightData(pos, scale, rotate, color, intensity, name, skipParam);
}

void ZOptixViewport::stopRender()
{
    emit stopRenderOptix();
}

void ZOptixViewport::resumeRender()
{
    emit resumeWork();
}

void ZOptixViewport::recordVideo(VideoRecInfo recInfo)
{
    emit sigRecordVideo(recInfo);
}

void ZOptixViewport::screenshoot(QString path, QString type, int resx, int resy)
{
    std::string sType = type.toStdString();
    // bool ret = m_renderImage.save(path, sType.c_str());
    // if (!ret)
    // {
        //meet some unsupported type by QImage.
        emit sigscreenshoot(path, type, resx, resy);
    // }
}

void ZOptixViewport::cancelRecording(VideoRecInfo recInfo)
{
    m_worker->cancelRecording();
}

void ZOptixViewport::onFrameRunFinished(int frame)
{
    emit sig_frameRunFinished(frame);
}

void ZOptixViewport::updateCameraProp(float aperture, float disPlane, UI_VECTYPE skipParam)
{
    emit sig_updateCameraProp(aperture, disPlane, skipParam);
}

void ZOptixViewport::updatePerspective()
{
    m_camera->updatePerspective();
}
void ZOptixViewport::setCameraScale(const int scale)
{
    m_camera->setScale(scale);
}
void ZOptixViewport::setCameraRes(const QVector2D& res)
{
    m_camera->setRes(res);
}

void ZOptixViewport::setSafeFrames(bool bLock, int nx, int ny)
{
    emit sig_setSafeFrames(bLock, nx, ny);
}

void ZOptixViewport::setNumSamples(int samples)
{
    auto scene = m_zenovis->getSession()->get_scene();
    if (scene) {
        scene->drawOptions->num_samples = samples;
    }
}

void ZOptixViewport::showBackground(bool bShow)
{
    emit sig_setBackground(bShow);
}

void ZOptixViewport::setSampleNumber(int sample_number)
{
    emit sig_setSampleNumber(sample_number);
}

void ZOptixViewport::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    QSize sz = event->size();

    int nx = sz.width();
    int ny = sz.height();

    float ratio = devicePixelRatioF();
    zeno::log_trace("nx={}, ny={}, dpr={}", nx, ny, ratio);
    m_camera->setRes(QVector2D(nx * ratio, ny * ratio));
    m_camera->updatePerspective();
}

void ZOptixViewport::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
        setSimpleRenderOption();
    }
    else if (event->button() == Qt::LeftButton) {
        m_bMovingNode = true;
        auto currentPos = event->pos();
        start_pos = zeno::vec2f(currentPos.x(), currentPos.y());
        last_pos = start_pos;
        try_axis = {};

        if (!gizmo_id_buffer.isNull()) {
            auto gizmo_id = gizmo_id_buffer.pixelColor(currentPos);
            if (gizmo_id.isValid()) {
                auto gizmo_painted = gizmo_id.red();
                auto gizmo_mode = gizmo_id.green();
                auto gizmo_type = gizmo_id.blue();
                axis = gizmo_type_to_axis.at(gizmo_type);
                try_axis = axis;
            }
        }

        setSimpleRenderOption();
    } else if(event->button() == Qt::RightButton) {
        setSimpleRenderOption();
    }
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ZOptixViewport::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = false;
    }
    else if (event->button() == Qt::LeftButton) {
        m_bMovingNode = false;
        start_pos = {};
        last_pos = {};
        try_axis = {};
    }
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseReleaseEvent(event);
    update();
}

void ZOptixViewport::mouseMoveEvent(QMouseEvent* event)
{
    if (event->buttons() == Qt::NoButton) {
        if (!m_bMovingNode) {
            auto currentPos = event->pos();
            if (!gizmo_id_buffer.isNull()) {
                auto gizmo_id = gizmo_id_buffer.pixelColor(currentPos);
                if (gizmo_id.isValid()) {
                    auto gizmo_painted = gizmo_id.red();
                    auto gizmo_mode = gizmo_id.green();
                    auto gizmo_type = gizmo_id.blue();
                    auto old_try_axis = try_axis;
                    try_axis = gizmo_type_to_axis.at(gizmo_type);
                    if (old_try_axis != try_axis) {
                        update();
                    }
                }
            }
        }
        return;
    }
    if (m_bMovingNode) {
        auto currentPos = event->pos();
        auto cur_pos = zeno::vec2f(currentPos.x(), currentPos.y());
        if (!last_pos.has_value()) {
            start_pos = cur_pos;
            last_pos = cur_pos;
            return;
        }
        auto delta = cur_pos - last_pos.value();
        Json msg;
        msg["MessageType"] = "Xform";
        msg["Mode"] = mode;
        msg["Axis"] = axis;
        msg["LocalSpace"] = local_space;
        msg["Delta"] = {delta[0], delta[1]};
        msg["LastPos"] = {last_pos.value()[0], last_pos.value()[1]};
        msg["CurPos"] = {cur_pos[0], cur_pos[1]};
        last_pos = cur_pos;
        auto msg_str = msg.dump();
        emit sig_sendOptixMessage(QString::fromStdString(msg_str));

        update();
        return;
    }
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
    }
    setSimpleRenderOption();

    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ZOptixViewport::mouseDoubleClickEvent(QMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseDoubleClickEvent(event);
    update();
}

void ZOptixViewport::wheelEvent(QWheelEvent* event)
{
    m_bMovingCamera = true;
    //m_wheelEventDally->start(100);
    setSimpleRenderOption();

    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
    update();
}

void ZOptixViewport::keyPressEvent(QKeyEvent* event)
{
    _base::keyPressEvent(event);
    //qInfo() << event->key();
    ZenoSettingsManager &settings = ZenoSettingsManager::GetInstance();
    int key = settings.getShortCut(ShortCut_MovingHandler);
    int uKey = event->key();
    Qt::KeyboardModifiers modifiers = event->modifiers();
    if (modifiers & Qt::ShiftModifier) {
        uKey += Qt::SHIFT;
    }
    if (modifiers & Qt::ControlModifier) {
        uKey += Qt::CTRL;
    }
    if (modifiers & Qt::AltModifier) {
        uKey += Qt::ALT;
    }
    if (m_camera->fakeKeyPressEvent(uKey)) {
        zenoApp->getMainWindow()->updateViewport();
        return;
    }
    /*
    if (uKey == key)
        this->changeTransformOperation(0);
    key = settings.getShortCut(ShortCut_RevolvingHandler);
    if (uKey == key)
        this->changeTransformOperation(1);
    key = settings.getShortCut(ShortCut_ScalingHandler);
    if (uKey == key)
        this->changeTransformOperation(2);
    key = settings.getShortCut(ShortCut_CoordSys);
    if (uKey == key)
        this->changeTransformCoordSys();
    */

    key = settings.getShortCut(ShortCut_FrontView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::front_view);
    key = settings.getShortCut(ShortCut_RightView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::right_view);
    key = settings.getShortCut(ShortCut_VerticalView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::top_view);
    key = settings.getShortCut(ShortCut_InitViewPos);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::back_to_origin);

    key = settings.getShortCut(ShortCut_BackView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::back_view);
    key = settings.getShortCut(ShortCut_LeftView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::left_view);
    key = settings.getShortCut(ShortCut_UpwardView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::bottom_view);

    key = settings.getShortCut(ShortCut_InitHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(0);
    key = settings.getShortCut(ShortCut_AmplifyHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(1);
    key = settings.getShortCut(ShortCut_ReduceHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(2);
    {
        auto old_mode = mode;
        if (uKey == Qt::Key_Escape) {
            mode = "";
        }
        else if(uKey == Qt::Key_E) {
            mode = (mode != "EasyScale")? "EasyScale": "Scale";
        }
        else if(uKey == Qt::Key_R) {
            mode = (mode == "Rotate")? "RotateScreen": "Rotate";
        }
        else if(uKey == Qt::Key_T) {
            mode = "Translate";
        }
//        zeno::log_info("{} -> {}", old_mode, mode);
    }
}

void ZOptixViewport::keyReleaseEvent(QKeyEvent* event)
{
    _base::keyReleaseEvent(event);
    int uKey = event->key();
    if (m_camera->fakeKeyReleaseEvent(uKey)) {
        return;
    }
}

void ZOptixViewport::paintEvent(QPaintEvent* event)
{
    if (!m_renderImage.isNull())
    {
        QPainter painter(this);
        auto *session = m_zenovis->getSession();
        if (session != nullptr && session->is_lock_window()) {
            auto *scene = session->get_scene();
            auto offset = scene->camera->viewport_offset;
            painter.drawImage(offset[0], offset[1], m_renderImage);
        }
        else {
            painter.drawImage(0, 0, m_renderImage);
        }
    }
}

std::tuple<std::string, std::string, bool> ZOptixViewport::get_srt_mode_axis() {
    return {mode, axis, local_space};
}

void ZOptixViewport::set_srt_mode_axis(const std::string &_mode, const std::string &_axis, bool _local_space) {
    mode = _mode;
    axis = _axis;
    local_space = _local_space;
    if (mode.size()) {
        zenoApp->getMainWindow()->statusbarShowMessage(zeno::format("Mode: {}, Axis: {}, local: {}", mode, axis, local_space));
    }
    else {
        zenoApp->getMainWindow()->statusbarShowMessage("");
    }
}

static glm::vec2 pos_ws2ss(glm::vec3 pos_WS, glm::mat4 const &vp_mat, glm::vec2 resolution) {

    auto pivot_CS = vp_mat * glm::vec4(pos_WS, 1.0f);
    glm::vec2 pivot_SS = (pivot_CS / pivot_CS[3]);
    pivot_SS = pivot_SS * 0.5f + 0.5f;
    pivot_SS[1] = 1 - pivot_SS[1];
    pivot_SS = pivot_SS * resolution;
    return pivot_SS;
}
static bool is_valid(glm::vec3 pos_WS, glm::mat4 const &vp_mat, glm::vec2 resolution) {

    auto pivot_CS = vp_mat * glm::vec4(pos_WS, 1.0f);
    float value = pivot_CS.z/pivot_CS.w;
    return value>-1 && value<1;
}
void draw_3d_segment_to_screen(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 p1_WS, glm::vec3 p2_WS, QColor color, float line_width, QColor color_id) {
    bool p1_valid = is_valid(p1_WS, vp_mat, resolution);
    bool p2_valid = is_valid(p1_WS, vp_mat, resolution);
    auto p1_SS = pos_ws2ss(p1_WS, vp_mat, resolution);
    auto p2_SS = pos_ws2ss(p2_WS, vp_mat, resolution);
    if(p1_valid==false || p2_valid==false)
        return;

    painter.setPen(QPen(color, line_width));
    painter.drawLine(p1_SS.x, p1_SS.y, p2_SS.x, p2_SS.y);
    painter2.setPen(QPen(color_id, line_width * 5));
    painter2.drawLine(p1_SS.x, p1_SS.y, p2_SS.x, p2_SS.y);
}

void draw_3d_point_to_screen(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 p1_WS, QColor color, float half_width, QColor color_id) {
    auto p1_SS = pos_ws2ss(p1_WS, vp_mat, resolution);
    if(is_valid(p1_WS, vp_mat, resolution)==false)
        return;
    painter.fillRect(p1_SS.x - half_width, p1_SS.y - half_width, half_width * 2, half_width * 2, color);
    painter2.fillRect(p1_SS.x - half_width, p1_SS.y - half_width, half_width * 2, half_width * 2, color_id);
}

void draw_2d_circle(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 p1_WS, QColor color, float radius, float line_width, QColor color_id) {
    auto p1_SS = pos_ws2ss(p1_WS, vp_mat, resolution);
    if(is_valid(p1_WS, vp_mat, resolution)==false)
        return;
    painter.setPen(QPen(color, line_width));
    painter.drawEllipse(QPointF(p1_SS.x, p1_SS.y), radius, radius);
    painter2.setPen(QPen(color_id, line_width * 2));
    painter2.drawEllipse(QPointF(p1_SS.x, p1_SS.y), radius, radius);
}

void draw_2d_circle_with_filled_id(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 p1_WS, QColor color, float radius, float line_width, QColor color_id) {
    auto p1_SS = pos_ws2ss(p1_WS, vp_mat, resolution);
    if(is_valid(p1_WS, vp_mat, resolution)==false)
        return;
    painter.setPen(QPen(color, line_width));
    painter.drawEllipse(QPointF(p1_SS.x, p1_SS.y), radius, radius);
    painter2.setBrush(QBrush(color_id));
    painter2.setPen(QPen(color_id, line_width * 2));
    painter2.drawEllipse(QPointF(p1_SS.x, p1_SS.y), radius, radius);
}
static void draw_axis(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat
               , glm::vec3 center_WS, glm::vec3 e0, glm::vec3 e1, glm::vec3 e2
               , float axis_len, const std::string &try_axis, QColor id1, QColor id2, QColor id3
)
{
    auto x_axis_tip_WS = center_WS + e0 * axis_len;
    auto y_axis_tip_WS = center_WS + e1 * axis_len;
    auto z_axis_tip_WS = center_WS + e2 * axis_len;
    auto r_color = QColor(200, 50, 50);
    auto g_color = QColor(50, 200, 50);
    auto b_color = QColor(50, 50, 200);
    auto gray_color = QColor(200, 200, 200);
    auto l_r_color = QColor(255, 50, 50);
    auto l_g_color = QColor(50, 255, 50);
    auto l_b_color = QColor(50, 50, 255);
    auto white_color = QColor(255, 255, 255);
    for(int i=0;i<10;i++) {
        float t0 = ((float)i)/10.0f;
        float t1 = ((float)i + 1.0)/10.0f;
        glm::vec3 dir0 = x_axis_tip_WS - center_WS;
        glm::vec3 dir1 = y_axis_tip_WS - center_WS;
        glm::vec3 dir2 = z_axis_tip_WS - center_WS;
        auto px0 = center_WS + t0 * dir0;
        auto px1 = center_WS + t1 * dir0;
        auto py0 = center_WS + t0 * dir1;
        auto py1 = center_WS + t1 * dir1;
        auto pz0 = center_WS + t0 * dir2;
        auto pz1 = center_WS + t1 * dir2;
        draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, px0, px1,
                                  try_axis == "X" ? l_r_color : r_color, 2, id1);
        draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, py0, py1,
                                  try_axis == "Y" ? l_g_color : g_color, 2, id2);
        draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, pz0, pz1,
                                  try_axis == "Z" ? l_b_color : b_color, 2, id3);
    }

}
static void draw_translate_axis(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat
               , glm::vec3 center_WS, glm::vec3 e0, glm::vec3 e1, glm::vec3 e2
               , float axis_len, const std::string &try_axis
) {
    auto x_axis_tip_WS = center_WS + e0 * axis_len;
    auto y_axis_tip_WS = center_WS + e1 * axis_len;
    auto z_axis_tip_WS = center_WS + e2 * axis_len;

    auto r_color = QColor(200, 50, 50);
    auto g_color = QColor(50, 200, 50);
    auto b_color = QColor(50, 50, 200);
    auto gray_color = QColor(200, 200, 200);

    auto l_r_color = QColor(255, 50, 50);
    auto l_g_color = QColor(50, 255, 50);
    auto l_b_color = QColor(50, 50, 255);
    auto white_color = QColor(255, 255, 255);
    draw_axis(painter, painter2, resolution, vp_mat, center_WS, e0, e1, e2, axis_len, try_axis,QColor(1, 1, 1), QColor(1, 1, 2),QColor(1, 1, 3));

    draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, center_WS, try_axis == "XYZ"? white_color: gray_color, 5, QColor(1, 1, 4));

    auto x_plane_tip_WS = center_WS + e1 * axis_len + e2 * axis_len;
    auto y_plane_tip_WS = center_WS + e0 * axis_len + e2 * axis_len;
    auto z_plane_tip_WS = center_WS + e0 * axis_len + e1 * axis_len;

    draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, x_plane_tip_WS, try_axis == "YZ"? l_r_color: r_color, 5, QColor(1, 1, 5));
    draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, y_plane_tip_WS, try_axis == "XZ"? l_g_color: g_color, 5, QColor(1, 1, 6));
    draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, z_plane_tip_WS, try_axis == "XY"? l_b_color: b_color, 5, QColor(1, 1, 7));
}

static void draw_display_axis(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat
               , glm::vec3 center_WS, glm::vec3 e0, glm::vec3 e1, glm::vec3 e2
               , float axis_len
) {
    auto x_axis_tip_WS = center_WS + e0 * axis_len;
    auto y_axis_tip_WS = center_WS + e1 * axis_len;
    auto z_axis_tip_WS = center_WS + e2 * axis_len;

    auto r_color = QColor(200, 50, 50);
    auto g_color = QColor(50, 200, 50);
    auto b_color = QColor(50, 50, 200);
    draw_axis(painter, painter2, resolution, vp_mat, center_WS, e0, e1, e2, axis_len, "",QColor(0, 0, 0), QColor(0, 0, 0),QColor(0, 0, 0));
//    draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, center_WS, x_axis_tip_WS, r_color, 2, QColor(0, 0, 0));
//    draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, center_WS, y_axis_tip_WS, g_color, 2, QColor(0, 0, 0));
//    draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, center_WS, z_axis_tip_WS, b_color, 2, QColor(0, 0, 0));
}

static void draw_scale_axis(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat
               , glm::vec3 center_WS, glm::vec3 e0, glm::vec3 e1, glm::vec3 e2
               , float axis_len, const std::string &try_axis, bool easy
) {
    auto x_axis_tip_WS = center_WS + e0 * axis_len;
    auto y_axis_tip_WS = center_WS + e1 * axis_len;
    auto z_axis_tip_WS = center_WS + e2 * axis_len;

    auto r_color = QColor(200, 50, 50);
    auto g_color = QColor(50, 200, 50);
    auto b_color = QColor(50, 50, 200);
    auto gray_color = QColor(200, 200, 200);

    auto l_r_color = QColor(255, 50, 50);
    auto l_g_color = QColor(50, 255, 50);
    auto l_b_color = QColor(50, 50, 255);
    auto white_color = QColor(255, 255, 255);
    if (!easy) {
        draw_axis(painter, painter2, resolution, vp_mat, center_WS, e0, e1, e2, axis_len, try_axis,QColor(1, 3, 1), QColor(1, 3, 2),QColor(1, 3, 3));

        auto x_plane_tip_WS = center_WS + e1 * axis_len + e2 * axis_len;
        auto y_plane_tip_WS = center_WS + e0 * axis_len + e2 * axis_len;
        auto z_plane_tip_WS = center_WS + e0 * axis_len + e1 * axis_len;

        draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, x_plane_tip_WS, try_axis == "YZ"? l_r_color: r_color, 5, QColor(1, 3, 5));
        draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, y_plane_tip_WS, try_axis == "XZ"? l_g_color: g_color, 5, QColor(1, 3, 6));
        draw_3d_point_to_screen(painter, painter2, resolution, vp_mat, z_plane_tip_WS, try_axis == "XY"? l_b_color: b_color, 5, QColor(1, 3, 7));
    }

    draw_2d_circle(painter, painter2, resolution, vp_mat, center_WS, try_axis == "XYZ"? white_color: gray_color, 50, 4, QColor(1, 3, 4));
}
void draw_circle(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 center, glm::vec3 e0, glm::vec3 e1, float radius, QColor color, int segment, QColor color_id) {
    float dtheta = glm::radians(360.0f / float(segment));
    for (int i = 0; i < segment; i++) {
        float theta0 = float(i) * dtheta;
        float theta1 = float(i + 1) * dtheta;
        glm::vec3 p0 = cos(theta0) * e0 * radius + sin(theta0) * e1 * radius + center;
        glm::vec3 p1 = cos(theta1) * e0 * radius + sin(theta1) * e1 * radius + center;
        draw_3d_segment_to_screen(painter, painter2, resolution, vp_mat, p0, p1, color, 2, color_id);
    }
}
void draw_rotation_axis(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 center, glm::vec3 e0, glm::vec3 e1, glm::vec3 e2, float radius, const std::string &try_axis)
{
    auto r_color = QColor(200, 50, 50);
    auto g_color = QColor(50, 200, 50);
    auto b_color = QColor(50, 50, 200);

    auto l_r_color = QColor(255, 50, 50);
    auto l_g_color = QColor(50, 255, 50);
    auto l_b_color = QColor(50, 50, 255);

    draw_circle(painter, painter2, resolution, vp_mat, center, e1, e2, radius, try_axis == "X"? l_r_color: r_color, 30, QColor(1, 2, 1));
    draw_circle(painter, painter2, resolution, vp_mat, center, e0, e2, radius, try_axis == "Y"? l_g_color: g_color, 30, QColor(1, 2, 2));
    draw_circle(painter, painter2, resolution, vp_mat, center, e0, e1, radius, try_axis == "Z"? l_b_color: b_color, 30, QColor(1, 2, 3));
}

void draw_rotation_screen_axis(QPainter &painter, QPainter &painter2, glm::vec2 resolution, glm::mat4 vp_mat, glm::vec3 center, const std::string &try_axis)
{
    auto color_CameraUpRight = QColor(50, 50, 200);
    auto l_color_CameraUpRight = QColor(50, 50, 255);

    draw_2d_circle_with_filled_id(painter, painter2, resolution, vp_mat, center
                                  , try_axis == "CameraUpRight"? l_color_CameraUpRight: color_CameraUpRight, 50, 4, QColor(1, 2, 8));
}
void ZOptixViewport::drawAxis(QImage &img) {
    gizmo_id_buffer = QImage(img.size(), img.format());
    gizmo_id_buffer.fill(Qt::black);

    if (!axis_coord.has_value()) {
        return;
    }

    auto center_WS = glm::vec3(axis_coord.value()[3]);
    float axis_len = 1.0f / 10.0f;
    auto scale_factor = glm::distance(m_camera->getPos(), center_WS);
    auto x_axis_dir = glm::normalize(glm::vec3(axis_coord.value()[0]));
    auto y_axis_dir = glm::normalize(glm::vec3(axis_coord.value()[1]));
    auto z_axis_dir = glm::normalize(glm::vec3(axis_coord.value()[2]));
    auto res = m_camera->res();
    auto resolution = glm::vec2(res.x(), res.y());
    auto scene = m_zenovis->getSession()->get_scene();
    auto vp_mat = scene->camera->get_proj_matrix() * scene->camera->get_view_matrix();

    QPainter painter(&img);

    QPainter painter2(&gizmo_id_buffer);

    if (mode=="") {
        //draw_display_axis(painter, painter2, resolution, vp_mat, center_WS, x_axis_dir, y_axis_dir, z_axis_dir, scale_factor * axis_len);
    }
    else if (mode == "Rotate") {
        draw_rotation_axis(painter, painter2, resolution, vp_mat, center_WS, x_axis_dir, y_axis_dir, z_axis_dir, scale_factor * axis_len, try_axis);
    }
    else if (mode == "RotateScreen") {
        draw_rotation_screen_axis(painter, painter2, resolution, vp_mat, center_WS, try_axis);
    }
    else if (mode == "Translate") {
        draw_translate_axis(painter, painter2, resolution, vp_mat, center_WS, x_axis_dir, y_axis_dir, z_axis_dir, scale_factor * axis_len, try_axis);
    }
    else if (mode == "Scale") {
        draw_scale_axis(painter, painter2, resolution, vp_mat, center_WS, x_axis_dir, y_axis_dir, z_axis_dir, scale_factor * axis_len, try_axis, false);
    }
    else if (mode == "EasyScale") {
        draw_scale_axis(painter, painter2, resolution, vp_mat, center_WS, x_axis_dir, y_axis_dir, z_axis_dir, scale_factor * axis_len, try_axis, true);
    }

    painter.end();
    painter2.end();
}