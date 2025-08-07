#ifndef __ZOPTIX_VIEWPORT_H__
#define __ZOPTIX_VIEWPORT_H__

#include <QtWidgets>
#include "recordvideomgr.h"
#include "zenovis/Camera.h"
#include <zenomodel/include/modeldata.h>
#include "launch/corelaunch.h"
#include "tinygltf/json.hpp"
using Json = nlohmann::json;

class Zenovis;
class CameraControl;

class OptixWorker : public QObject
{
    Q_OBJECT
public:
    OptixWorker(QObject* parent = nullptr);
    OptixWorker(Zenovis *pzenoVis);
    ~OptixWorker();
    QImage renderImage() const;

signals:
    void renderIterate(QImage);
    void sig_recordFinished();
    void sig_frameRecordFinished(int frame);
    void sig_recordCanceled();

    void sig_sendToOutline(QString);
    void sig_sendToNodeEditor(QString);
    void sig_sendToOptixViewport(QString);
    void sig_sendToXformPanel(QString);

public slots:
    void stop();
    void work();
    void needUpdateCamera();
    void updateFrame();
    void recordVideo(VideoRecInfo recInfo);
    void screenShoot(QString path, QString type, int resx, int resy);
    void onPlayToggled(bool bToggled);
    void onFrameSwitched(int frame);
    void cancelRecording();
    void setRenderSeparately(int runtype);
    void onSetSafeFrames(bool bLock, int nx, int ny);
    bool recordFrame_impl(VideoRecInfo recInfo, int frame);
    void onSetLoopPlaying(bool enbale);
    void onSetSlidFeq(int feq);
    void onModifyLightData(UI_VECTYPE pos, UI_VECTYPE scale, UI_VECTYPE rotate, UI_VECTYPE color, float intensity, QString nodename, UI_VECTYPE skipParam);
    void onUpdateCameraProp(float aperture, float disPlane, UI_VECTYPE skipParam = UI_VECTYPE());
    void onCleanUpScene();
    void onCleanUpView();
    void onSetBackground(bool bShowBg);
    void onSetSampleNumber(int sample_number);
    void onSendOptixMessage(QString);

    void onSetData(float, float, float, int, bool, bool, bool, bool, float);

private:
    Zenovis *m_zenoVis;
    QImage m_renderImg;
    QTimer* m_pTimer;           //optix sample timer
    bool m_bRecording;
    VideoRecInfo m_recordInfo;
    int m_slidFeq = 1000 / 24;
    const int m_sampleFeq = 16;
    std::optional<std::string> cur_node_uuid;
    std::unordered_map<std::string, std::string> outline_node_to_uuid;
};

class ZOptixViewport : public QWidget
{
    Q_OBJECT
    typedef QWidget _base;
public:
    ZOptixViewport(QWidget* parent = nullptr);
    ~ZOptixViewport();
    void setSimpleRenderOption();
    void setRenderSeparately(runType runtype);
    void cameraLookTo(zenovis::CameraLookToDir dir);
    void updateCameraProp(float aperture, float disPlane, UI_VECTYPE skipParam = UI_VECTYPE());
    void updatePerspective();
    void setCameraRes(const QVector2D& res);
    void setCameraScale(const int scale);
    void setSafeFrames(bool bLock, int nx, int ny);
    void setNumSamples(int samples);
    void showBackground(bool bShow);
    void setSampleNumber(int sample_number);
    Zenovis* getZenoVis() const;
    bool isCameraMoving() const;
    void updateCamera();
    void stopRender();
    void resumeRender();
    void recordVideo(VideoRecInfo recInfo);
    void screenshoot(QString path, QString type, int resx, int resy);
    void cancelRecording(VideoRecInfo recInfo);
    void killThread();
    void setSlidFeq(int feq);
    void modifyLightData(UI_VECTYPE pos, UI_VECTYPE scale, UI_VECTYPE rotate, UI_VECTYPE color, float intensity, QString name, UI_VECTYPE skipParam);
    void cleanUpScene();
    void cleanupView();

    zenovis::ZOptixCameraSettingInfo getdata_from_optix_thread();
    void setdata_on_optix_thread(zenovis::ZOptixCameraSettingInfo value);
    std::tuple<std::string, std::string, bool> get_srt_mode_axis();
    void set_srt_mode_axis(std::string const& mode, std::string const& axis, bool local_space);

signals:
    void cameraAboutToRefresh();
    void stopRenderOptix();
    void resumeWork();
    void sigRecordVideo(VideoRecInfo recInfo);
    void sig_recordFinished();
    void sig_frameRecordFinished(int frame);
    void sig_frameRunFinished(int frame);
    void sig_togglePlayButton(bool bToggled);
    void sig_switchTimeFrame(int frame);
    void sig_setSafeFrames(bool bLock, int nx, int ny);
    void sig_cancelRecording();
    void sig_setRunType(int runtype);
    void sig_setLoopPlaying(bool enable);
    void sig_setSlidFeq(int feq);
    void sigscreenshoot(QString, QString, int, int);
    void sig_modifyLightData(UI_VECTYPE pos, UI_VECTYPE scale, UI_VECTYPE rotate, UI_VECTYPE color, float intensity, QString name, UI_VECTYPE skipParam);
    void sig_updateCameraProp(float aperture, float disPlane, UI_VECTYPE skipParam = UI_VECTYPE());
    void sig_cleanUpScene();
    void sig_cleanUpView();
    void sig_setBackground(bool bShowBg);
    void sig_setSampleNumber(int sample_number);
    void sig_setdata_on_optix_thread(float, float, float, int, bool, bool, bool, bool, float);

    void sig_viewportSendToOutline(QString);
    void sig_viewportSendToNodeEditor(QString);
    void sig_viewportSendToXformPanel(QString);
    void sig_sendOptixMessage(QString);

public slots:
    void onFrameRunFinished(int frame);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;

private:
    void generateModificationNode(std::string _outNodeId, std::string _outSock, std::string _inNodeType, std::string _inSock, std::string _inModifyInfoSock, Json& msg);
    CameraControl* m_camera;
    Zenovis* m_zenovis;
    QThread m_thdOptix;
    bool updateLightOnce;
    bool m_bMovingCamera;
    bool m_bMovingNode = false;
    std::optional<zeno::vec2f> start_pos;
    std::optional<zeno::vec2f> last_pos;
    QImage m_renderImage;
    OptixWorker* m_worker;
    std::string mode;
    std::string axis;
    std::string try_axis;
    bool local_space = true;
    QImage gizmo_id_buffer;
    std::optional<glm::mat4> axis_coord;
    void drawAxis(QImage &img);
    const std::map<int, std::string> gizmo_type_to_axis =  {
        {0, ""},
        {1, "X"},
        {2, "Y"},
        {3, "Z"},
        {4, "XYZ"},
        {5, "YZ"},
        {6, "XZ"},
        {7, "XY"},
        {8, "CameraUpRight"},
    };

    QTimer* m_pauseRenderDally;
};

#endif