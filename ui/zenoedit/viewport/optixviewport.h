#ifndef __ZOPTIX_VIEWPORT_H__
#define __ZOPTIX_VIEWPORT_H__

#include <QtWidgets>
#include "recordvideomgr.h"
#include "zenovis/Camera.h"
#include <zenomodel/include/modeldata.h>
#include "launch/corelaunch.h"

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
    void onNodeRemoved(QString);

    void onSetData(float, float, float, bool, bool, bool, bool, float);

private:
    Zenovis *m_zenoVis;
    QImage m_renderImg;
    QTimer* m_pTimer;           //optix sample timer
    bool m_bRecording;
    VideoRecInfo m_recordInfo;
    int m_slidFeq = 1000 / 24;
    const int m_sampleFeq = 16;
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
    void sig_setdata_on_optix_thread(float, float, float, bool, bool, bool, bool, float);

    void sig_viewportSendToOutline(QString);
    void sig_viewportSendToNodeEditor(QString);
    void sig_sendOptixMessage(QString);
    void sig_nodeRemoved(QString);

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
    CameraControl* m_camera;
    Zenovis* m_zenovis;
    QThread m_thdOptix;
    bool updateLightOnce;
    bool m_bMovingCamera;
    QImage m_renderImage;
    OptixWorker* m_worker;
};

#endif