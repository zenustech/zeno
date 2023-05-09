#ifndef __ZOPTIX_VIEWPORT_H__
#define __ZOPTIX_VIEWPORT_H__

#include <QtWidgets>
#include "recordvideomgr.h"

class Zenovis;
class CameraControl;

class OptixWorker : public QObject
{
    Q_OBJECT
public:
    OptixWorker(Zenovis *pzenoVis);
    QImage renderImage() const;

signals:
    void renderIterate(QImage);
    void sig_recordFinished();
    void sig_frameRecordFinished(int frame);
    void sig_recordInfoSetuped();
    void sig_recordCanceled();

public slots:
    void stop();
    void work();
    void needUpdateCamera();
    void updateFrame();
    void recordVideo(VideoRecInfo recInfo);
    void setupRecording(VideoRecInfo recInfo);
    void onFrameRunFinished(int frame);
    void onPlayToggled(bool bToggled);
    void onFrameSwitched(int frame);
    void cancelRecording();

private:
    bool recordFrame_impl(VideoRecInfo recInfo, int frame);

    Zenovis *m_zenoVis;
    QImage m_renderImg;
    QTimer* m_pTimer;
    bool m_bRecording;
    VideoRecInfo m_recordInfo;
};

class ZOptixViewport : public QWidget
{
    Q_OBJECT
    typedef QWidget _base;
public:
    ZOptixViewport(QWidget* parent = nullptr);
    ~ZOptixViewport();
    void setSimpleRenderOption();
    void cameraLookTo(int dir);
    void updateCameraProp(float aperture, float disPlane);
    Zenovis* getZenoVis() const;
    bool isCameraMoving() const;
    void updateCamera();
    void stopRender();
    void resumeRender();
    void recordVideo(VideoRecInfo recInfo);
    void setupRecording(VideoRecInfo recInfo);
    void cancelRecording(VideoRecInfo recInfo);
    void killThread();

signals:
    void cameraAboutToRefresh();
    void stopRenderOptix();
    void resumeWork();
    void sigRecordVideo(VideoRecInfo recInfo);
    void sig_recordFinished();
    void sig_frameRecordFinished(int frame);
    void sig_frameRunFinished(int frame);
    void sig_setupRecordInfo(VideoRecInfo recInfo);
    void sig_recordInfoSetuped();
    void sig_togglePlayButton(bool bToggled);
    void sig_switchTimeFrame(int frame);
    void sig_cancelRecording();

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