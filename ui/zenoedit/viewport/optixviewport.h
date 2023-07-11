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
    OptixWorker(QObject* parent = nullptr);
    ~OptixWorker();
    QImage renderImage() const;

signals:
    void renderIterate(QImage);
    void sig_recordFinished();
    void sig_frameRecordFinished(int frame);
    void sig_recordCanceled();

public slots:
    void initialize();
    void stop();
    void onWorkThreadStarted();
    void needUpdateCamera();
    void updateFrame();
    void recordVideo(VideoRecInfo recInfo);
    void onPlayToggled(bool bToggled);
    void onFrameSwitched(int frame);
    void cancelRecording();
    void setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly);
    void setNumSamples(int samples);
    void onSetSafeFrames(bool bLock, int nx, int ny);
    void setResolution(const QVector2D& res);
    void setSimpleRenderOption();
    void cameraLookTo(int dir);
    void updateCameraProp(float aperture, float disPlane);
    void resizeTransformHandler(int dir);
    //QMouseEvent unknown:
    void fakeMousePressEvent(QMouseEvent* event);
    void fakeMouseReleaseEvent(QMouseEvent* event);
    void fakeMouseMoveEvent(QMouseEvent* event);
    void fakeWheelEvent(QWheelEvent* event);
    void fakeMouseDoubleClickEvent(QMouseEvent* event);

private:
    bool recordFrame_impl(VideoRecInfo recInfo, int frame);

    CameraControl* m_camera;
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
    void setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly);
    void cameraLookTo(int dir);
    void updateCameraProp(float aperture, float disPlane);
    void updatePerspective();
    void setCameraRes(const QVector2D& res);
    void setSafeFrames(bool bLock, int nx, int ny);
    void setNumSamples(int samples);
    Zenovis* getZenoVis() const;
    bool isCameraMoving() const;
    void updateCamera();
    void stopRender();
    void resumeRender();
    void recordVideo(VideoRecInfo recInfo);
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
    void sig_togglePlayButton(bool bToggled);
    void sig_switchTimeFrame(int frame);
    void sig_setSafeFrames(bool bLock, int nx, int ny);
    void sig_cancelRecording();
    void sig_setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly);
    void sig_setResolution(const QVector2D& res);
    void sig_setSimpleRenderOption();
    void sig_cameraLookTo(int);
    void sig_updateCameraProp(float aperture, float disPlane);
    void sig_resizeTransformHandler(int dir);
    void sig_setNumSamples(int samples);
    //QMouseEvent unknown:
    void sig_fakeMousePressEvent(QMouseEvent* event);
    void sig_fakeMouseReleaseEvent(QMouseEvent* event);
    void sig_fakeMouseMoveEvent(QMouseEvent* event);
    void sig_fakeWheelEvent(QWheelEvent* event);
    void sig_fakeMouseDoubleClickEvent(QMouseEvent* event);

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
    QThread m_thdOptix;
    bool updateLightOnce;
    bool m_bMovingCamera;
    QImage m_renderImage;
};

#endif