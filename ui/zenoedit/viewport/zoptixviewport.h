#ifndef __ZOPTIX_VIEWPORT_PROC_H__
#define __ZOPTIX_VIEWPORT_PROC_H__

#include <QtWidgets>
#include "optixviewport.h"

class Zenovis;
class CameraControl;

class ZOptixProcViewport : public QWidget
{
    Q_OBJECT
    typedef QWidget _base;
public:
    ZOptixProcViewport(QWidget* parent = nullptr);
    ~ZOptixProcViewport();
    void setSimpleRenderOption();
    void setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly);
    void cameraLookTo(int dir);
    void updateViewport();
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
    void onMouseHoverMoved();
    void onFrameSwitched(int frame);

signals:
    void sig_frameRunFinished(int frame);
    void sig_frameRecordFinished(int frame);
    void sig_recordFinished();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

public slots:
    void onFrameRunFinished(int frame);

private:
    void pauseWorkerAndResume();

    CameraControl* m_camera;
    Zenovis* m_zenovis;
    QTimer* m_pauseTimer = nullptr;
    static const int m_resumeTime = 100;

    bool updateLightOnce;
    bool m_bMovingCamera;
    QImage m_renderImage;
    OptixWorker* m_worker;
};

#endif