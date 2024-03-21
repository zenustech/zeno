#ifndef __DISPLAY_WIDGET_H__
#define __DISPLAY_WIDGET_H__

#include <QtWidgets>
#include "common.h"
#include "recordvideomgr.h"
#include "viewportinteraction/picker.h"
#include "launch/corelaunch.h"
#include "dock/docktabcontent.h"
#include <zenoio/include/common.h>
#include <zenovis/Camera.h>

class ViewportWidget;
#ifdef ZENO_OPTIX_PROC
class ZOptixProcViewport;
#else
class ZOptixViewport;
#endif
class CameraKeyframeWidget;

class DisplayWidget : public QWidget
{
    Q_OBJECT
public:
    DisplayWidget(bool bGLView, QWidget* parent = nullptr);
    ~DisplayWidget();
    void init();
    QSize sizeHint() const override;
    Zenovis* getZenoVis() const;
    void runAndRecord(const VideoRecInfo& info);
    void testCleanUp();
    void cleanUpScene();
    void beforeRun();
    void afterRun();
    void changeTransformOperation(const QString &node);
    void changeTransformOperation(int mode);
    QSize viewportSize() const;
    void resizeViewport(QSize sz);
    std::shared_ptr<zeno::Picker> picker() const;
    void updateCameraProp(float aperture, float disPlane);
    void updatePerspective();
    void setNumSamples(int samples);
    void setSafeFrames(bool bLock, int nx, int ny);
    void setCameraRes(const QVector2D& res);
    void setSimpleRenderOption();
    void setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly);
    bool isCameraMoving() const;
    bool isPlaying() const;
    bool isGLViewport() const;
    void setViewWidgetInfo(DockContentWidgetInfo& info);
    void setSliderFeq(int feq);
#ifdef ZENO_OPTIX_PROC
    ZOptixProcViewport* optixViewport() const;
#else
    ZOptixViewport* optixViewport() const;
#endif
    void killOptix();
    void moveToFrame(int frame);
    void setIsCurrent(bool isCurrent);
    bool isCurrent();
    void setLoopPlaying(bool enable);
    std::tuple<int, int, bool> getOriginWindowSizeInfo();
    void cameraLookTo(int dir);
protected:
    void mouseReleaseEvent(QMouseEvent* event) override;
public slots:
    void updateFrame(const QString& action = "");
    void onRun();
    void onRun(LAUNCH_PARAM launchParam);
    void onRecord();
    void onRecord_slient(const VideoRecInfo& recInfo);
    bool onRecord_cmd(const VideoRecInfo& recInfo);
    void onScreenShoot();
    void onKill();
    void onPlayClicked(bool);
    void onSliderValueChanged(int);
    void onRunFinished();
    void onCommandDispatched(int actionType, bool bTriggered);
    void onNodeSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    void onMouseHoverMoved();
    void onDockViewAction(bool triggered);
    void onSetCamera(zenovis::ZOptixCameraSettingInfo value);
    void onSetBackground(bool bShowBackground);
    zenovis::ZOptixCameraSettingInfo getCamera() const;

signals:
    void frameUpdated(int new_frame);
    void frameRunFinished(int frame);
    void optixProcStartRecord();

public:
    enum DockViewActionType {
        ACTION_FRONT_VIEW = 0,
        ACTION_RIGHT_VIEW,
        ACTION_TOP_VIEW,
        ACTION_BACK_VIEW,
        ACTION_LEFT_VIEW,
        ACTION_BOTTOM_VIEW,
        ACTION_ORIGIN_VIEW,
        ACTION_FOCUS,
    };

private slots:
    void onFrameFinish(int frame);

private:
    bool isOptxRendering() const;
    void initRecordMgr();

    ViewportWidget* m_glView;
#ifdef ZENO_OPTIX_PROC
    ZOptixProcViewport* m_optixView;
#else
    ZOptixViewport* m_optixView;
#endif
    CameraKeyframeWidget* m_camera_keyframe;
    QTimer* m_pTimer;
    RecordVideoMgr m_recordMgr;
    bool m_bRecordRun;
    const bool m_bGLView;
    int m_sliderFeq = 1000 / 24;
    bool bIsCurrent = false;

    std::tuple<int, int, bool> originWindowSizeInfo{-1, -1, false};
};

#endif