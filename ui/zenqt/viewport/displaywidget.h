#ifndef __DISPLAY_WIDGET_H__
#define __DISPLAY_WIDGET_H__

#include <QtWidgets>
#include "uicommon.h"
#include "recordvideomgr.h"
#include "viewport/picker.h"

#include "layout/docktabcontent.h"
#include "layout/winlayoutrw.h"
#include <zenovis/Camera.h>

class ViewportWidget;
class ZOpenGLQuickView;
#ifdef ZENO_OPTIX_PROC
class ZOptixProcViewport;
#else
class ZOptixViewport;
#endif
class CameraKeyframeWidget;

#define BASE_QML_VIEWPORT


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
    void cleanupView();
    void cleanUpScene();
    void beforeRun();
    void afterRun();
    void reload(const zeno::render_reload_info& info);
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
    void setRenderSeparately(/*runType runtype*/);
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
    ZOpenGLQuickView* quickGLViewport() const;
    void killOptix();
    void moveToFrame(int frame);
    void setIsCurrent(bool isCurrent);
    bool isCurrent();
    void setLoopPlaying(bool enable);
    std::tuple<int, int, bool> getOriginWindowSizeInfo();
    void cameraLookTo(zenovis::CameraLookToDir dir);
protected:
    void mouseReleaseEvent(QMouseEvent* event) override;

public slots:
    void updateFrame(const QString& action = "");
    void onRun();
    void onRecord();
    void onRecord_slient(const VideoRecInfo& recInfo);
    bool onRecord_cmd(const VideoRecInfo& recInfo);
    void onScreenShoot();
    void onKill();
    void onPlayClicked(bool);
    void onSliderValueChanged(int);
    void onRunFinished();
    void onCommandDispatched(int actionType, bool bTriggered);
    void onNodeSelected(GraphModel* subgraph, const QModelIndexList& nodes, bool select);
    void onMouseHoverMoved();
    void onDockViewAction(bool triggered);
    void onRenderRequest(QString nodeuuidpath);
    void onCalcFinished(bool bSucceed, QString, QString, const zeno::render_reload_info&);
    void onSetCamera(zenovis::ZOptixCameraSettingInfo value);
    void onSetBackground(bool bShowBackground);
    void setSampleNumber(int sample_number);
    zenovis::ZOptixCameraSettingInfo getCamera() const;

signals:
    void frameUpdated(int new_frame);
    void frameRunFinished(int frame);
    void optixProcStartRecord();
    void render_reload_finished();

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
    void sendTaskToServer(const VideoRecInfo& info);
    void submit(std::vector<zeno::render_update_info> infos);
    void submit(const zeno::render_reload_info& render_summary);

#ifdef BASE_QML_VIEWPORT
    ZOpenGLQuickView* m_glView;
#else
    ViewportWidget* m_glView;
#endif

#ifdef ZENO_OPTIX_PROC
    ZOptixProcViewport* m_optixView;
#else
    ZOptixViewport* m_optixView;
#endif
    QScopedPointer<CameraKeyframeWidget> m_camera_keyframe;
    QTimer* m_pTimer;
    RecordVideoMgr m_recordMgr;
    bool m_bRecordRun;
    const bool m_bGLView;
    int m_sliderFeq = 1000 / 24;
    bool bIsCurrent = false;

    std::tuple<int, int, bool> originWindowSizeInfo{-1, -1, false};
};

#endif