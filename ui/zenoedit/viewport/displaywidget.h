#ifndef __DISPLAY_WIDGET_H__
#define __DISPLAY_WIDGET_H__

#include <QtWidgets>
#include "common.h"
#include "recordvideomgr.h"
#include "viewportinteraction/picker.h"

class ViewportWidget;
class ZOptixViewport;
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
    void beforeRun();
    void afterRun();
    void changeTransformOperation(const QString &node);
    void changeTransformOperation(int mode);
    QSize viewportSize() const;
    void resizeViewport(QSize sz);
    std::shared_ptr<zeno::Picker> picker() const;
    void updateCameraProp(float aperture, float disPlane);
    void setSimpleRenderOption();
    bool isCameraMoving() const;
    bool isPlaying() const;
    bool isGLViewport() const;
    ZOptixViewport* optixViewport() const;
    void killOptix();

public slots:
    void updateFrame(const QString& action = "");
    void onRun();
    void onRun(int frameStart, int frameEnd, bool applyLightAndCameraOnly = false);
    void onRecord();
    void onScreenShoot();
    void onKill();
    void onPlayClicked(bool);
    void onSliderValueChanged(int);
    void onRunFinished();
    void onCommandDispatched(int actionType, bool bTriggered);
    void onNodeSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);

signals:
    void frameUpdated(int new_frame);
    void frameRunFinished(int frame);

private:
    bool isOptxRendering() const;
    void initRecordMgr();
    void moveToFrame(int frame);

    ViewportWidget* m_glView;
    ZOptixViewport* m_optixView;
    CameraKeyframeWidget* m_camera_keyframe;
    QTimer* m_pTimer;       //actually this timer is only applied on glviewport.
    RecordVideoMgr m_recordMgr;
    bool m_bRecordRun;
    const bool m_bGLView;
    static const int m_sliderFeq = 16;
};

#endif