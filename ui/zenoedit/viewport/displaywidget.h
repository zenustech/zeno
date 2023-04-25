#ifndef __DISPLAY_WIDGET_H__
#define __DISPLAY_WIDGET_H__

#include <QtWidgets>
#include "common.h"
#include "recordvideomgr.h"

class ViewportWidget;
class CameraKeyframeWidget;

class DisplayWidget : public QWidget
{
    Q_OBJECT
public:
    DisplayWidget(QWidget* parent = nullptr);
    ~DisplayWidget();
    void init();
    QSize sizeHint() const override;
    ViewportWidget* getViewportWidget();
    void runAndRecord(const VideoRecInfo& info);
    void testCleanUp();
    void beforeRun();
    void afterRun();

public slots:
    void updateFrame(const QString& action = "");
    void onRun();
    void onRun(int frameStart, int frameEnd, bool applyLightAndCameraOnly = false);
    void onRecord();
    void onScreenShoot();
    void onKill();
    void onPlayClicked(bool);
    void onSliderValueChanged(int);
    void onFinished();
    void onCommandDispatched(int actionType, bool bTriggered);
    void onNodeSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);

signals:
    void frameUpdated(int new_frame);

private:
    bool isOptxRendering() const;
    void initRecordMgr();
    void moveToFrame(int frame);

    ViewportWidget* m_view;
    CameraKeyframeWidget* m_camera_keyframe;
    QTimer* m_pTimer;
    RecordVideoMgr m_recordMgr;
    bool m_bRecordRun;
    static const int m_updateFeq = 16;
    static const int m_sliderFeq = 16;
};

#endif