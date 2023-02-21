#ifndef __VIEWPORT_WIDGET_H__
#define __VIEWPORT_WIDGET_H__

#include <QtWidgets>
#include <QtOpenGL>
#include "comctrl/zmenubar.h"
#include "comctrl/zmenu.h"
#include "common.h"
#include "recordvideomgr.h"

#include <viewportinteraction/transform.h>
#include <viewportinteraction/picker.h>

class ZTimeline;
class ZenoMainWindow;

class CameraControl : public QWidget
{
    Q_OBJECT
public:
    CameraControl(QWidget* parent = nullptr);
    void setRes(QVector2D res);
    QVector2D res() const { return m_res; }
    void setAperture(float aperture);
    void setDisPlane(float disPlane);
    void updatePerspective();
    void setKeyFrame();

    void fakeMousePressEvent(QMouseEvent* event);
    void fakeMouseReleaseEvent(QMouseEvent* event);
    void fakeMouseMoveEvent(QMouseEvent* event);
    void fakeWheelEvent(QWheelEvent* event);
     void fakeMouseDoubleClickEvent(QMouseEvent* event);
    void focus(QVector3D center, float radius);
    QVector3D realPos() const;
    QVector3D screenToWorldRay(float x, float y) const;
    QVariant hitOnFloor(float x, float y) const;
    void lookTo(int dir);
    void clearTransformer();
    void changeTransformOperation(const QString& node);
    void changeTransformOperation(int mode);
    void changeTransformCoordSys();
    void resizeTransformHandler(int dir);

private:
    bool m_mmb_pressed;
    float m_theta;
    float m_phi;
    QPointF m_lastPos;
    QPoint m_boundRectStartPos;
    QVector3D  m_center;
    bool m_ortho_mode;
    float m_fov;
    float m_radius;
    float m_aperture;
    float m_focalPlaneDistance;
    QVector2D m_res;

    QSet<int> m_pressedKeys;
};

class ViewportWidget : public QGLWidget
{
    Q_OBJECT
    typedef QGLWidget _base;
public:
    ViewportWidget(QWidget* parent = nullptr);
    ~ViewportWidget();
    void initializeGL() override;
    void resizeGL(int nx, int ny) override;
    void paintGL() override;
    QVector2D cameraRes() const;
    void setCameraRes(const QVector2D& res);
    void updatePerspective();
    void updateCameraProp(float aperture, float disPlane);
    void cameraLookTo(int dir);
    void clearTransformer();
    void changeTransformOperation(const QString& node);
    void changeTransformOperation(int mode);
    void changeTransformCoordSys();
    void setPickTarget(const string& prim_name);
    void bindNodeToPicker(const QModelIndex& node, const QModelIndex& subgraph, const std::string& sock_name);
    void unbindNodeFromPicker();
    void setSimpleRenderOption();

signals:
    void frameRecorded(int);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

private:
    std::shared_ptr<CameraControl> m_camera;
    QVector2D record_res;
    QPointF m_lastPos;
    QTimer* m_pauseRenderDally;
    QTimer* m_wheelEventDally;

public:
    bool simpleRenderChecked;
    bool updateLightOnce;
    bool m_bMovingCamera;
};

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

public slots:
    void updateFrame(const QString& action = "");
    void onRun();
    void onRecord();
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

    ViewportWidget* m_view;
    CameraKeyframeWidget* m_camera_keyframe;
    QTimer* m_pTimer;
    RecordVideoMgr m_recordMgr;
    bool m_bRecordRun;
    static const int m_updateFeq = 16;
    static const int m_sliderFeq = 16;
};

#endif
