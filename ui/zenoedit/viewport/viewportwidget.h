#ifndef __VIEWPORT_WIDGET_H__
#define __VIEWPORT_WIDGET_H__

#include <QtWidgets>
#include <QtOpenGL>
#include "comctrl/zmenubar.h"
#include "comctrl/zmenu.h"

class ZTimeline;

class QDMDisplayMenu : public ZMenu
{
public:
    QDMDisplayMenu();
};

class QDMRecordMenu : public ZMenu
{
public:
    QDMRecordMenu();
};

class CameraControl : public QWidget
{
    Q_OBJECT
public:
    CameraControl(QWidget* parent = nullptr);
    void setRes(QVector2D res);
    QVector2D res() const { return m_res; }
    void updatePerspective();
    void setKeyFrame();

    void fakeMousePressEvent(QMouseEvent* event);
    void fakeMouseMoveEvent(QMouseEvent* event);
    void fakeWheelEvent(QWheelEvent* event);

private:
    bool m_mmb_pressed;
    float m_theta;
    float m_phi;
    QPointF m_lastPos;
    QVector3D  m_center;
    bool m_ortho_mode;
    float m_fov;
    float m_radius;
    QVector2D m_res;
};

class ViewportWidget : public QOpenGLWidget
{
    typedef QOpenGLWidget _base;
public:
    ViewportWidget(QWidget* parent = nullptr);
    ~ViewportWidget();
    void initializeGL();
    void resizeGL(int nx, int ny) override;
    void paintGL() override;
    void checkRecord();

protected:
    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void wheelEvent(QWheelEvent* event);

private:
    std::shared_ptr<CameraControl> m_camera;
    std::string record_path;
    QVector2D record_res;
};

class CameraKeyframeWidget;

class DisplayWidget : public QWidget
{
    Q_OBJECT
public:
    DisplayWidget(QWidget* parent = nullptr);
    ~DisplayWidget();
    void init();
    QSize sizeHint() const;

public slots:
    void updateFrame();
    void onRunClicked(int nFrames);

signals:
    void frameUpdated(int new_frame);

private:
    ViewportWidget* m_view;
    ZTimeline* m_timeline;
    CameraKeyframeWidget* m_camera_keyframe;
};

#endif
