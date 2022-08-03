#ifndef __VIEWPORT_WIDGET_H__
#define __VIEWPORT_WIDGET_H__

#include <QtWidgets>
#include <QtOpenGL>
#include "comctrl/zmenubar.h"
#include "comctrl/zmenu.h"

class ZTimeline;
class ZenoMainWindow;

#if 0
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
#endif

struct VideoRecInfo
{
    QString record_path;
    QString videoname;
    QVector2D res;
    QPair<int, int> frameRange;
    int fps;
    int bitrate;
    VideoRecInfo() {
        res = { 0,0 };
        fps = bitrate = 0;
        frameRange = { -1, -1 };
    }
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
    void fakeMouseReleaseEvent(QMouseEvent* event);
    void fakeMouseMoveEvent(QMouseEvent* event);
    void fakeWheelEvent(QWheelEvent* event);
    void focus(QVector3D center, float radius);
    QVector3D realPos() const;
    QVector3D screenToWorldRay(float x, float y) const;
    QVariant hitOnFloor(float x, float y) const;


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
    QVector2D m_res;
};

class ViewportWidget : public QOpenGLWidget
{
    Q_OBJECT
    typedef QOpenGLWidget _base;
public:
    ViewportWidget(QWidget* parent = nullptr);
    ~ViewportWidget();
    void initializeGL() override;
    void resizeGL(int nx, int ny) override;
    void paintGL() override;
    void checkRecord(std::string a_record_file, QVector2D a_record_res, int a_nsamples);
    QVector2D cameraRes() const;
    void setCameraRes(const QVector2D& res);
    void updatePerspective();

signals:
    void frameRecorded(int);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

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
    DisplayWidget(ZenoMainWindow* parent = nullptr);
    ~DisplayWidget();
    void init();
    QSize sizeHint() const override;

public slots:
    void updateFrame(const QString& action = "");
    void onRun();
    void onRecord();
    void onKill();
    void onModelDataChanged();
    void onPlayClicked(bool);
    void onSliderValueChanged(int);

signals:
    void frameUpdated(int new_frame);

private:
    bool isOptxRendering() const;

    ViewportWidget* m_view;
    ZTimeline* m_timeline;
    ZenoMainWindow* m_mainWin;
    CameraKeyframeWidget* m_camera_keyframe;
    QTimer* m_pTimer;
    QThread m_recThread;
    static const int m_updateFeq = 16;
    static const int m_sliderFeq = 16;
};

#endif
