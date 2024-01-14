#ifndef __VIEWPORT_WIDGET_H__
#define __VIEWPORT_WIDGET_H__

#include <QtWidgets>
#include <QtOpenGL>
#include "uicommon.h"
#include "recordvideomgr.h"
#include "layout/docktabcontent.h"
#include "layout/winlayoutrw.h"
#include <viewport/transform.h>
#include <viewport/picker.h>

class ZTimeline;
class ZenoMainWindow;
class Zenovis;
class ViewportWidget;
class Picker;
class CameraControl;

class ViewportWidget : public QGLWidget
{
    Q_OBJECT
    typedef QGLWidget _base;
public:
    ViewportWidget(QWidget* parent = nullptr);
    ~ViewportWidget();
    void testCleanUp();
    void initializeGL() override;
    void resizeGL(int nx, int ny) override;
    void paintGL() override;
    QVector2D cameraRes() const;
    Zenovis* getZenoVis() const;
    std::shared_ptr<zeno::Picker> picker() const;
    std::shared_ptr<zeno::FakeTransformer> fakeTransformer() const;
    zenovis::Session* getSession() const;
    bool isPlaying() const;
    void startPlay(bool bPlaying);
    void setCameraRes(const QVector2D& res);
    void setSafeFrames(bool bLock, int nx, int ny);
    void updatePerspective();
    void updateCameraProp(float aperture, float disPlane);
    void cameraLookTo(int dir);
    void clearTransformer();
    void changeTransformOperation(const QString& node);
    void changeTransformOperation(int mode);
    void changeTransformCoordSys();
    void cleanUpScene();
    void setNumSamples(int samples);
    void setPickTarget(const std::string& prim_name);
    void bindNodeToPicker(const QModelIndex& node, const QModelIndex& subgraph, const std::string& sock_name);
    void unbindNodeFromPicker();
    void setSimpleRenderOption();
    void setViewWidgetInfo(DockContentWidgetInfo& info);
    void glDrawForCommandLine();

signals:
    void frameRecorded(int);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void enterEvent(QEvent * event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

private:
    CameraControl* m_camera;
    Zenovis* m_zenovis;
    QVector2D record_res;
    QPointF m_lastPos;
    QTimer* m_pauseRenderDally;
    QTimer* m_wheelEventDally;
    std::shared_ptr<zeno::Picker> m_picker;
    std::shared_ptr<zeno::FakeTransformer> m_fakeTrans;

    std::tuple<int, int, bool, double, double, double>viewInfo{ 0, 0, true, 0, 0, 0 };
    bool loadSettingFromZsg = false;

public:
    int simpleRenderTime;
    bool updateLightOnce;
    bool m_bMovingCamera;
};

#endif
