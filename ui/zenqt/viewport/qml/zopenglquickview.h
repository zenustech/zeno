#pragma once

#ifndef _ZOPENGL_QUICKVIEW_H__
#define _ZOPENGL_QUICKVIEW_H__

#include <QtWidgets>
#include <QQuickView>
#include "zqmlrender.h"
#include <queue>
#include <mutex>

class ZQmlRender;
class Camera;

Q_DECLARE_METATYPE(ViewMouseInfo);

class ZOpenGLQuickView : public QQuickView
{
    Q_OBJECT
public:
    explicit ZOpenGLQuickView(QWindow* parent = 0);
    void reload_objects(const zeno::render_reload_info& info);

//TODO:
    void cleanUpView() {}
    void testCleanUp() {}
    void cleanUpScene();
    Zenovis* getZenoVis();
    zenovis::Session* getSession() const { return m_renderer->getSession(); }
    void resizeGL(int nx, int ny);
    void updateGL() { update(); }
    std::shared_ptr<zeno::Picker> picker() const { return nullptr; }
    void updateCameraProp(float aperture, float disPlane) {

    }
    void updatePerspective();
    void setNumSamples(int samples);
    void setCameraRes(const QVector2D& res);
    void setSafeFrames(bool bLock, int nx, int ny);
    void setShowPtnum(bool bShow);
    bool isCameraMoving() const;
    void setSimpleRenderOption();
    void clearTransformer();
    void changeTransformOperation(const QString& node);
    void changeTransformOperation(int mode);
    void glDrawForCommandLine() {

    }
    void cameraLookTo(zenovis::CameraLookToDir dir);
    void setViewWidgetInfo(DockContentWidgetInfo& info);

signals:
    void sig_Reload(const zeno::render_reload_info&);
    void sig_render_reload_finished();
    void sig_MouseEvent(ViewMouseInfo);
    void sig_Resize(int, int);
    void sig_cleanScene();
    void sig_UpdatePerspective();
    void sig_SetNumSamples(int);
    void sig_setCameraRes(const QVector2D&);
    void sig_setSafeFrames(bool, int, int);
    void sig_setSimpleRenderOption();
    void sig_clearTransformer();
    void sig_changeTransformOperationByNode(const QString&);
    void sig_changeTransformOperation(int);
    void sig_cameraLookTo(zenovis::CameraLookToDir);
    void sig_showptnum(bool);

public slots:
    void onUpdateRequest();

protected:
    void resizeEvent(QResizeEvent*) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void initializeUnderlay();
    void synchronizeUnderlay();
    void renderUnderlay();
    void invalidateUnderlay();

    QScopedPointer<ZQmlRender> m_renderer;
};

#endif