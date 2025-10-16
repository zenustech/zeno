#pragma once

#ifndef _ZQML_RENDER_H__
#define _ZQML_RENDER_H__

#include <QObject>
#include <QScopedPointer>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include "../viewport/transform.h"
#include "../viewport/picker.h"
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <zeno/core/common.h>
#include <zenovis/Session.h>
#include <zenovis/Camera.h>
#include "layout/winlayoutrw.h"


class Zenovis;
class CameraControl;

class ZQmlRender : public QObject
{
    Q_OBJECT
public:
    explicit ZQmlRender(QSize res, QObject* parent = 0);
    ~ZQmlRender();

    enum CoordinateMirroring {
        DoNotMirrorCoordinates,
        MirrorYCoordinate
    };

    // All assume that the GL context is current.
    void initialize(QOpenGLContext* context, CoordinateMirroring cm = DoNotMirrorCoordinates);
    void render();
    Zenovis* getZenovis() const;
    void fakeMousePressEvent(ViewMouseInfo event_info);
    void fakeMouseReleaseEvent(ViewMouseInfo event_info);
    void fakeMouseMoveEvent(ViewMouseInfo event_info);
    void fakeWheelEvent(ViewMouseInfo event_info);
    zenovis::Session* getSession() const;

signals:
    void requestUpdate();
    void sig_reload_finished();

public slots:
    void reload_objects(const zeno::render_reload_info& info);
    void onMouseEvent(ViewMouseInfo event_info);
    void resize(int nx, int ny);
    void setSafeFrames(bool bLock, int nx, int ny);
    void setSimpleRenderOption();
    void clearTransformer();
    void changeTransformOperationByNode(const QString& node);
    void changeTransformOperation(int mode);
    void cameraLookTo(zenovis::CameraLookToDir dir);
    void setViewWidgetInfo(DockContentWidgetInfo& info);
    void updatePerspective();
    void setNumSamples(int samples);
    void setCameraRes(const QVector2D& res);
    void cleanUpScene();
    void setShowptnum(bool bShow);

private:
    std::shared_ptr<zeno::Picker> m_picker;
    std::shared_ptr<zeno::FakeTransformer> m_fakeTrans;
    Zenovis* m_zenovis;
    CameraControl* m_camera;
};

#endif