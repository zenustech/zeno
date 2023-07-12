#ifndef __CAMERA_CONTROL_H__
#define __CAMERA_CONTROL_H__

//#include <QtOpenGL>
#include <QtWidgets>
#include <viewportinteraction/picker.h>
#include <viewportinteraction/transform.h>

class Zenovis;

class CameraControl : public QObject
{
    Q_OBJECT
public:
    CameraControl(Zenovis* pZenovis,
                  std::shared_ptr<zeno::FakeTransformer> transform,
                  std::shared_ptr<zeno::Picker> picker,
                  QObject* parent = nullptr);
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

    std::shared_ptr<zeno::Picker> m_picker;
    std::shared_ptr<zeno::FakeTransformer> m_transformer;
    Zenovis* m_zenovis;
};


#endif