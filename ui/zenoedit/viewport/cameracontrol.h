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

    float getRoll() const;
    void setRoll(float roll);
    float getTheta() const;
    void setTheta(float theta);
    float getPhi() const;
    void setPhi(float phi);
    zeno::vec3f getCenter() const;
    void setCenter(zeno::vec3f center);
    bool getOrthoMode() const;
    void setOrthoMode(bool OrthoMode);
    float getRadius() const;
    void setRadius(float radius);
    float getFOV() const;
    void setFOV(float fov);
    float getAperture() const;
    void setAperture(float aperture);
    float getDisPlane() const;
    void setDisPlane(float disPlane);
    void updatePerspective();
    void setKeyFrame();

    bool fakeKeyPressEvent(int uKey);
    bool fakeKeyReleaseEvent(int uKey);
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
    QPointF m_lastMidButtonPos;
    QPoint m_boundRectStartPos;
    QVector2D m_res;
    QSet<int> m_pressedKeys;

    std::shared_ptr<zeno::Picker> m_picker;
    std::shared_ptr<zeno::FakeTransformer> m_transformer;
    Zenovis* m_zenovis;

    bool middle_button_pressed = false;
};


#endif