#ifndef __CAMERA_CONTROL_H__
#define __CAMERA_CONTROL_H__

//#include <QtOpenGL>
#include <QtWidgets>
#include <viewportinteraction/picker.h>
#include <viewportinteraction/transform.h>
#include <zenovis/Camera.h>

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

    glm::vec3 getPos() const;
    void setPos(glm::vec3 value);
    glm::vec3 getPivot() const;
    void setPivot(glm::vec3 value);
    glm::quat getRotation();
    void setRotation(glm::quat value);
    zeno::vec3f getCenter() const;
    void setCenter(zeno::vec3f center);
    bool getOrthoMode() const;
    void setOrthoMode(bool OrthoMode);
    float getRadius() const;
    float getFOV() const;
    void setFOV(float fov);
    float getAperture() const;
    void setAperture(float aperture);
    float getDisPlane() const;
    void setDisPlane(float disPlane);
    void updatePerspective();

    bool fakeKeyPressEvent(int uKey);
    bool fakeKeyReleaseEvent(int uKey);
    void fakeMousePressEvent(QMouseEvent* event);
    void fakeMouseReleaseEvent(QMouseEvent* event);
    void fakeMouseMoveEvent(QMouseEvent* event);
    void fakeWheelEvent(QWheelEvent* event);
    void fakeMouseDoubleClickEvent(QMouseEvent* event);
    void focus(QVector3D center, float radius);
    QVector3D realPos() const;
    QVector3D screenToWorldRay(float x, float y);
    QVariant hitOnFloor(float x, float y);
    void lookTo(zenovis::CameraLookToDir dir);
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
    std::optional<glm::vec3> hit_posWS;

    std::weak_ptr<zeno::Picker> m_picker;
    std::weak_ptr<zeno::FakeTransformer> m_transformer;
    Zenovis* m_zenovis;

    bool middle_button_pressed = false;
};


#endif