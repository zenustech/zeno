#ifndef CAMERADATA_H
#define CAMERADATA_H


#include <QMatrix4x4>
#include <memory>


class CameraData
{
    struct Impl;
    std::unique_ptr<Impl> impl;

public:
    CameraData();
    ~CameraData();

    void zoom(float dy, bool fov_mode);
    void move(float dx, double dy, bool pan_mode);
    void resize(int nx, int ny);

    QMatrix4x4 getView() const;
    QMatrix4x4 getProjection() const;
    float getPointScale() const;
};

#endif // CAMERADATA_H
