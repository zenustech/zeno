#include "cameradata.h"
#include <QVector3D>
#include <cmath>

CameraData::CameraData()
    : impl(std::make_unique<Impl>())
{
}

CameraData::~CameraData() = default;

struct CameraData::Impl
{
    QMatrix4x4 view;
    QMatrix4x4 proj;
    float point_scale = 1.f;

    int nx = 512, ny = 512;
    QVector3D center;
    float theta = 0.f;
    float phi = 0.f;
    float radius = 4.f;
    float fov = 30.f;
    bool ortho_mode = false;

    Impl() { updateMatrices(); }

    void updateMatrices() {
        float fov_rad = fov * M_PI / 180.f;
        point_scale = ny / (50.f * std::tan(fov_rad * 0.5f));
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        float cos_p = std::cos(phi);
        float sin_p = std::sin(phi);
        QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        float aspect = (float)nx / ny;

        view.setToIdentity();
        proj.setToIdentity();
        if (ortho_mode) {
            view.lookAt(center - back, center, up);
            proj.ortho(-radius * aspect, radius * aspect, -radius, radius, -100.0f, 100.0f);
        } else {
            view.lookAt(center - back * radius, center, up);
            proj.perspective(fov_rad, aspect, 0.05f, 20000.f);
        }
    }
};

void CameraData::zoom(float dy, bool fov_mode)
{
    auto scale = std::pow(0.89f, dy);
    if (fov_mode)
        impl->fov /= scale;
    impl->radius *= scale;
    impl->updateMatrices();
}

void CameraData::move(float dx, double dy, bool pan_mode)
{
    if (pan_mode) {
        float cos_t = std::cos(impl->theta);
        float sin_t = std::sin(impl->theta);
        float cos_p = std::cos(impl->phi);
        float sin_p = std::sin(impl->phi);
        QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        QVector3D right = QVector3D::crossProduct(up, back).normalized();
        up = QVector3D::crossProduct(right, back).normalized();
        QVector3D delta = right * dx + up * dy;
        impl->center += delta * impl->radius;
    } else {
        impl->theta += dy * M_PI;
        impl->phi += dx * M_PI;
    }
    impl->updateMatrices();
}

QMatrix4x4 CameraData::getView() const
{
    return impl->view;
}

QMatrix4x4 CameraData::getProjection() const
{
    return impl->proj;
}

float CameraData::getPointScale() const
{
    return impl->point_scale;
}
