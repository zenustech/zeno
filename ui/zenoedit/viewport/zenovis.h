#ifndef __ZENOVIS_H__
#define __ZENOVIS_H__

#include <QtWidgets>
#include <zenovis/Session.h>

class CameraControl;
class CameraKeyframeWidget;

struct PerspectiveInfo
{
    double cx;
    double cy;
    double cz;
    double theta;
    double phi;
    double radius;
    double fov;
    bool ortho_mode;
    double aperture;
    double focalPlaneDistance;
    PerspectiveInfo() : cx(0), cy(0), cz(0), theta(0), phi(0), radius(0), fov(0), ortho_mode(0), aperture(0), focalPlaneDistance(0) {}
    PerspectiveInfo(double cx, double cy, double cz, double theta, double phi, double radius, double fov, bool ortho_mode, double aperture, double focalPlaneDistance)
        : cx(cx), cy(cy), cz(cz), theta(theta), phi(phi), radius(radius), fov(fov), ortho_mode(ortho_mode), aperture(aperture), focalPlaneDistance(focalPlaneDistance) {}
};

class Zenovis : public QObject
{
    Q_OBJECT
    typedef std::tuple<QString, QString, QString> FRAME_FILE;
    std::unique_ptr<zenovis::Session> session;

signals:
    void frameUpdated(int);
    void objectsUpdated(int);
    void frameDrawn(int);
    void perspectiveUpdated(PerspectiveInfo);

public slots:
    void startPlay(bool bPlaying);
    int setCurrentFrameId(int frameid);

public:
    Zenovis(QObject* parent = nullptr);
    void loadGLAPI(void *procaddr);
    void initializeGL();
    void paintGL();
    //void recordGL(const std::string& record_path);
    int getCurrentFrameId();
    bool isPlaying() const;
    zenovis::Session *getSession() const;
    void updatePerspective(QVector2D const &resolution, PerspectiveInfo const &perspective);
    void updateCameraFront(QVector3D center, QVector3D front, QVector3D up);

//private:
    void doFrameUpdate();

    //static QString sIoPath;

    int m_solver_frameid;
    int m_solver_interval;
    int m_render_fps;
    QVector2D m_resolution;
    PerspectiveInfo m_perspective;
    int m_cache_frames;
    bool m_playing;
    CameraKeyframeWidget* m_camera_keyframe{};
    CameraControl* m_camera_control{};
    //QList<FRAME_FILE> m_frame_files;

    //bool m_bAddPoint = false;
};

#endif
