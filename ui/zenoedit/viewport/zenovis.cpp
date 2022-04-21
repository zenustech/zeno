#include "camerakeyframe.h"
#include "viewportwidget.h"
#include "../launch/corelaunch.h"
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/logger.h>
#include <zeno/zeno.h>


Zenovis::Zenovis()
    : m_solver_frameid(0)
    , m_solver_interval(0)
    , m_render_fps(0)
    , m_resolution(QPoint(1,1))
    , m_cache_frames(10)
    , m_playing(false)
    , m_camera_keyframe(nullptr)
{
}

Zenovis& Zenovis::GetInstance()
{
    static Zenovis instance;
    return instance;
}

void Zenovis::loadGLAPI(void *procaddr)
{
    zenovis::loadGLAPI(procaddr);
}

void Zenovis::initializeGL()
{
    session = std::make_unique<zenovis::Session>();
}

void Zenovis::paintGL()
{
    doFrameUpdate();
    session->new_frame();
}

void Zenovis::recordGL(const std::string& record_path)
{
    session->new_frame_offline(record_path);
}

int Zenovis::getCurrentFrameId()
{
    return session->get_curr_frameid();
}

void Zenovis::updatePerspective(QVector2D const &resolution, PerspectiveInfo const &perspective)
{
    m_resolution = resolution;
    m_perspective = perspective;
    if (session) {
        session->set_window_size(m_resolution.x(), resolution.y());
        session->look_perspective(m_perspective.cx, m_perspective.cy, m_perspective.cz,
                                  m_perspective.theta, m_perspective.phi, m_perspective.radius,
                                  m_perspective.fov, m_perspective.ortho_mode);
    }
    emit perspectiveUpdated(perspective);
}

void Zenovis::startPlay(bool bPlaying)
{
    m_playing = bPlaying;
}

zenovis::Session *Zenovis::getSession() const
{
    return session.get();
}

int Zenovis::setCurrentFrameId(int frameid)
{
    if (frameid < 0)
        frameid = 0;
    int nFrames = zeno::getSession().globalComm->maxPlayFrames();
    if (frameid >= nFrames)
        frameid = std::max(0, nFrames - 1);
    zeno::log_trace("now frame {}/{}", frameid, nFrames);
    int old_frameid = session->get_curr_frameid();
    session->set_curr_frameid(frameid);
    if (old_frameid != frameid) {
        if (m_camera_keyframe && m_camera_control) {
            PerspectiveInfo r;
            if (m_camera_keyframe->queryFrame(frameid, r)) {
                m_camera_control->setKeyFrame();
                m_camera_control->updatePerspective();
            }
        }
        emit frameUpdated(frameid);
    }
    return frameid;
}

void Zenovis::doFrameUpdate()
{
    //if fileio.isIOPathChanged() :
    //    core.clear_graphics()
    int frameid = getCurrentFrameId();
    if (m_playing) {
        zeno::log_trace("playing at frame {}", frameid);
        frameid += 1;
    }
    frameid = setCurrentFrameId(frameid);
    //zenvis::auto_gc_frame_data(m_cache_frames);

    auto viewObjs = zeno::getSession().globalComm->getViewObjects(frameid);

    zeno::log_trace("_frameUpdate: {} objects at frame {}", viewObjs.size(), frameid);
    session->load_objects(viewObjs);
}

/*
QList<Zenovis::FRAME_FILE> Zenovis::getFrameFiles(int frameid)
{
    QList<Zenovis::FRAME_FILE> framefiles;
    if (g_iopath.isEmpty())
        return framefiles;

    QString dirPath = QString("%1/%2").arg(g_iopath).arg(QString::number(frameid), 6, QLatin1Char('0'));
    QDir frameDir(dirPath);
    if (!frameDir.exists("done.lock"))
        return framefiles;

    frameDir.setFilter(QDir::Files);

    foreach(QFileInfo fileInfo, frameDir.entryInfoList())
    {
        QString fn = fileInfo.fileName();
        QString path = QString("%1/%2").arg(dirPath).arg(fn);
        QString ext = QString(".") + fileInfo.suffix();
        framefiles.append(FRAME_FILE(fn, ext, path));
    }
    return framefiles;
}*/
