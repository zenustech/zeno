#include "zenovis.h"
#include "../zenovis/zenvis.h"
#include "camerakeyframe.h"
#include "viewportwidget.h"
#include "../launch/corelaunch.h"
#include <zeno/extra/GlobalState.h>


Zenovis::Zenovis()
    : m_solver_frameid(0)
    , m_solver_interval(0)
    , m_render_fps(0)
    , m_resolution(QPoint(1,1))
    , m_cache_frames(10)
    , m_show_grid(true)
    , m_playing(false)
    , m_camera_keyframe(nullptr)
{
}

Zenovis& Zenovis::GetInstance()
{
    static Zenovis instance;
    return instance;
}

void Zenovis::initializeGL()
{
    zenvis::initialize();
}

void Zenovis::paintGL()
{
    _frameUpdate();
    _uploadStatus();
    zenvis::new_frame();
    _recieveStatus();
}

void Zenovis::recordGL(const std::string& record_path)
{
    zenvis::set_window_size(m_resolution[0], m_resolution[1]);
    zenvis::look_perspective(m_perspective.cx, m_perspective.cy, m_perspective.cz, 
        m_perspective.theta, m_perspective.phi, m_perspective.radius, m_perspective.fov, m_perspective.ortho_mode);
    zenvis::new_frame_offline(record_path);
}

int Zenovis::getCurrentFrameId()
{
    return zenvis::get_curr_frameid();
}

void Zenovis::startPlay(bool bPlaying)
{
    m_playing = bPlaying;
}

int Zenovis::setCurrentFrameId(int frameid)
{
    if (frameid < 0)
        frameid = 0;
    int nFrames = zeno::state.countFrames();
    if (frameid >= nFrames)
        frameid = nFrames - 1;
    int cur_frameid = zenvis::get_curr_frameid();
    zenvis::set_curr_frameid(frameid);
    if (cur_frameid != frameid && m_camera_keyframe && m_camera_control)
    {
        PerspectiveInfo r;
        bool ret = m_camera_keyframe->queryFrame(frameid, r);
        if (ret)
        {
            m_camera_control->setKeyFrame();
            m_camera_control->updatePerspective();
            emit frameUpdated(frameid);
        }
    }
    return frameid;
}

void Zenovis::_uploadStatus()
{
    zenvis::set_window_size(m_resolution[0], m_resolution[1]);
    zenvis::look_perspective(m_perspective.cx, m_perspective.cy, m_perspective.cz, m_perspective.theta,
        m_perspective.phi, m_perspective.radius, m_perspective.fov, false);
}

void Zenovis::_recieveStatus()
{
    int frameid = zenvis::get_curr_frameid();
    double solver_interval = zenvis::get_solver_interval();
    double render_fps = zenvis::get_render_fps();
    m_solver_interval = solver_interval;
    m_render_fps = render_fps;
}

void Zenovis::_frameUpdate()
{
    //if fileio.isIOPathChanged() :
    //    core.clear_graphics()
    int frameid = getCurrentFrameId();
    if (m_playing)
        frameid += 1;
    frameid = setCurrentFrameId(frameid);
    zenvis::auto_gc_frame_data(m_cache_frames);
    zenvis::set_show_grid(m_show_grid);

    auto viewObjs = zeno::state.getViewObjects(frameid);

    for (auto const &obj: viewObjs) {
        zenvis::load_object(obj, frameid);
    }
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
