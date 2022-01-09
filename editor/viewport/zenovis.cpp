#include "zenovis.h"
#include "../zenvis/zenvis.h"
#include "camerakeyframe.h"
#include "viewportwidget.h"
#include "../launch/corelaunch.h"


Zenvis::Zenvis()
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

Zenvis& Zenvis::GetInstance()
{
    static Zenvis instance;
    return instance;
}

void Zenvis::initializeGL()
{
    zenvis::initialize();
}

void Zenvis::paintGL()
{
    _frameUpdate();
    _uploadStatus();
    zenvis::new_frame();
    _recieveStatus();
}

void Zenvis::recordGL(const std::string& record_path)
{
    zenvis::set_window_size(m_resolution[0], m_resolution[1]);
    zenvis::look_perspective(m_perspective.cx, m_perspective.cy, m_perspective.cz, 
        m_perspective.theta, m_perspective.phi, m_perspective.radius, m_perspective.fov, m_perspective.ortho_mode);
    zenvis::new_frame_offline(record_path);
}

int Zenvis::getCurrentFrameId()
{
    return zenvis::get_curr_frameid();
}

void Zenvis::startPlay(bool bPlaying)
{
    m_playing = bPlaying;
}

int Zenvis::setCurrentFrameId(int frameid)
{
    if (frameid < 0)
        frameid = 0;
    int nFrames = getFrameCount();
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

void Zenvis::_uploadStatus()
{
    zenvis::set_window_size(m_resolution[0], m_resolution[1]);
    zenvis::look_perspective(m_perspective.cx, m_perspective.cy, m_perspective.cz, m_perspective.theta,
        m_perspective.phi, m_perspective.radius, m_perspective.fov, false);
}

void Zenvis::_recieveStatus()
{
    int frameid = zenvis::get_curr_frameid();
    double solver_interval = zenvis::get_solver_interval();
    double render_fps = zenvis::get_render_fps();
    m_solver_interval = solver_interval;
    m_render_fps = render_fps;
}

void Zenvis::_frameUpdate()
{
    //if fileio.isIOPathChanged() :
    //    core.clear_graphics()
    int frameid = getCurrentFrameId();
    if (m_playing)
        frameid += 1;
    frameid = setCurrentFrameId(frameid);
    zenvis::auto_gc_frame_data(m_cache_frames);
    zenvis::set_show_grid(m_show_grid);

    auto frame_files = getFrameFiles(frameid);
    if (m_frame_files != frame_files)
    {
        foreach(auto frame_file, frame_files)
        {
            QString name = std::get<0>(frame_file);
            QString ext = std::get<1>(frame_file);
            QString path = std::get<2>(frame_file);

            std::string sName = name.toStdString();
            std::string sExt = ext.toStdString();
            std::string sPath = path.toStdString();
            zenvis::load_file(sName, sExt, sPath, frameid);
        }
    }
    m_frame_files = frame_files;
}

QList<Zenvis::FRAME_FILE> Zenvis::getFrameFiles(int frameid)
{
    QList<Zenvis::FRAME_FILE> framefiles;
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
}

int Zenvis::getFrameCount(int* max_frameid)
{
    if (g_iopath.isEmpty())
        return 0;

    int frameid = 0;
    while (!max_frameid || frameid < *max_frameid)
    {
        QString dirPath = QString("%1/%2").arg(g_iopath).arg(QString::number(frameid), 6, QLatin1Char('0'));
        QString lockfile = QString("%1/done.lock").arg(dirPath);
        if (!QDir(dirPath).exists("done.lock"))
            return frameid;
        frameid += 1;
    }
    return *max_frameid;
}