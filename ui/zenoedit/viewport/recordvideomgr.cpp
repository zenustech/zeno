#include "recordvideomgr.h"
#include <viewport/zenovis.h>
#include <viewport/viewportwidget.h>
#include <zeno/utils/format.h>
#include <zeno/utils/log.h>


RecordVideoMgr::RecordVideoMgr(ViewportWidget* view, const VideoRecInfo& record, QObject* parent)
    : QObject(parent)
    , m_recordInfo(record)
    , m_view(view)
    , m_timer(nullptr)
{
    m_timer = new QTimer;
    m_currFrame = m_recordInfo.frameRange.first;

    connect(m_timer, SIGNAL(timeout()), this, SLOT(recordFrame()));
    m_timer->start(0);
    Zenovis::GetInstance().blockSignals(true);
}

RecordVideoMgr::~RecordVideoMgr()
{
    cancelRecord();
}

void RecordVideoMgr::cancelRecord()
{
    m_timer->stop();
    Zenovis::GetInstance().blockSignals(false);
}

void RecordVideoMgr::recordFrame()
{
    if (m_currFrame > m_recordInfo.frameRange.second)
    {
        m_timer->stop();
        Zenovis::GetInstance().blockSignals(false);
        emit recordFinished();
        return;
    }

    auto& inst = Zenovis::GetInstance();

    inst.setCurrentFrameId(m_currFrame);
    inst.paintGL();

    auto record_file = zeno::format("{}/{:06d}.png", m_recordInfo.record_path.toStdString(), m_currFrame);
    int nsamples = 16;

    QVector2D oldRes = m_view->cameraRes();
    m_view->setCameraRes(m_recordInfo.res);
    m_view->updatePerspective();

    auto extname = QFileInfo(QString::fromStdString(record_file)).suffix().toStdString();
    Zenovis::GetInstance().getSession()->do_screenshot(record_file, extname, nsamples);
    m_view->setCameraRes(oldRes);
    m_view->updatePerspective();

    emit frameFinished(m_currFrame);

    m_currFrame++;
}