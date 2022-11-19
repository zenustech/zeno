#include "recordvideomgr.h"
#include <viewport/zenovis.h>
#include <viewport/viewportwidget.h>
#include <zenovis/DrawOptions.h>
#include <zeno/utils/format.h>
#include <zeno/utils/log.h>
#include <util/log.h>


RecordVideoMgr::RecordVideoMgr(ViewportWidget* view, const VideoRecInfo& record, QObject* parent)
    : QObject(parent)
    , m_recordInfo(record)
    , m_view(view)
    , m_timer(nullptr)
{
    m_timer = new QTimer;
    m_currFrame = m_recordInfo.frameRange.first;

    //create directory to store screenshot pngs.
    QDir dir(record.record_path);
    ZASSERT_EXIT(dir.exists());
    dir.mkdir("P");

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

        if (m_recordInfo.saveAsImageSequence) {
            emit recordFinished();
            return;
        }

        {
            QString dir_path = m_recordInfo.record_path + "/P/";
            QDir qDir = QDir(dir_path);
            qDir.setNameFilters(QStringList("*.png"));
            QStringList fileList = qDir.entryList(QDir::Files);
            fileList.removeOne(".");
            fileList.removeOne("..");
            fileList.sort();
            for (auto i = 0; i < fileList.size(); i++) {
                auto new_name = dir_path + zeno::format("{:06d}.png", i).c_str();
                auto old_name = dir_path + fileList.at(i);
                QFile::rename(old_name,new_name);
            }
        }

        QString path = m_recordInfo.record_path + "/P/%06d.png";
        QString outPath = m_recordInfo.record_path + "/" + m_recordInfo.videoname;
        QStringList cmd = { "ffmpeg", "-y", "-r", QString::number(m_recordInfo.fps), "-i", path, "-c:v", "mpeg4", "-b:v", QString::number(m_recordInfo.bitrate) + "k", outPath};

        //zeno::log_info("record cmd {}", cmd.join(" ").toStdString());

        int ret = QProcess::execute(cmd.join(" "));
        if (ret == 0)
        {
            emit recordFinished();
        }
        else
        {
            //todo get the error string from QProcess.
            emit recordFailed(QString());
        }
        return;
    }

    auto& inst = Zenovis::GetInstance();

    inst.setCurrentFrameId(m_currFrame);
    inst.paintGL();

    auto record_file = zeno::format("{}/P/{:06d}.png", m_recordInfo.record_path.toStdString(), m_currFrame);

    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    auto old_num_samples = scene->drawOptions->num_samples;
    scene->drawOptions->num_samples = m_recordInfo.numOptix;
    scene->drawOptions->msaa_samples = m_recordInfo.numMSAA;

    auto [x, y] = Zenovis::GetInstance().getSession()->get_window_size();

    auto extname = QFileInfo(QString::fromStdString(record_file)).suffix().toStdString();
    Zenovis::GetInstance().getSession()->set_window_size( (int)m_recordInfo.res.x(), (int)m_recordInfo.res.y());
    Zenovis::GetInstance().getSession()->do_screenshot(record_file, extname);
    Zenovis::GetInstance().getSession()->set_window_size(x, y);
    scene->drawOptions->num_samples = old_num_samples;

    m_pics.append(QString::fromStdString(record_file));

    emit frameFinished(m_currFrame);

    m_currFrame++;
}
