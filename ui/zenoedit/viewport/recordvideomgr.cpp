#include "recordvideomgr.h"
#include <viewport/zenovis.h>
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "viewport/optixviewport.h"
#include "viewport/zoptixviewport.h"
#include <zenovis/DrawOptions.h>
#include <zeno/utils/format.h>
#include <zeno/utils/log.h>
#include <util/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zenoedit/zenomainwindow.h>
#include <zeno/types/HeatmapObject.h>
#include "launch/corelaunch.h"
#include <zeno/extra/GlobalStatus.h>
#include <zeno/core/Session.h>
#include <filesystem>


RecordVideoMgr::RecordVideoMgr(QObject* parent)
    : QObject(parent)
    , m_currFrame(0)
{
}

RecordVideoMgr::~RecordVideoMgr()
{
    cancelRecord();
}

Zenovis* RecordVideoMgr::getZenovis()
{
    DisplayWidget* pWid =  qobject_cast<DisplayWidget *>(parent());
    ZASSERT_EXIT(pWid, nullptr);
    return pWid->getZenoVis();
}

void RecordVideoMgr::cancelRecord()
{
    disconnectSignal();

    DisplayWidget* pWid = qobject_cast<DisplayWidget*>(parent());
    if (!pWid)
        return;

    if (!pWid->isGLViewport())
    {
        auto pView = pWid->optixViewport();
        ZASSERT_EXIT(pView);
        pView->cancelRecording(m_recordInfo);
    }
    else
    {
        ZenoMainWindow *mainWin = zenoApp->getMainWindow();
        if (mainWin)
            mainWin->toggleTimelinePlay(false);
    }
}

void RecordVideoMgr::initRecordInfo(const VideoRecInfo& recInfo)
{
    m_recordInfo = recInfo;
    m_currFrame = m_recordInfo.frameRange.first;
}

void RecordVideoMgr::setRecordInfo(const VideoRecInfo& recInfo)
{
    m_recordInfo = recInfo;
    m_currFrame = m_recordInfo.frameRange.first;

    //create directory to store screenshot pngs.
    QDir dir(m_recordInfo.record_path);
    ZASSERT_EXIT(dir.exists());
    dir.mkdir("P");
    // remove old image
    if (m_recordInfo.bExportVideo) {
        QString dir_path = m_recordInfo.record_path + "/P/";
        QDir qDir = QDir(dir_path);
        qDir.setNameFilters(QStringList("*.jpg"));
        QStringList fileList = qDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
        for (auto i = 0; i < fileList.size(); i++) {
//            qDir.remove(fileList.at(i));
        }
    }

    DisplayWidget *pWid = qobject_cast<DisplayWidget *>(parent());
    ZASSERT_EXIT(pWid);
    if (!pWid->isGLViewport())
    {
        //we can only record on another thread, the optix worker thread.
        auto pView = pWid->optixViewport();
        ZASSERT_EXIT(pView);
        bool ret = connect(pView, SIGNAL(sig_frameRecordFinished(int)), this, SIGNAL(frameFinished(int)));
        ret = connect(pView, SIGNAL(sig_recordFinished()), this, SLOT(endRecToExportVideo()));
    }
    else
    {
        Zenovis *pVis = getZenovis();
        bool ret = connect(pVis, SIGNAL(frameDrawn(int)), this, SLOT(onFrameDrawn(int)));
        ZASSERT_EXIT(ret);
    }
}

VideoRecInfo RecordVideoMgr::getRecordInfo() const
{
    return m_recordInfo;
}

REC_RETURN_CODE RecordVideoMgr::endRecToExportVideo()
{
    if (m_recordInfo.bExportEXR) {
        emit recordFinished(m_recordInfo.record_path);
        return REC_NOERROR;
    }
    // denoising
    if (m_recordInfo.needDenoise) {
        QString dir_path = m_recordInfo.record_path + "/P/";
        QDir qDir = QDir(dir_path);
        qDir.setNameFilters(QStringList("*.jpg"));
        QStringList fileList = qDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
        fileList.sort();
        for (auto i = 0; i < fileList.size(); i++) {
            auto jpg_path = (dir_path + fileList.at(i)).toStdString();
            auto pfm_path = jpg_path + ".pfm";
            auto pfm_dn_path = jpg_path + ".dn.pfm";
            // jpg to pfm
            {
                auto image = zeno::readImageFile(jpg_path);
                std::string native_pfm_path = std::filesystem::u8path(pfm_path).string();
                write_pfm(native_pfm_path, image);
            }

            const auto albedo_pfm_path = jpg_path + ".albedo.pfm";
            const auto normal_pfm_path = jpg_path + ".normal.pfm";

            std::string auxiliaryParams; 
            std::vector<std::function<void(void)>> auxiliaryTasks;

            QFile fileAlbedo(QString::fromStdString(albedo_pfm_path));
            if (fileAlbedo.exists()) {
                auxiliaryParams += " --alb " + albedo_pfm_path;
                auxiliaryTasks.push_back([&]() {
                    fileAlbedo.remove();
                });
            }
            QFile fileNormal(QString::fromStdString(normal_pfm_path));
            if (fileNormal.exists()) {
                auxiliaryParams += " --nrm " + normal_pfm_path;
                auxiliaryTasks.push_back([&]() {
                    fileNormal.remove();
                });
            }

            // cmd
            {
                QString cmd = QString("oidnDenoise --ldr %1 -o %2 %3").arg(QString::fromStdString(pfm_path))
                        .arg(QString::fromStdString(pfm_dn_path)).arg(QString::fromStdString(auxiliaryParams));
                qDebug() << cmd;

                int ret = QProcess::execute(cmd);
                qDebug() << ret;
                // pfm to jpg
                if (ret == 0) {
                    std::string native_pfm_dn_path = std::filesystem::u8path(pfm_dn_path).string();
                    auto image = zeno::readPFMFile(native_pfm_dn_path);
                    {
                        std::string native_jpg_path = std::filesystem::u8path(jpg_path).string();
                        write_jpg(native_jpg_path, image);
                    }
                }
                for (auto& task : auxiliaryTasks) {
                    task();
                }
                QFile::remove(QString::fromStdString(pfm_path));
                QFile::remove(QString::fromStdString(pfm_dn_path));
            }
            QCoreApplication::processEvents();
        }
    }
    if (!m_recordInfo.bExportVideo) {
        emit recordFinished(m_recordInfo.record_path);
        return REC_NO_RECORD_OPTION;
    }
    //Zenovis::GetInstance().blockSignals(false);
    QString imgPath = m_recordInfo.record_path + "/P/%07d.jpg";
    QString outPath = m_recordInfo.record_path + "/" + m_recordInfo.videoname;

    QString cmd = QString("ffmpeg -y -start_number %1 -r %2 -i %3 -b:v %4k -c:v mpeg4 %5")
              .arg(m_recordInfo.frameRange.first)
              .arg(m_recordInfo.fps)
              .arg(imgPath)
              .arg(m_recordInfo.bitrate)
              .arg(outPath);
    int ret = QProcess::execute(cmd);
    if (ret == 0)
    {
        if (!m_recordInfo.audioPath.isEmpty()) {
            cmd = QString("ffmpeg -y -i %1 -i %2 -c:v copy -c:a aac output_av.mp4")
                      .arg(outPath)
                      .arg(m_recordInfo.audioPath);
            ret = QProcess::execute(cmd);
            if (ret == 0) {
                emit recordFinished(m_recordInfo.record_path);
                return REC_NOERROR;
            }
            else {
                emit recordFailed(QString());
                return REC_FFMPEG_FATAL;
            }
        }
        emit recordFinished(m_recordInfo.record_path);
        return REC_NOERROR;
    }
    else
    {
        //todo get the error string from QProcess.
        QString err_info = QString(tr("ffmpeg command failed, please whether check ffmpeg exists."));
        emit recordFailed(err_info);
        return REC_NOFFMPEG;
    }
}

void RecordVideoMgr::disconnectSignal()
{
    DisplayWidget *pWid = qobject_cast<DisplayWidget *>(parent());
    if (!pWid)
        return;

    if (pWid->isGLViewport()) {
        Zenovis *pVis = getZenovis();
        bool ret = disconnect(pVis, SIGNAL(frameDrawn(int)), this, SLOT(onFrameDrawn(int)));
    } else {
        auto pView = pWid->optixViewport();
        ZASSERT_EXIT(pView);
        bool ret = disconnect(pView, SIGNAL(sig_frameRecordFinished(int)), this, SLOT(frameFinished(int)));
        ret = disconnect(pView, SIGNAL(sig_recordFinished()), this, SLOT(endRecToExportVideo()));
    }
}

void RecordVideoMgr::onFrameDrawn(int currFrame)
{
    auto& pGlobalStatus = zeno::getSession().globalStatus;
    if (m_recordInfo.bRecordByCommandLine && pGlobalStatus->failed())
    {
        emit recordFailed(QString::fromStdString(pGlobalStatus->error->message));
        return;
    }
    auto& pGlobalComm = zeno::getSession().globalComm;
    ZASSERT_EXIT(pGlobalComm);

    bool bFrameCompleted = pGlobalComm->isFrameCompleted(currFrame);
    bool bFrameRecorded = m_recordInfo.m_bFrameFinished[currFrame];

    Zenovis* pVis = getZenovis();
    ZASSERT_EXIT(pVis);

    if (bFrameCompleted && !bFrameRecorded)
    {
        if (currFrame >= m_recordInfo.frameRange.first && currFrame <= m_recordInfo.frameRange.second)
        {
            auto record_file = zeno::format("{}/P/{:07d}.jpg", m_recordInfo.record_path.toStdString(), currFrame);
            QFileInfo fileInfo(QString::fromStdString(record_file));

            auto scene = pVis->getSession()->get_scene();
            auto old_num_samples = scene->drawOptions->num_samples;
            scene->drawOptions->num_samples = m_recordInfo.numOptix;
            scene->drawOptions->msaa_samples = m_recordInfo.numMSAA;
            scene->drawOptions->denoise = m_recordInfo.needDenoise;

            auto [x, y] = pVis->getSession()->get_window_size();

            auto extname = QFileInfo(QString::fromStdString(record_file)).suffix().toStdString();
            if (pVis->getSession()->is_lock_window())
            {
                zeno::vec2i offset = pVis->getSession()->get_viewportOffset();
                pVis->getSession()->set_window_size((int)m_recordInfo.res.x(), (int)m_recordInfo.res.y(), zeno::vec2i{0,0});
                pVis->getSession()->do_screenshot(record_file, extname);
                pVis->getSession()->set_window_size(x, y, offset);
            }
            else {
                pVis->getSession()->set_window_size((int)m_recordInfo.res.x(), (int)m_recordInfo.res.y());
                pVis->getSession()->do_screenshot(record_file, extname);
                pVis->getSession()->set_window_size(x, y);
            }
            scene->drawOptions->num_samples = old_num_samples;

            m_recordInfo.m_bFrameFinished[currFrame] = true;
            emit frameFinished(currFrame);

            if (m_recordInfo.bAutoRemoveCache)
                zeno::getSession().globalComm->removeCache(currFrame);
        }

        if (currFrame == m_recordInfo.frameRange.second)
        {
            //disconnect first, to stop receiving the signal from viewport.
            disconnectSignal();

            endRecToExportVideo();

            zeno::log_critical("after executing endRecToExportVideo()");

            //clear issues:
            m_recordInfo = VideoRecInfo();
        }
    }
    else if (pGlobalComm->isFrameBroken(currFrame) && !bFrameRecorded)
    {
        //recordErrorImg(currFrame);
        zeno::log_warn("The zencache of frame {} has been removed.", currFrame);
    }
}

void RecordVideoMgr::recordErrorImg(int currFrame)
{
    QImage img(QSize((int)m_recordInfo.res.x(), (int)m_recordInfo.res.y()), QImage::Format_RGBA8888);
    img.fill(Qt::black);
    QPainter painter(&img);
    painter.setPen(Qt::white);
    QFont fnt = zenoApp->font();
    fnt.setPointSize(16);
    painter.setFont(fnt);
    painter.drawText(img.rect(), Qt::AlignCenter, QString(tr("the zencache of this frame has been removed")));
    img.save(QString::fromStdString(zeno::format("{}/P/{:07d}.jpg", m_recordInfo.record_path.toStdString(), currFrame)), "JPG");

    m_recordInfo.m_bFrameFinished[currFrame] = true;

    if (currFrame == m_recordInfo.frameRange.second)
    {
        //disconnect first, to stop receiving the signal from viewport.
        disconnectSignal();

        endRecToExportVideo();

        zeno::log_critical("after executing endRecToExportVideo()");

        //clear issues:
        m_recordInfo = VideoRecInfo();
    }
}
