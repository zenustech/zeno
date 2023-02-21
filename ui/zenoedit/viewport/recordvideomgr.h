#ifndef __RECORD_VIDEO_MGR_H__
#define __RECORD_VIDEO_MGR_H__

#include <QtWidgets>

struct VideoRecInfo
{
    QString record_path;    //store screenshot img and mp4.
    QString audioPath;
    QString videoname;
    QVector2D res;
    QPair<int, int> frameRange;
    QMap<int, bool> m_bFrameFinished;
    int fps;
    int bitrate;
    int numMSAA = 0;
    int numOptix = 1;
    int numSamples = 16;
    bool bRecordRun;
    bool bExportVideo;
    bool exitWhenRecordFinish = false;
    VideoRecInfo()
        : bRecordRun(false)
        , bExportVideo(false)
        , fps(0)
        , bitrate(0)
    {
        res = { 0,0 };
        frameRange = { -1, -1 };
    }
};

class RecordVideoMgr : public QObject
{
    Q_OBJECT
public:
    RecordVideoMgr(QObject* parent = nullptr);
    ~RecordVideoMgr();
    void setRecordInfo(const VideoRecInfo& recInfo);

public slots:
    void recordFrame();
    void cancelRecord();
    void onFrameDrawn(int);

signals:
    void frameFinished(int);
    void recordFinished();
    void recordFailed(QString);

private:
    void finishRecord();

    VideoRecInfo m_recordInfo;
    QStringList m_pics;
    QTimer* m_timer;
    int m_currFrame;
};


#endif