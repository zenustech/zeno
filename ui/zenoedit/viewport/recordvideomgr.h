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
    bool bRecordAfterRun;
    bool bExportVideo;
    bool exitWhenRecordFinish = false;
    VideoRecInfo()
        : bRecordAfterRun(false)
        , bExportVideo(false)
        , fps(0)
        , bitrate(0)
    {
        res = { 0,0 };
        frameRange = { -1, -1 };
    }
};

class Zenovis;

class RecordVideoMgr : public QObject
{
    Q_OBJECT
public:
    RecordVideoMgr(QObject* parent = nullptr);
    ~RecordVideoMgr();
    void setRecordInfo(const VideoRecInfo& recInfo);

public slots:
    void cancelRecord();
    void onFrameDrawn(int);

signals:
    void frameFinished(int);
    void recordFinished(QString);
    void recordFailed(QString);

private:
    void endRecToExportVideo();
    Zenovis* getZenovis();
    void disconnectSignal();

    VideoRecInfo m_recordInfo;
    QStringList m_pics;
    int m_currFrame;
};


#endif