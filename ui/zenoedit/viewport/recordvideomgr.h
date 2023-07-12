#ifndef __RECORD_VIDEO_MGR_H__
#define __RECORD_VIDEO_MGR_H__

#include <QtWidgets>

struct VideoRecInfo
{
    QString record_path;    //store screenshot img and mp4.
    QString audioPath;
    QString videoname;
    QVector2D res = { 0,0 };
    QPair<int, int> frameRange = { -1, -1 };
    QMap<int, bool> m_bFrameFinished;
    int fps = 0;
    int bitrate = 0;
    int numMSAA = 0;
    int numOptix = 1;
    bool bExportVideo = false;
    bool needDenoise = false;
    bool exitWhenRecordFinish = false;
    bool bRecordByCommandLine = false;
    bool bAutoRemoveCache = false;
};
Q_DECLARE_METATYPE(VideoRecInfo);

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

private slots:
    void endRecToExportVideo();

private:
    Zenovis* getZenovis();
    void disconnectSignal();

    VideoRecInfo m_recordInfo;
    QStringList m_pics;
    int m_currFrame;
};


#endif