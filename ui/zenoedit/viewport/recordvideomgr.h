#ifndef __RECORD_VIDEO_MGR_H__
#define __RECORD_VIDEO_MGR_H__

#include <QObject>
#include "viewportwidget.h"
#include "dialog/zrecprogressdlg.h"

class RecordVideoMgr : public QObject
{
    Q_OBJECT
public:
    RecordVideoMgr(ViewportWidget* view, const VideoRecInfo& record, QObject* parent = nullptr);
    ~RecordVideoMgr();

public slots:
    void recordFrame();
    void cancelRecord();

signals:
    void frameFinished(int);
    void recordFinished();

private:
    VideoRecInfo m_recordInfo;
    ViewportWidget* m_view;     //should find specific view by mainwindow.
    QTimer* m_timer;
    int m_currFrame;
};


#endif