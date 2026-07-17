#pragma once

#include <QtWidgets>
#include <ffmpeg/ImageSequenceEncoder.h>
#include <QThread>

QT_BEGIN_NAMESPACE
namespace Ui { class ZComposeVideoDlgClass; };
QT_END_NAMESPACE

class ZComposeVideoDlg : public QDialog
{
    Q_OBJECT

public:
    ZComposeVideoDlg(QWidget *parent = nullptr);
    ~ZComposeVideoDlg();

    bool combineVideo();

public slots:
    void onAcceptClicked();

    //合成视频worker和进度条相关
    void startVideoCompose();
    void updateProgress(int currentFrame);
    void onComposeFinished(bool success);

private:
    Ui::ZComposeVideoDlgClass * m_ui;

    //合成视频worker和进度条相关
    QThread* m_workerThread;
    QProgressDialog* m_progressDialog;
    bool m_cancelCompose;
};
