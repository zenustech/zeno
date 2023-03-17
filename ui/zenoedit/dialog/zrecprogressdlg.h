#ifndef __ZREC_PROGRESS_DLG_H__
#define __ZREC_PROGRESS_DLG_H__

namespace Ui
{
    class RecProgressDlg;
}

#include <QDialog>
#include "../viewport/recordvideomgr.h"

class ZRecordProgressDlg : public QDialog
{
    Q_OBJECT
public:
    ZRecordProgressDlg(const VideoRecInfo& info, QWidget* parent = nullptr);
    ~ZRecordProgressDlg();

signals:
    void cancelTriggered();
    void pauseTriggered();
    void continueTriggered();

protected:
    void paintEvent(QPaintEvent* event) override;

public slots:
    void onFrameFinished(int frame);
    void onRecordFinished(QString);
    void onRecordFailed(QString);

private slots:
    void onBtnClicked();
    void onPauseBtnClicked();

private:
    VideoRecInfo m_info;
    Ui::RecProgressDlg* m_ui;
    bool m_bCompleted;
    bool m_bAborted;
    bool m_bPause;
};


#endif