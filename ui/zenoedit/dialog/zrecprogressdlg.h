#ifndef __ZREC_PROGRESS_DLG_H__
#define __ZREC_PROGRESS_DLG_H__

namespace Ui
{
    class RecProgressDlg;
}

#include <QDialog>
#include "../viewport/viewportwidget.h"

class ZRecordProgressDlg : public QDialog
{
    Q_OBJECT
public:
    ZRecordProgressDlg(const VideoRecInfo& info, QWidget* parent = nullptr);
    ~ZRecordProgressDlg();

signals:
    void cancelTriggered();

protected:
    void paintEvent(QPaintEvent* event) override;

public slots:
    void onFrameFinished(int frame);
    void onRecordFinished();
    void onRecordFailed(QString);

private slots:
    void onBtnClicked();

private:
    VideoRecInfo m_info;
    Ui::RecProgressDlg* m_ui;
    bool m_bCompleted;
};


#endif