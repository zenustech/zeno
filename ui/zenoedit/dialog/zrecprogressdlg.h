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

protected:
    void paintEvent(QPaintEvent* event) override;

public slots:
    void onFrameFinished(int frame);

private:
    VideoRecInfo m_info;
    Ui::RecProgressDlg* m_ui;
};


#endif