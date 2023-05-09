#ifndef __ZREC_FRAME_SELECT_DIG_H__
#define __ZREC_FRAME_SELECT_DIG_H__

#include <QtWidgets>

namespace Ui
{
    class RecFrameSelectDlg;
}

class ZRecFrameSelectDlg : public QDialog
{
    Q_OBJECT
public:
    ZRecFrameSelectDlg(QWidget* parent = nullptr);
    QPair<int, int> recordFrameRange(bool& runBeforeRun) const;

private slots:
    void onRunNow();
    void onRecordLastRun();
    void onRecordNow();
    void onCancelRecord();
    void onRecFromEdited();
    void onRecToEdited();

private:
    Ui::RecFrameSelectDlg* m_ui;
    bool validateFrame();

    int m_recStartF;
    int m_recEndF;
    bool m_bRunBeforeRecord;
};


#endif