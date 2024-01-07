#ifndef __ZFEEDBACK_DLG_H__
#define __ZFEEDBACK_DLG_H__

#include <QtWidgets>

namespace Ui {
    class FeedBackDlg;
}

class ZFeedBackDlg : public QDialog
{
    Q_OBJECT
public:
    ZFeedBackDlg(QWidget *parent = nullptr);
    QString content() const;
    bool isSendFile() const;
    void sendEmail(const QString& subject, const QString& content, const QString& zsgContent);

private:
    Ui::FeedBackDlg *m_ui;
};

#endif