#include "zfeedbackdlg.h"
#include "ui_zfeedbackdlg.h"
#include <QDesktopServices>
#include <QUrl>
//#include "email/email.h"


ZFeedBackDlg::ZFeedBackDlg(QWidget * parent)
    : QDialog(parent)
{
    m_ui = new Ui::FeedBackDlg;
    m_ui->setupUi(this);
}

QString ZFeedBackDlg::content() const
{
    return m_ui->textEdit->toPlainText();
}

bool ZFeedBackDlg::isSendFile() const
{
    return m_ui->cbSendThisFile->isChecked();
}

void ZFeedBackDlg::sendEmail(const QString& subject, const QString& content, const QString& zsgContent)
{
    /*
    //todo: support open eml file with attachment.
    Email email;
    email.setReceiverAddress("626332185@qq.com");
    email.setSubject("bug feedback");
    email.setMessageText("233");
    email.addAttachment("C:\\zeno\\enum_example.zsg");
    email.openInDefaultProgram();
    */
    QString msg = QString("mailto:pengyb@zenustech.com?subject=%1&body=%2").arg(subject).arg(content);
    QDesktopServices::openUrl(QUrl(msg));
}
