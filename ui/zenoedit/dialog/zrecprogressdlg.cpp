#include "zrecprogressdlg.h"
#include "ui_zrecprogressdlg.h"


ZRecordProgressDlg::ZRecordProgressDlg(const VideoRecInfo& info, QWidget* parent)
    : QDialog(parent)
    , m_info(info)
    , m_bCompleted(false)
{
    m_ui = new Ui::RecProgressDlg;
    m_ui->setupUi(this);
    m_ui->progressBar->setRange(info.frameRange.first, info.frameRange.second);
    m_ui->progressBar->setValue(info.frameRange.first);
    m_ui->btn->setText(tr("Cancel"));

    connect(m_ui->btn, SIGNAL(clicked()), this, SLOT(onBtnClicked()));
}

ZRecordProgressDlg::~ZRecordProgressDlg()
{

}

void ZRecordProgressDlg::onFrameFinished(int frame)
{
    m_ui->lblFrameHint->setText(QString("Recording frame %1").arg(QString::number(frame)));
    m_ui->progressBar->setValue(frame);
}

void ZRecordProgressDlg::onRecordFinished()
{
    m_bCompleted = true;
    m_ui->lblFrameHint->setText(tr("Record completed:"));
    m_ui->btn->setText(tr("Open file location"));
}

void ZRecordProgressDlg::onRecordFailed(QString msg)
{
    reject();
}

void ZRecordProgressDlg::onBtnClicked()
{
    if (m_bCompleted) {
        //open record dir.
        bool ok = QDesktopServices::openUrl(QUrl(m_info.record_path));
        accept();
    }
    else {
        reject();
    }
}

void ZRecordProgressDlg::paintEvent(QPaintEvent* event)
{
    QDialog::paintEvent(event);
}
