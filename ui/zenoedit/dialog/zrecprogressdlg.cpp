#include "zrecprogressdlg.h"
#include "ui_zrecprogressdlg.h"


ZRecordProgressDlg::ZRecordProgressDlg(const VideoRecInfo& info, QWidget* parent)
    : QDialog(parent)
    , m_info(info)
    , m_bCompleted(false)
    , m_bAborted(false)
    , m_bPause(false)
{
    m_ui = new Ui::RecProgressDlg;
    m_ui->setupUi(this);
    m_ui->progressBar->setRange(info.frameRange.first, info.frameRange.second);
    m_ui->progressBar->setValue(info.frameRange.first);
    m_ui->btn->setText(tr("Cancel"));
    m_ui->pauseBtn->setText(tr("Pause"));

    connect(m_ui->btn, SIGNAL(clicked()), this, SLOT(onBtnClicked()));
    connect(m_ui->pauseBtn, SIGNAL(clicked()), this, SLOT(onPauseBtnClicked()));
}

ZRecordProgressDlg::~ZRecordProgressDlg()
{

}

void ZRecordProgressDlg::onFrameFinished(int frame)
{
    m_ui->lblFrameHint->setText(QString("Recording frame %1").arg(QString::number(frame)));
    m_ui->progressBar->setValue(frame);
}

void ZRecordProgressDlg::onRecordFinished(QString)
{
    m_bCompleted = true;
    m_ui->lblFrameHint->setText(tr("Record completed:"));
    m_ui->progressBar->setValue(m_info.frameRange.second);
    m_ui->btn->setText(tr("Open file location"));
    m_ui->pauseBtn->hide();
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
    else if(m_bAborted) {
        bool ok = QDesktopServices::openUrl(QUrl(m_info.record_path));
        reject();
    }
    else {
        m_ui->lblFrameHint->setText(tr("Record Aborted:"));
        m_ui->btn->setText(tr("Open file location"));
        m_ui->pauseBtn->hide();
        m_bAborted = true;
        emit cancelTriggered();
    }
}

void ZRecordProgressDlg::onPauseBtnClicked() {
    if (m_bPause)
    {
        emit continueTriggered();
        m_bPause = !m_bPause;
        m_ui->pauseBtn->setText(tr("Pause"));
    }
    else
    {
        emit pauseTriggered();
        m_bPause = !m_bPause;
        m_ui->pauseBtn->setText(tr("Continue"));
    }
}

void ZRecordProgressDlg::paintEvent(QPaintEvent* event)
{
    QDialog::paintEvent(event);
}
