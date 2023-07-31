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
    m_ui->progressBar->setFormat(tr("%1%").arg(QString::number(0, 'f', 1)));
    m_ui->progressBar->setAlignment(Qt::AlignLeft | Qt::AlignVCenter); // ¶ÔÆë·½Ê½

    //todo: pause recording.
    m_ui->pauseBtn->hide();
    m_ui->pauseBtn->setText(tr("Pause"));

    connect(m_ui->btnOpenLoc, SIGNAL(clicked()), this, SLOT(onOpenLocClicked()));
    connect(m_ui->btnCancel, SIGNAL(clicked()), this, SLOT(onBtnClicked()));
    connect(m_ui->pauseBtn, SIGNAL(clicked()), this, SLOT(onPauseBtnClicked()));
}

ZRecordProgressDlg::~ZRecordProgressDlg()
{

}

void ZRecordProgressDlg::onOpenLocClicked()
{
    bool ok = QDesktopServices::openUrl(QUrl(m_info.record_path));
    if (m_bCompleted) {
        accept();
    }
    else if (m_bAborted) {
        reject();
    }
}

void ZRecordProgressDlg::onFrameFinished(int frame)
{
    double dProgress =
        (double)(frame - m_info.frameRange.first) / (m_info.frameRange.second - m_info.frameRange.first + 1);
    if (frame == m_info.frameRange.second)
        dProgress = 1;
    m_ui->progressBar->setFormat(tr("%1%").arg(QString::number(dProgress * 100, 'f', 1)));
    m_ui->lblFrameHint->setText(tr("Recording frame %1").arg(QString::number(frame)));
    m_ui->progressBar->setValue(frame);
}

void ZRecordProgressDlg::onRecordFinished(QString)
{
    m_bCompleted = true;
    m_ui->lblFrameHint->setText(tr("Record completed:"));
    m_ui->progressBar->setValue(m_info.frameRange.second);
    m_ui->btnCancel->setText(tr("Finished"));
    m_ui->btnCancel->hide();
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
        accept();
    }
    else if(m_bAborted) {
        reject();
    }
    else {
        m_ui->lblFrameHint->setText(tr("Record Aborted:"));
        m_ui->pauseBtn->hide();
        m_bAborted = true;
        emit cancelTriggered();
        reject();
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
