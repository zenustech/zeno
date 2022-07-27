#include "zrecprogressdlg.h"
#include "ui_zrecprogressdlg.h"


ZRecordProgressDlg::ZRecordProgressDlg(const VideoRecInfo& info, QWidget* parent)
    : QDialog(parent)
    , m_info(info)
{
    m_ui = new Ui::RecProgressDlg;
    m_ui->setupUi(this);
    m_ui->progressBar->setRange(info.frameRange.first, info.frameRange.second);
    m_ui->progressBar->setValue(info.frameRange.first);

    connect(m_ui->buttonBox, SIGNAL(rejected()), this, SLOT(reject()));
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
    this->accept();
}

void ZRecordProgressDlg::paintEvent(QPaintEvent* event)
{
    QDialog::paintEvent(event);
}
