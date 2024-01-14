#include "zrestartdlg.h"
#include "ui_zrestartdlg.h"
#include "setup/zsinstance.h"
#include "startup/zstartup.h"
#include <zeno/utils/logger.h>
#include "style/zenostyle.h"


ZRestartDlg::ZRestartDlg(QWidget* parent)
    : ZFramelessDialog(parent)
{
    m_ui = new Ui::ZRestartDlg;
    m_ui->setupUi(this);
    QString path = ":/icons/update-reboot.png";
    this->setTitleIcon(QIcon(path));
    this->setTitleText(this->windowTitle());
    int margin = ZenoStyle::dpiScaled(20);
    m_ui->m_mainWidget->setContentsMargins(margin, ZenoStyle::dpiScaled(30), margin, margin);
    this->setMainWidget(m_ui->m_mainWidget);
    QPixmap pixmap(path);
    m_ui->m_iconLabel->setPixmap(pixmap);
    m_ui->m_noteLabel->setStyleSheet("font-size:12pt;");
    m_ui->m_noteLabel->setWordWrap(true);
    m_ui->m_noteLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    connect(m_ui->m_restartBtn, &QPushButton::clicked, this, &ZRestartDlg::accept);
    connect(m_ui->m_saveAsBtn, &QPushButton::clicked, this, [=]() {
        emit saveSignal(true);
        accept();
    });
    connect(m_ui->m_saveBtn, &QPushButton::clicked, this, [=]() {
        emit saveSignal(false);
        accept();
    });
}
