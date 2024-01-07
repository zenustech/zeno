#include "zaboutdlg.h"
#include "ui_zaboutdlg.h"
#include <QFileDialog>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zassert.h"
#include "startup/zstartup.h"


ZAboutDlg::ZAboutDlg(QWidget* parent)
    : QDialog(parent)
{
    m_ui = new Ui::AboutDlg;
    m_ui->setupUi(this);

    QPixmap img;
    img.load(":/icons/ZENO-logo128.png");
    m_ui->lblLogo->clear();
    m_ui->lblLogo->setPixmap(img);
    m_ui->lblProductName->setText("Zeno");
    m_ui->lblVersion->setText(QString::fromStdString(getZenoVersion()));
}
