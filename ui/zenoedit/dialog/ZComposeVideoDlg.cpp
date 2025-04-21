#include "ZComposeVideoDlg.h"
#include "ui_ZComposeVideoDlg.h"

#include "zenomainwindow.h"


ZComposeVideoDlg::ZComposeVideoDlg(QWidget *parent)
    : QDialog(parent)
    , m_ui(new Ui::ZComposeVideoDlgClass)
{
    m_ui->setupUi(this);

    const RECORD_SETTING& info = zenoApp->graphsManagment()->recordSettings();
    m_ui->fps->setValidator(new QIntValidator);
    m_ui->fps->setText(QString::number(info.fps));
    m_ui->bitrate->setValidator(new QIntValidator);
    m_ui->bitrate->setText(QString::number(info.bitrate));
    m_ui->linePath->setText(info.record_path);

    connect(m_ui->btnGroup, SIGNAL(accepted()), this, SLOT(onAcceptClicked()));
    connect(m_ui->btnGroup, SIGNAL(rejected()), this, SLOT(reject()));
    connect(m_ui->btnOpen, &QPushButton::clicked, this, [=]() {
        DlgInEventLoopScope;
        QString path = QFileDialog::getExistingDirectory(nullptr, tr("File to Load"), "");
        if (path.isEmpty())
            return;
        m_ui->linePath->setText(path);
        });
}

ZComposeVideoDlg::~ZComposeVideoDlg()
{
    delete m_ui;
}

bool ZComposeVideoDlg::combineVideo()
{
    QDir dir(m_ui->linePath->text());
    if (m_ui->linePath->text().isEmpty() || !dir.exists()) {
        QMessageBox::information(nullptr, tr("Info"), tr("Invalid input path"));
        return false;
    }
    QString dir_path = m_ui->linePath->text();
    QDir qDir = QDir(dir_path);
    qDir.setNameFilters(QStringList("*.jpg"));
    QStringList fileList = qDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    if (fileList.empty()) {
        QMessageBox::information(nullptr, tr("Info"), tr("Jpg file not exist"));
        return false;
    }
    fileList.sort();
    QString baseName = QFileInfo(fileList[0]).baseName();
    bool ok;
    int number = baseName.toInt(&ok);
    if (!ok) {
        QMessageBox::information(nullptr, tr("Info"), tr("Jpg file not exist"));
        return false;
    }

    QString imgPath = m_ui->linePath->text() + "/%07d.jpg";
    QString outPath = m_ui->linePath->text() + "/" + (m_ui->filename->text().isEmpty() ? "output.mp4" : m_ui->filename->text() + ".mp4");
    if (QFile::exists(outPath)) {
        QMessageBox::information(nullptr, tr("Info"), tr("Output file exists"));
        return false;
    }
    QString cmd = QString("ffmpeg -y -start_number %1 -r %2 -i %3 -b:v %4k -c:v mpeg4 %5")
        .arg(number)
        .arg(m_ui->fps->text())
        .arg(imgPath)
        .arg(m_ui->bitrate->text())
        .arg(outPath);
    int ret = QProcess::execute(cmd);
    if (ret == 0) {
        QMessageBox::information(nullptr, tr("Info"), tr("Export success"));
        return true;
    } else {
        QMessageBox::information(nullptr, tr("Info"), tr("Export faild"));
        return false;
    }
}

void ZComposeVideoDlg::onAcceptClicked()
{
    if (combineVideo()) {
        accept();
    }
}

