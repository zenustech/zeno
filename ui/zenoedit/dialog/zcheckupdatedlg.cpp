#include "zcheckupdatedlg.h"
#include "ui_zcheckupdatedlg.h"
#include "updaterequest/zsinstance.h"
#include "startup/zstartup.h"
#include <zeno/utils/logger.h>
#include <zenoui/style/zenostyle.h>

ZCheckUpdateDlg::ZCheckUpdateDlg(QWidget* parent)
    : ZFramelessDialog(parent)
{
    m_ui = new Ui::ZCheckUpdateDlg;
    m_ui->setupUi(this);
    
    initUI();

    initConnection();

    getCudaVersion();
}

void ZCheckUpdateDlg::initUI()
{
    int margin = ZenoStyle::dpiScaled(20);
    m_ui->m_mainWidget->setContentsMargins(margin, ZenoStyle::dpiScaled(30), margin, margin);
    setMainWidget(m_ui->m_mainWidget);
    setTitleText(this->windowTitle());
    QString path = ":/icons/update_main.png";
    this->setTitleIcon(QIcon(path));
    QPixmap pixmap(path);
    m_ui->m_iconLabel->setPixmap(pixmap);
    m_ui->m_notesText->setReadOnly(true);
    m_ui->verticalLayout->setAlignment(Qt::AlignTop);
    m_ui->m_checkResLabel->setStyleSheet("font-size:14pt;");
    m_ui->m_notesLabel->setStyleSheet("font-size:12pt;");
    m_ui->m_checkResLabel->setText(tr("Checking for updates..."));
    m_ui->m_versionLabl->setWordWrap(true);

    m_ui->m_line->hide();
    updateView(false);
}

void ZCheckUpdateDlg::initConnection()
{
    connect(m_ui->m_ignoreBtn, &QPushButton::clicked, this, &ZCheckUpdateDlg::reject);
    connect(m_ui->m_remindBtn, &QPushButton::clicked, this, [=]() {
        emit remindSignal();
        reject();
    });
    connect(m_ui->m_updateBtn, &QPushButton::clicked, this, [=]() {
        emit updateSignal(m_version, m_url);
        accept();
    });
}

void ZCheckUpdateDlg::updateView(bool bVisible)
{
    m_ui->m_notesLabel->setVisible(bVisible);
    m_ui->m_notesText->setVisible(bVisible);
    m_ui->m_ignoreBtn->setVisible(bVisible);
    m_ui->m_remindBtn->setVisible(bVisible);
    m_ui->m_updateBtn->setVisible(bVisible);

    QSize size = bVisible ? QSize(560, 600): QSize(500, 180);
    m_ui->m_mainWidget->setFixedSize(ZenoStyle::dpiScaledSize(size));
}

void ZCheckUpdateDlg::requestLatestVersion()
{
    connect(ZsInstance::Instance(), &ZsInstance::sig_netReqFinish, this, &ZCheckUpdateDlg::slt_netReqFinish);
    QTimer::singleShot(10, [=] {
        ZsInstance::Instance()->NetRequest(GET_VERS, 0, URL, "");
    });
}

void ZCheckUpdateDlg::slt_netReqFinish(const QString& data, const QString& id)
{
    if (data.isEmpty())
        return;

    QJsonParseError e;
    QJsonDocument jsonDoc = QJsonDocument::fromJson(data.toUtf8(), &e);
    if (e.error != QJsonParseError::NoError && jsonDoc.isNull())
    {
        zeno::log_error(e.errorString().toStdString());
        return;
    }
    QJsonObject jsonObj = jsonDoc.object();
    if (jsonObj.value("code").toInt() == 20000)
    {
        QJsonObject tempObj = jsonObj.value("data").toObject();
        QJsonArray tempAry = tempObj.value("records").toArray();
        for (auto p : tempAry)
        {
            QJsonObject tempSubObj = p.toObject();
            m_version = tempSubObj.value("version").toString();
            QString currVersion = QString::fromStdString(getZenoVersion());
            if (m_version == currVersion)
            {
                m_ui->m_checkResLabel->setText(tr("It is the latest version!"));
                m_ui->m_versionLabl->setText(tr("current version %1 (Release) (64 bit)").arg(currVersion));
                QPixmap pixmap(":/icons/ZENO-logo128.png");
                m_ui->m_iconLabel->setPixmap(pixmap);
                m_ui->m_line->show();
                break;
            }
            QJsonArray tempSubAry = tempSubObj.value("platforms").toArray();
            QString notes;
            for (auto pSub : tempSubAry)
            {
                QJsonObject tempSub2Obj = pSub.toObject();
                QString url = tempSub2Obj.value("url").toString();
                if (tempSub2Obj.value("platformName").toString() == m_cudaVersion)
                {
                    m_url = url;
                    break;
                }
                notes = tempSub2Obj.value("discription").toString();
            }
            if (m_url.isEmpty())
            {
                m_ui->m_checkResLabel->setText(tr("The latest version of ZENO is unavailable!"));
            }
            else
            {
                m_ui->m_checkResLabel->setText(tr("The latest version of ZENO is available!"));
                m_ui->m_versionLabl->setText(tr("The %1 version of ZENO is available. The version you installed is %2, do you want to download the latest version?").arg(m_version, currVersion));
                m_ui->m_notesText->setText(notes);
                updateView(true);
            }
            break;
        }
    }
}

void ZCheckUpdateDlg::getCudaVersion()
{
    QProcess *process = new QProcess(this);
    connect(process, &QProcess::readyReadStandardOutput, this, [=] {
        QString output = process->readAllStandardOutput();
        //Cuda compilation tools, release 12.1
        QRegExp rx("Cuda compilation tools, release (\\d+)\\.(\\d+)");
        if (rx.indexIn(output) != -1)
        {
            auto caps = rx.capturedTexts();
            int ver;
            if (caps.length() == 3) {
                ver = caps[1].toInt();
            }
            if (ver == 11)
            {
                m_cudaVersion = "cuda11";
            }
            else if (ver == 12)
            {
                m_cudaVersion = "cuda12";
            }
            else
            {
                m_cudaVersion = "cpu";
            }
            requestLatestVersion();
        }
    });
    connect(process, &QProcess::errorOccurred, this, [=](QProcess::ProcessError error) {
        m_ui->m_checkResLabel->setText(tr("Can not find cuda!"));
    });
    process->start("nvcc --version");
}