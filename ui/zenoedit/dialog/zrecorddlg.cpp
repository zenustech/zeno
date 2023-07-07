#include "zrecorddlg.h"
#include "ui_zrecorddlg.h"
#include <QFileDialog>
#include <zeno/core/Session.h>
#include "zenomainwindow.h"
#include "zeno/utils/UserData.h"
#include "zassert.h"
#include "settings/zsettings.h"


ZRecordVideoDlg::ZRecordVideoDlg(QWidget* parent)
    : QDialog(parent)
{
    m_ui = new Ui::RecordVideoDlg;
    m_ui->setupUi(this);

    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("recInfo");
    m_ui->fps->setValidator(new QIntValidator);
    m_ui->fps->setText(settings.value("fps").isValid() ? settings.value("fps").toString() : "24");
    m_ui->bitrate->setValidator(new QIntValidator);
    m_ui->bitrate->setText(settings.value("bitrate").isValid() ? settings.value("bitrate").toString() : "200000");
    m_ui->lineWidth->setValidator(new QIntValidator);
    m_ui->lineWidth->setText(settings.value("width").isValid() ? settings.value("width").toString() : "1280");
    m_ui->lineHeight->setValidator(new QIntValidator);
    m_ui->lineHeight->setText(settings.value("height").isValid() ? settings.value("height").toString() : "720");
    m_ui->msaaSamplerNumber->setValidator(new QIntValidator);
    m_ui->msaaSamplerNumber->setText(settings.value("numMSAA").isValid() ? settings.value("numMSAA").toString() : "0");
    m_ui->optixSamplerNumber->setValidator(new QIntValidator);
    m_ui->optixSamplerNumber->setText(settings.value("numOptix").isValid() ? settings.value("numOptix").toString() : "1");
    settings.endGroup();

    m_ui->cbRemoveAfterRender->setCheckState(Qt::Unchecked);
    m_ui->cbPresets->addItems({"540P", "720P", "1080P", "2K", "4K"});
    m_ui->cbPresets->setCurrentIndex(1);

    connect(m_ui->cbPresets, &QComboBox::currentTextChanged, this, [=](auto res) {
        auto v = std::map<QString, std::tuple<int, int>> {
                {"540P", {960, 540}},
                {"720P", {1280, 720}},
                {"1080P", {1920, 1080}},
                {"2K", {2560, 1440}},
                {"4K", {3840, 2160}},
        }.at(res);
        m_ui->lineWidth->setText(QString::number(std::get<0>(v)));
        m_ui->lineHeight->setText(QString::number(std::get<1>(v)));
    });

    connect(m_ui->btnOpen, &QPushButton::clicked, this, [=]() {
        DlgInEventLoopScope;
        QString path = QFileDialog::getExistingDirectory(nullptr, tr("File to Save"), "");
        if (path.isEmpty())
            return;
        m_ui->linePath->setText(path);
    });

    connect(m_ui->btnGroup, SIGNAL(accepted()), this, SLOT(accept()));
    connect(m_ui->btnGroup, SIGNAL(rejected()), this, SLOT(reject()));
}

bool ZRecordVideoDlg::getInfo(VideoRecInfo &info)
{
    auto &ud = zeno::getSession().userData();
    ud.set2("output_aov", m_ui->cbAOV->checkState() == Qt::Checked);
    auto &path = info.record_path;
    auto &fn = info.videoname;
    info.fps = m_ui->fps->text().toInt();
    info.bitrate = m_ui->bitrate->text().toInt();
    info.numMSAA = m_ui->msaaSamplerNumber->text().toInt();
    info.numOptix = m_ui->optixSamplerNumber->text().toInt();
    info.res[0] = m_ui->lineWidth->text().toFloat();
    info.res[1] = m_ui->lineHeight->text().toFloat();
    path = m_ui->linePath->text();
    info.bExportVideo = m_ui->cbExportVideo->checkState() == Qt::Checked;
    info.needDenoise = m_ui->cbNeedDenoise->checkState() == Qt::Checked;
    info.bAutoRemoveCache = m_ui->cbRemoveAfterRender->checkState() == Qt::Checked;
    if (path.isEmpty())
    {
        QTemporaryDir dir;
        dir.setAutoRemove(false);
        path = dir.path();
    }
    //create directory to store screenshot pngs.
    QDir dir(path);
    if (!QFileInfo(path).isDir())
        return false;
    dir.mkdir("P");

    fn = m_ui->lineName->text();
    if (fn.isEmpty())
    {
        fn = "capture";
        const QString& suffix = ".mp4";
        int idx = 1;
        while (QFileInfo::exists(path + "/" + fn + suffix))
        {
            fn = "capture_" + QString::number(idx);
            idx++;
        }
        fn += suffix;
    }
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("recInfo");
    settings.setValue("fps", info.fps);
    settings.setValue("bitrate", info.bitrate);
    settings.setValue("numMSAA", info.numMSAA);
    settings.setValue("numOptix", info.numOptix);
    settings.setValue("width", info.res[0]);
    settings.setValue("height", info.res[1]);
    settings.endGroup();
    return true;
}