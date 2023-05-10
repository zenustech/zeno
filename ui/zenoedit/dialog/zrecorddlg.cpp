#include "zrecorddlg.h"
#include "ui_zrecorddlg.h"
#include <QFileDialog>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zassert.h"


ZRecordVideoDlg::ZRecordVideoDlg(QWidget* parent)
    : QDialog(parent)
{
    m_ui = new Ui::RecordVideoDlg;
    m_ui->setupUi(this);

    m_ui->fps->setValidator(new QIntValidator);
    m_ui->fps->setText("24");
    m_ui->bitrate->setValidator(new QIntValidator);
    m_ui->bitrate->setText("200000");
    m_ui->lineWidth->setValidator(new QIntValidator);
    m_ui->lineWidth->setText("1280");
    m_ui->lineHeight->setValidator(new QIntValidator);
    m_ui->lineHeight->setText("720");
    m_ui->msaaSamplerNumber->setValidator(new QIntValidator);
    m_ui->msaaSamplerNumber->setText("0");
    m_ui->optixSamplerNumber->setValidator(new QIntValidator);
    m_ui->optixSamplerNumber->setText("1");

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

bool ZRecordVideoDlg::getInfo(int& fps, int& bitrate, float& width, float& height, 
             QString& path, QString& fn, int &numOptix, int &numMSAA, bool& bExportVideo)
{
    fps = m_ui->fps->text().toInt();
    bitrate = m_ui->bitrate->text().toInt();
    numMSAA = m_ui->msaaSamplerNumber->text().toInt();
    numOptix = m_ui->optixSamplerNumber->text().toInt();
    width = m_ui->lineWidth->text().toFloat();
    height = m_ui->lineHeight->text().toFloat();
    path = m_ui->linePath->text();
    bExportVideo = m_ui->cbExportVideo->checkState() == Qt::Checked;
    if (path.isEmpty())
    {
        QTemporaryDir dir;
        dir.setAutoRemove(false);
        path = dir.path();
    }
    //create directory to store screenshot pngs.
    QDir dir(path);
    ZASSERT_EXIT(dir.exists(), false);
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
    return true;
}