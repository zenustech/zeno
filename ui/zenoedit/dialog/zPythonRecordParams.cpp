#include "zPythonRecordParams.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QIntValidator>
#include <zeno/core/Session.h>
#include "zeno/utils/UserData.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>

zPythonRecordParams::zPythonRecordParams(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Python Record Parameters");
    setMinimumWidth(400);
    setMinimumHeight(250);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // Create form layout for parameters
    QGridLayout* formLayout = new QGridLayout();
    
    int row = 0;
    
    // cmdParamsJson
    formLayout->addWidget(new QLabel("Command Params JSON:"), row, 0);
    m_cmdParamsJsonEdit = new QLineEdit();
    m_cmdParamsJsonEdit->setText(R"({"imageFolder":"O:/ZENO_WORKFLOWTESTS/depth/P"})");
    formLayout->addWidget(m_cmdParamsJsonEdit, row++, 1);
    
    // cachePath
    formLayout->addWidget(new QLabel("Cache Path:"), row, 0);
    m_cachePathEdit = new QLineEdit();
    m_cachePathEdit->setText("C:/tmp/");
    formLayout->addWidget(m_cachePathEdit, row++, 1);
    
    // executorPath
    formLayout->addWidget(new QLabel("Executor Path:"), row, 0);
    m_executorPathEdit = new QLineEdit();
    m_executorPathEdit->setText("O:/zenobins/bin/zenoedit.exe");
    formLayout->addWidget(m_executorPathEdit, row++, 1);
    
    // optix
    formLayout->addWidget(new QLabel("Use Optix:"), row, 0);
    m_optixCheckBox = new QCheckBox();
    m_optixCheckBox->setChecked(true);
    formLayout->addWidget(m_optixCheckBox, row++, 1);
    
    // batchSize
    formLayout->addWidget(new QLabel("Batch Size:"), row, 0);
    m_batchSizeEdit = new QLineEdit();
    m_batchSizeEdit->setValidator(new QIntValidator(1, 5000, this));
    m_batchSizeEdit->setText("20");
    formLayout->addWidget(m_batchSizeEdit, row++, 1);
    
    formLayout->addWidget(new QLabel("frame start:"), row, 0);
    m_frameStart = new QLineEdit();
    m_frameStart->setValidator(new QIntValidator(0, INT_MAX, this));
    m_frameStart->setText("0");
    formLayout->addWidget(m_frameStart, row++, 1);

    formLayout->addWidget(new QLabel("frame end:"), row, 0);
    m_frameEnd = new QLineEdit();
    m_frameEnd->setValidator(new QIntValidator(0, INT_MAX, this));
    m_frameEnd->setText("0");
    formLayout->addWidget(m_frameEnd, row++, 1);

    // machineGroup
    formLayout->addWidget(new QLabel("Machine Group:"), row, 0);
    m_machineGroupEdit = new QLineEdit();
    m_machineGroupEdit->setText("all");
    formLayout->addWidget(m_machineGroupEdit, row++, 1);
    
    mainLayout->addLayout(formLayout);
    
    // Add button box
    QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    
    mainLayout->addWidget(buttonBox);
}

zPythonRecordParams::~zPythonRecordParams()
{}

bool zPythonRecordParams::getInfo(PythonRecordInfo& info, const VideoRecInfo& videoInfo) const
{
    auto& ud = zeno::getSession().userData();

    // Copy VideoRecInfo parameters
    info.fstart = m_frameStart->text().toInt();
    info.resolutionx = videoInfo.res.x();
    info.resolutiony = videoInfo.res.y();
    //info.bitrate = videoInfo.bitrate;
    info.needDenoise = videoInfo.needDenoise;
    info.samples = videoInfo.numOptix;
    info.bAov = ud.get2("output_aov", false);
    info.fend = m_frameEnd->text().toInt();
    info.bExportEXR = videoInfo.bExportEXR;

    QString path = zenoApp->graphsManagment()->zsgPath();
    if (path.isEmpty())
        return false;
    QFileInfo fi(path);
    QString zsgpath = fi.absoluteFilePath();

    // Set Python-specific parameters
    info.cmdParamsJson = m_cmdParamsJsonEdit->text();
    info.cachePath = m_cachePathEdit->text();
    info.executorPath = m_executorPathEdit->text();
    info.useOptix = m_optixCheckBox->isChecked();
    info.renderTaskPath = zsgpath;
    info.batchSize = m_batchSizeEdit->text().toInt();
    info.machineGroup = m_machineGroupEdit->text();

    return true;
}

