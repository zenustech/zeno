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
#include "zenomainwindow.h"

zPythonRecordParams::zPythonRecordParams(const VideoRecInfo& videoInfo, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Python Record Parameters");
    setMinimumWidth(400);
    setMinimumHeight(350); // Increased height to accommodate new controls
    
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // Create form layout for parameters
    QGridLayout* formLayout = new QGridLayout();
    
    int row = 0;
    
    QString path = zenoApp->graphsManagment()->zsgPath();
    QFileInfo fi(path);
    QString zsgpath = fi.absoluteFilePath();

    formLayout->addWidget(new QLabel("shot Name:"), row, 0);
    m_shotName = new QLineEdit();
    m_shotName->setText(fi.baseName());
    formLayout->addWidget(m_shotName, row++, 1);

    // Add VideoRecInfo parameters section with a separator
    formLayout->addWidget(new QLabel("<b>Video Recording Parameters</b>"), row, 0, 1, 2);
    row++;

    // Resolution X
    formLayout->addWidget(new QLabel("Resolution X:"), row, 0);
    m_resolutionXEdit = new QLineEdit();
    m_resolutionXEdit->setValidator(new QIntValidator(1, 10000, this));
    m_resolutionXEdit->setText(QString::number(videoInfo.res.x()));
    m_resolutionXEdit->setReadOnly(true); // Make read-only as it comes from VideoRecInfo
    formLayout->addWidget(m_resolutionXEdit, row++, 1);

    // Resolution Y
    formLayout->addWidget(new QLabel("Resolution Y:"), row, 0);
    m_resolutionYEdit = new QLineEdit();
    m_resolutionYEdit->setValidator(new QIntValidator(1, 10000, this));
    m_resolutionYEdit->setText(QString::number(videoInfo.res.y()));
    m_resolutionYEdit->setReadOnly(true); // Make read-only as it comes from VideoRecInfo
    formLayout->addWidget(m_resolutionYEdit, row++, 1);

    // Need Denoise
    formLayout->addWidget(new QLabel("Need Denoise:"), row, 0);
    m_needDenoiseCheckBox = new QCheckBox();
    m_needDenoiseCheckBox->setChecked(videoInfo.needDenoise);
    formLayout->addWidget(m_needDenoiseCheckBox, row++, 1);

    // Samples (numOptix)
    formLayout->addWidget(new QLabel("Samples:"), row, 0);
    m_samplesEdit = new QLineEdit();
    m_samplesEdit->setValidator(new QIntValidator(1, 1000, this));
    m_samplesEdit->setText(QString::number(videoInfo.numOptix));
    formLayout->addWidget(m_samplesEdit, row++, 1);

    // AOVs
    formLayout->addWidget(new QLabel("AOVs:"), row, 0);
    m_aovsCheckBox = new QCheckBox();
    auto& ud = zeno::getSession().userData();
    bool aovsValue = ud.get2("output_aov", false);
    m_aovsCheckBox->setChecked(aovsValue);
    formLayout->addWidget(m_aovsCheckBox, row++, 1);

    // Export EXR
    formLayout->addWidget(new QLabel("Export EXR:"), row, 0);
    m_exportEXRCheckBox = new QCheckBox();
    m_exportEXRCheckBox->setChecked(videoInfo.bExportEXR);
    formLayout->addWidget(m_exportEXRCheckBox, row++, 1);

    // Add separator for Python parameters
    formLayout->addWidget(new QLabel("<b>Python Execution Parameters</b>"), row, 0, 1, 2);
    row++;

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
    
    formLayout->addWidget(new QLabel("render task path:"), row, 0);
    m_renderTaskZsgPath = new QLineEdit();
    m_renderTaskZsgPath->setText(zsgpath);
    formLayout->addWidget(m_renderTaskZsgPath, row++, 1);

    // batchSize
    formLayout->addWidget(new QLabel("Batch Size:"), row, 0);
    m_batchSizeEdit = new QLineEdit();
    m_batchSizeEdit->setValidator(new QIntValidator(1, 5000, this));
    m_batchSizeEdit->setText("20");
    formLayout->addWidget(m_batchSizeEdit, row++, 1);
    
    auto mainw = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainw);
    auto timelineInfo = mainw->timelineInfo();

    formLayout->addWidget(new QLabel("frame start:"), row, 0);
    m_frameStart = new QLineEdit();
    m_frameStart->setValidator(new QIntValidator(0, INT_MAX, this));
    m_frameStart->setText(QString::number(timelineInfo.beginFrame));
    formLayout->addWidget(m_frameStart, row++, 1);

    formLayout->addWidget(new QLabel("frame end:"), row, 0);
    m_frameEnd = new QLineEdit();
    m_frameEnd->setValidator(new QIntValidator(0, INT_MAX, this));
    m_frameEnd->setText(QString::number(timelineInfo.endFrame));
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

bool zPythonRecordParams::getInfo(PythonRecordInfo& info) const
{
    if (m_renderTaskZsgPath->text().isEmpty())
    {
        return false;
    }

    info.shotname = m_shotName->text();

    // Copy VideoRecInfo parameters from UI controls
    info.fstart = m_frameStart->text().toInt();
    info.resolutionx = m_resolutionXEdit->text().toInt();
    info.resolutiony = m_resolutionYEdit->text().toInt();
    //info.bitrate = videoInfo.bitrate;
    info.needDenoise = m_needDenoiseCheckBox->isChecked();
    info.samples = m_samplesEdit->text().toInt();
    info.bAov = m_aovsCheckBox->isChecked();
    info.fend = m_frameEnd->text().toInt();
    info.bExportEXR = m_exportEXRCheckBox->isChecked();

    // Set Python-specific parameters
    info.cmdParamsJson = m_cmdParamsJsonEdit->text();
    info.cachePath = m_cachePathEdit->text();
    info.executorPath = m_executorPathEdit->text();
    info.useOptix = m_optixCheckBox->isChecked();
    info.renderTaskPath = m_renderTaskZsgPath->text();
    info.batchSize = m_batchSizeEdit->text().toInt();
    info.machineGroup = m_machineGroupEdit->text();

    return true;
}

