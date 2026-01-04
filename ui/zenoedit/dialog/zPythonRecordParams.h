#pragma once

#include <QDialog>
#include <QLineEdit>
#include <QCheckBox>
#include <QString>
#include "viewport/recordvideomgr.h"


struct PythonRecordInfo
{
    // Parameters from VideoRecInfo
    int fstart = 0;
    int resolutionx = 1;
    int resolutiony = 1;
    bool needDenoise = false;
    int samples = 1;
    bool bAov = false;
    int fend = 0;
    bool bExportEXR = false;

    //QString jobName;
    //QString outputPath;
    
    // Parameters from zPythonRecordParams
    QString cmdParamsJson;
    QString cachePath;
    QString executorPath;
    bool useOptix = false;
    QString renderTaskPath;
    int batchSize = 1;
    QString machineGroup;

};

class zPythonRecordParams : public QDialog
{
    Q_OBJECT

public:
    zPythonRecordParams(const VideoRecInfo& videoInfo, QWidget *parent = nullptr);
    ~zPythonRecordParams();
    
    bool getInfo(PythonRecordInfo& info) const;

private:
    // VideoRecInfo parameters UI controls
    QLineEdit* m_resolutionXEdit;
    QLineEdit* m_resolutionYEdit;
    QCheckBox* m_needDenoiseCheckBox;
    QLineEdit* m_samplesEdit;
    QCheckBox* m_aovsCheckBox;
    QCheckBox* m_exportEXRCheckBox;

    // Original Python parameters UI controls
    QLineEdit* m_cmdParamsJsonEdit;
    QLineEdit* m_cachePathEdit;
    QLineEdit* m_executorPathEdit;
    QCheckBox* m_optixCheckBox;
    QLineEdit* m_batchSizeEdit;
    QLineEdit* m_frameStart;
    QLineEdit* m_frameEnd;
    QLineEdit* m_machineGroupEdit;
};

