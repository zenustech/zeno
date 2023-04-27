#include "zenosettingsmanager.h"
#include "settings/zsettings.h"
#include <rapidjson/document.h>
#include <zenomodel/include/jsonhelper.h>


ZenoSettingsManager& ZenoSettingsManager::GetInstance()
{
    static ZenoSettingsManager instance;
    return instance;
}

ZenoSettingsManager::ZenoSettingsManager(QObject *parent) : 
    QObject(parent)
{
    QSettings settings(zsCompanyName, zsEditor);
    if (settings.allKeys().indexOf(zsShowGrid) == -1) {
        //show grid by default.
        setValue(zsShowGrid, true);
    }
    initShortCutInfos();
}

void ZenoSettingsManager::setValue(const QString& name, const QVariant& value) 
{
    QSettings settings(zsCompanyName, zsEditor);
    QVariant oldValue = settings.value(name);
    if (oldValue != value)
    {
        settings.setValue(name, value);
        if (name != zsShortCut)
            emit valueChanged(name);
    }
}

QVariant ZenoSettingsManager::getValue(const QString& zsName) const
{
    if (zsName == zsShortCut)
        return QVariant::fromValue(m_shortCutInfos);

    QSettings settings(zsCompanyName, zsEditor);
    QVariant val = settings.value(zsName);
    if (!val.isValid()) {
        if (zsName == zsShowGrid)
            return true;
        else if (zsName == zsSnapGrid)
            return false;
        else if (zsTraceErrorNode == zsName)
            return false;
        else
            return QVariant();
    } else {
        return val;
    }
}

const int ZenoSettingsManager::getShortCut(const QString &key) 
{
    ShortCutInfo info;
    getShortCutInfo(key, info);
    QKeySequence keySeq = QKeySequence(info.shortcut);

    int ret = 0;
    for (int i = 0; i < keySeq.count(); i++) {
        ret += keySeq[i];
    }
    return ret;
}

void ZenoSettingsManager::setShortCut(const QString &key, const QString &value) 
{
    ShortCutInfo info;
    int index = getShortCutInfo(key, info);
    if (index >= 0)
    {
        info.shortcut = value;
        m_shortCutInfos[index] = info;
        emit valueChanged(key);
    }
}

void ZenoSettingsManager::writeShortCutInfo(const QVector<ShortCutInfo> &infos) 
{
    QVector<ShortCutInfo> defaultInfos = getDefaultShortCutInfo();
    rapidjson::StringBuffer str;
    PRETTY_WRITER writer(str);
    writer.StartArray();
    bool bChanged = false;
    for (int i = 0; i < infos.count(); i++)
    {
        ShortCutInfo info = infos.at(i);
        if (info.shortcut != m_shortCutInfos.at(i).shortcut) {
            setShortCut(info.key, info.shortcut);
        }
        if (info.shortcut == defaultInfos.at(i).shortcut) {
            continue;
        }
        writer.StartObject();
        writer.Key("key");
        writer.String(info.key.toUtf8());
        writer.Key("shortcut");
        writer.String(info.shortcut.toUtf8());
        writer.EndObject();
    }

    writer.EndArray();
    QString strJson = QString::fromUtf8(str.GetString());
    setValue(zsShortCut, strJson);
}

void ZenoSettingsManager::initShortCutInfos() 
{
    m_shortCutInfos = getDefaultShortCutInfo();
    QSettings settings(zsCompanyName, zsEditor);
    QVariant value = settings.value(zsShortCut);
    rapidjson::Document doc;
    doc.Parse(value.toByteArray());

    if (doc.IsArray()) {
        auto array = doc.GetArray();
        int rowCount = array.Size();
        for (int row = 0; row < rowCount; row++)
        {
            ShortCutInfo info;
            QString key = array[row]["key"].GetString();
            QString shortcut = array[row]["shortcut"].GetString();
            int index = getShortCutInfo(key, info);
            if (index >= 0 && info.shortcut != shortcut) {
                info.shortcut = shortcut;
                m_shortCutInfos[index] = info;
            }
        }
    }
}

QVector<ShortCutInfo> ZenoSettingsManager::getDefaultShortCutInfo() 
{
    QVector<ShortCutInfo> ret = {
        {ShortCut_Save, QObject::tr("Save"), "Ctrl+S"},
        {ShortCut_SaveAs, QObject::tr("Save As"), "Ctrl+Shift+S"},
        {ShortCut_New_File, QObject::tr("New File"), "Ctrl+N"},
        {ShortCut_Open, QObject::tr("Open"), "Ctrl+O"},
        {ShortCut_Undo, QObject::tr("Undo"), "Ctrl+Z"},
        {ShortCut_Redo, QObject::tr("Redo"), "Ctrl+Y"},
        {ShortCut_NewSubgraph, QObject::tr("New Subgraph"), "Ctrl+E"},
        {ShortCut_Focus, QObject::tr("Focus"), "Alt+F"},
        {ShortCut_Run, QObject::tr("Run"), "F2"},
        {ShortCut_Kill, QObject::tr("Kill"), "Shift+F2"},
        {ShortCut_SmoothShading, QObject::tr("Smooth Shading"), "F5"},
        {ShortCut_NormalCheck, QObject::tr("Normal Check"), "Shift+F5"},
        {ShortCut_WireFrame, QObject::tr("Wire Frame"), "F6"},
        {ShortCut_ShowGrid, QObject::tr("Show Grid"), "Shift+F6"},
        {ShortCut_Solid, QObject::tr("Solid"), "F7"},
        {ShortCut_Shading, QObject::tr("Shading"), "Shift+F7"},
        {ShortCut_Optix, QObject::tr("Optix"), "F8"},
        {ShortCut_ScreenShoot, QObject::tr("Screen Shoot"), "F12"},
        {ShortCut_RecordVideo, QObject::tr("Record Video"), "Shift+F12"},
        {ShortCut_Import, QObject::tr("Import"), "Ctrl+Shift+O"},
        {ShortCut_Export_Graph, QObject::tr("Export Graph"), "Ctrl+Shift+E"},
        {ShortCut_NewNode, QObject::tr("New Node"), "Tab"},
        {ShortCut_MovingHandler, QObject::tr("Moving Handler"), "T"},
        {ShortCut_RevolvingHandler, QObject::tr("Rotating Handler"), "R"},
        {ShortCut_ScalingHandler, QObject::tr("Scaling Handler"), "E"},
        {ShortCut_CoordSys, QObject::tr("CoordSys"), "M"},
        {ShortCut_InitHandler, QObject::tr("Init Handler"), "Backspace"},
        {ShortCut_ReduceHandler, QObject::tr("Reduce Handler"), "-"},
        {ShortCut_AmplifyHandler, QObject::tr("Amplify Handler"), "+"},
        {ShortCut_InitViewPos, QObject::tr("Init View Pos"), "0"},
        {ShortCut_FrontView, QObject::tr("Front View"), "1"},
        {ShortCut_RightView, QObject::tr("Right View"), "3"},
        {ShortCut_VerticalView, QObject::tr("Vertical View"), "7"},
        {ShortCut_BackView, QObject::tr("Back View"), "Ctrl+1"},
        {ShortCut_LeftView, QObject::tr("Left View"), "Ctrl+3"},
        {ShortCut_UpwardView, QObject::tr("Upward View"), "Ctrl+7"},
        {ShortCut_FloatPanel, QObject::tr("Float Panel"), "P"}, 
        {ShortCut_SelectAllNodes, QObject::tr("Float Panel"), "Ctrl+A"}
    };
    return ret;
}

int ZenoSettingsManager::getShortCutInfo(const QString& key, ShortCutInfo& info)
{
    for (int i = 0; i < m_shortCutInfos.size(); i++) {
        if (m_shortCutInfos.at(i).key == key) {
            info = m_shortCutInfos.at(i);
            return i;
        }
    }
    return -1;
}
