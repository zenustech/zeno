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
    } else {
        return val;
    }
}

const int ZenoSettingsManager::getShortCut(const QString &key) 
{
    ShortCutInfo *info = getShortCutInfo(key, m_shortCutInfos);
    if (!info)
        return Qt::Key_unknown;
    QKeySequence keySeq = QKeySequence(info->shortcut);
    int ret = 0;
    for (int i = 0; i < keySeq.count(); i++) {
        ret += keySeq[i];
    }
    return ret;
}

void ZenoSettingsManager::setShortCut(const QString &key, const QString &value) 
{
    ShortCutInfo *info = getShortCutInfo(key, m_shortCutInfos);
    if (info) 
    {
        info->shortcut = value;
        emit valueChanged(key);
    }
}

void ZenoSettingsManager::writeShortCutInfo(const ShortCutInfo *info) 
{
    ShortCutInfo *defaultInfos;
    getDefaultShortCutInfo(&defaultInfos);
    rapidjson::StringBuffer str;
    PRETTY_WRITER writer(str);
    writer.StartArray();
    bool bChanged = false;
    while (info) {
        ShortCutInfo *value = getShortCutInfo(info->key, m_shortCutInfos);
        if (info->shortcut != value->shortcut) {
            setShortCut(info->key, info->shortcut);
        }
        value = getShortCutInfo(info->key, defaultInfos);
        if (info->shortcut == value->shortcut) {
            info = info->next;
            continue;
        }
        writer.StartObject();
        writer.Key("key");
        writer.String(info->key.toUtf8());
        writer.Key("shortcut");
        writer.String(info->shortcut.toUtf8());
        writer.EndObject();
        info = info->next;
    }
    writer.EndArray();
    QString strJson = QString::fromUtf8(str.GetString());
    setValue(zsShortCut, strJson);
    delete defaultInfos;
    defaultInfos = nullptr;
}
void ZenoSettingsManager::initShortCutInfos() 
{
    getDefaultShortCutInfo(&m_shortCutInfos);
    QSettings settings(zsCompanyName, zsEditor);
    QVariant value = settings.value(zsShortCut);
    rapidjson::Document doc;
    doc.Parse(value.toByteArray());

    if (doc.IsArray()) {
        auto array = doc.GetArray();
        int rowCount = array.Size();
        for (int row = 0; row < rowCount; row++) {
            QString key = array[row]["key"].GetString();
            QString shortcut = array[row]["shortcut"].GetString();
            ShortCutInfo *info = getShortCutInfo(key, m_shortCutInfos);
            if (info) {
                info->shortcut = shortcut;
            }
        }
    }
}
void ZenoSettingsManager::getDefaultShortCutInfo(ShortCutInfo **info) 
{
    QVector<ShortCutInfo> infoVect;
    infoVect = {
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
    };

    ShortCutInfo *tmp = new ShortCutInfo();
    *info = tmp;
    for (auto item : infoVect)
    {
        *tmp = item;
        tmp->next = new ShortCutInfo();
        tmp = tmp->next;
    }
    tmp = nullptr;
}
ShortCutInfo* ZenoSettingsManager::getShortCutInfo(const QString &key, ShortCutInfo *shortCutInfos) 
{
    int index = 0;
    auto info = shortCutInfos;
    while (info) 
    {
        if (info->key == key) {
            return info;
        }
        info = info->next;
        index++;
    }
    return nullptr;
}
