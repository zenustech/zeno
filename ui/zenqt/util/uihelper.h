#ifndef __UI_HELPER_H__
#define __UI_HELPER_H__

#include <rapidjson/document.h>
#include "util/globalcontrolmgr.h"
#include "uicommon.h"
#include <zeno/core/data.h>
#include <QTabWidget>


class BlockSignalScope
{
public:
    BlockSignalScope(QObject* pObj);
    ~BlockSignalScope();

private:
    QObject* m_pObject;
};

class VarToggleScope
{
public:
    VarToggleScope(bool* pbVar);
    ~VarToggleScope();

private:
    bool* m_pbVar;
};

class UiHelper
{
public:
    static QString createNewNode(QModelIndex subgIdx, const QString& descName, const QPointF& pt);
    static QPainterPath getRoundPath(QRectF r, int lt, int rt, int lb, int rb, bool bFixRadius);
    static QString generateUuid(const QString &name = "x");
    static uint generateUuidInt();
    static QVariant initDefaultValue(const zeno::ParamType& type);
    static QVariant parseTextValue(const zeno::ParamType& type, const QString& textValue);
    static QSizeF viewItemTextLayout(QTextLayout& textLayout, int lineWidth, int maxHeight = -1, int* lastVisibleLine = nullptr);
    static zeno::ParamControl getControlByType(const QString& type);
    static CONTROL_INFO getControlByType(const QString &nodeCls, bool bInput, const QString &socketName,const QString &socketType);    
    static void getSocketInfo(const QString& objPath, QString& subgName, QString& nodeIdent, QString& paramPath);
    static QList<zeno::ParamControl> getControlLists(const zeno::ParamType& type);
    static QStringList getAllControls();
    //there is no description on control, for example,  lineedit may be a integer, string or float.
    static zeno::ParamControl getControlByDesc(const QString& descName);
    static QString getControlDesc(zeno::ParamControl ctrl, zeno::ParamType type);
    static bool isFloatType(zeno::ParamType type);
    static bool qIndexSetData(const QModelIndex& index, const QVariant& value, int role);
    static QStringList getCoreTypeList();
    static bool parseVecType(const QString& type, int& dim, bool& bFloat);
    static QString variantToString(const QVariant& var);
    static QString constructObjPath(const QString& subgraph, const QString& node, const QString& group, const QString& sockName);
    static QString constructObjPath(const QString& subgraph, const QString& node, const QString& paramPath);
    static QString getSockNode(const QString& sockPath);
    static QString getSockName(const QString& sockPath);
    static QString getParamPath(const QString& sockPath);
    static QString getSockSubgraph(const QString& sockPath);
    static float parseJsonNumeric(const rapidjson::Value& val, bool castStr, bool& bSucceed);
    static float parseNumeric(const QVariant& val, bool castStr, bool& bSucceed);
    static QPointF parsePoint(const rapidjson::Value& ptObj, bool& bSucceed);

    static int getMaxObjId(const QList<QString>& lst);
    static QString getUniqueName(const QList<QString>& existNames, const QString& prefix, bool bWithBrackets = true);
    static QVector<qreal> getSlideStep(const QString& name, zeno::ParamType type);
    static QString nthSerialNumName(QString name);

    static QVariant parseJsonByType(const QString& type, const rapidjson::Value& val, QObject* parentRef);
    static QVariant parseVarByType(const QString& type, const QVariant& var, QObject* parentRef);
    static QVariant parseStringByType(const QString& defaultValue, zeno::ParamType type);
    static QVariant parseJsonByValue(const QString &type, const rapidjson::Value &val, QObject *parentRef);
    static QVariant parseJson(const rapidjson::Value& val, QObject* parentRef = nullptr);

    static QString gradient2colorString(const QLinearGradient& grad);
    static QVariant getParamValue(const QModelIndex& idx, const QString& name);
    static int tabIndexOfName(const QTabWidget* pTabWidget, const QString& name);
    static void getAllParamsIndex(const QModelIndex &nodeIdx,
                                  QModelIndexList& inputs,
                                  QModelIndexList& params,
                                  QModelIndexList& outputs,
                                  bool bEnsureSRCDST_lastKey = true);
    static QVector<qreal> scaleFactors();
    static QString UiHelper::getNaiveParamPath(const QModelIndex& param, int dim = -1);

    static QPair<zeno::NodesData, zeno::LinksData> dumpNodes(const QModelIndexList& nodeIndice, const QModelIndexList& linkIndice);
    static void reAllocIdents(const QString& targetSubgraph,
                               const zeno::NodesData& inNodes,
                               const zeno::LinksData& inLinks,
                               zeno::NodesData& outNodes,
                               zeno::LinksData& outLinks);
private:
    static std::pair<qreal, qreal> getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize);
};

extern const char* g_setKey;
#endif
