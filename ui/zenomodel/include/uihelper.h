#ifndef __UI_HELPER_H__
#define __UI_HELPER_H__

#include <zenomodel/include/modeldata.h>
#include <rapidjson/document.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/globalcontrolmgr.h>


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
    static NODE_DESCS parseDescs(const rapidjson::Value &descs);
    static QPainterPath getRoundPath(QRectF r, int lt, int rt, int lb, int rb, bool bFixRadius);
    static QString generateUuid(const QString &name = "x");
    static uint generateUuidInt();
    static QVariant initDefaultValue(const QString& type);
    static bool validateVariant(const QVariant& var, const QString& type);
    static QVariant parseTextValue(PARAM_CONTROL editCtrl, const QString& textValue);
    static QSizeF viewItemTextLayout(QTextLayout& textLayout, int lineWidth, int maxHeight = -1, int* lastVisibleLine = nullptr);
    static PARAM_CONTROL getControlByType(const QString& type);
    static CONTROL_INFO getControlByType(const QString &nodeCls, PARAM_CLASS cls, const QString &socketName,const QString &socketType);    
    static QString getTypeByControl(PARAM_CONTROL ctrl);
    static void getSocketInfo(const QString& objPath, QString& subgName, QString& nodeIdent, QString& paramPath);
    static QStringList getControlLists(const QString& type, bool isNodeUI);
    static QStringList getAllControls();
    static QString getControlDesc(PARAM_CONTROL ctrl);
    static PARAM_CONTROL getControlByDesc(const QString& descName);
    static QStringList getCoreTypeList();
    static PARAM_CONTROL getControlType(const QString& type, const QString& sockName);
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
    static QVariant initVariantByControl(PARAM_CONTROL ctrl);
    static QPointF parsePoint(const rapidjson::Value& ptObj, bool& bSucceed);
    static NODE_TYPE nodeType(const QString& name);

    static int getMaxObjId(const QList<QString>& lst);
    static QString getUniqueName(const QList<QString>& existNames, const QString& prefix, bool bWithBrackets = true);
    static QVector<qreal> getSlideStep(const QString& name, PARAM_CONTROL ctrl);
    static QString nthSerialNumName(QString name);
    static QString correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput);

    static QVariant parseJsonByType(const QString& type, const rapidjson::Value& val, QObject* parentRef);
    static QVariant parseVarByType(const QString& type, const QVariant& var, QObject* parentRef);
    static QVariant parseStringByType(const QString &defaultValue, const QString &type);
    static QVariant parseJsonByValue(const QString &type, const rapidjson::Value &val, QObject *parentRef);
    static QVariant parseJson(const rapidjson::Value& val, QObject* parentRef = nullptr);

    static QString gradient2colorString(const QLinearGradient& grad);
    static QVariant getParamValue(const QModelIndex& idx, const QString& name);
    static int tabIndexOfName(const QTabWidget* pTabWidget, const QString& name);
    static QModelIndex findSubInOutputIdx(IGraphsModel *pModel, bool bSubInput, const QString &paramName,
                                          const QModelIndex &subgIdx);
    static void getAllParamsIndex(const QModelIndex &nodeIdx,
                                  QModelIndexList& inputs,
                                  QModelIndexList& params,
                                  QModelIndexList& outputs,
                                  bool bEnsureSRCDST_lastKey = true);
    static QVector<qreal> scaleFactors();

    static QPair<NODES_DATA, LINKS_DATA> dumpNodes(const QModelIndexList& nodeIndice, const QModelIndexList& linkIndice);
    static void reAllocIdents(const QString& targetSubgraph,
                               const NODES_DATA& inNodes,
                               const LINKS_DATA& inLinks,
                               NODES_DATA& outNodes,
                               LINKS_DATA& outLinks);
    static QString nativeWindowTitle(const QString& currentFilePath);

private:
    static std::pair<qreal, qreal> getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize);
};

#endif
