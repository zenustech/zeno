#ifndef __UI_HELPER_H__
#define __UI_HELPER_H__

#include <zenomodel/include/modeldata.h>
#include <rapidjson/document.h>
#include <zenomodel/include/igraphsmodel.h>


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
    static QVariant initDefaultValue(const QString& type);
    static bool validateVariant(const QVariant& var, const QString& type);
    static QVariant parseTextValue(PARAM_CONTROL editCtrl, const QString& textValue);
    static QSizeF viewItemTextLayout(QTextLayout& textLayout, int lineWidth, int maxHeight = -1, int* lastVisibleLine = nullptr);
    static PARAM_CONTROL getControlType(const QString& type);
    static bool parseVecType(const QString& type, int& dim, bool& bFloat);
    static QString variantToString(const QVariant& var);
    static float parseJsonNumeric(const rapidjson::Value& val, bool castStr, bool& bSucceed);
    static float parseNumeric(const QVariant& val, bool castStr, bool& bSucceed);
    static QVariant initVariantByControl(PARAM_CONTROL ctrl);
    static QPointF parsePoint(const rapidjson::Value& ptObj, bool& bSucceed);
    static NODE_TYPE nodeType(const QString& name);

    //todo: place at other helper.
    static int getMaxObjId(const QList<QString>& lst);
    static QString getUniqueName(const QList<QString>& existNames, const QString& prefix);
    static QVector<qreal> getSlideStep(const QString& name, PARAM_CONTROL ctrl);
    static void reAllocIdents(QMap<QString, NODE_DATA>& nodes, QList<EdgeInfo>& links, const QMap<QString, NODE_DATA>& oldGraphsToNew);
    static QString nthSerialNumName(QString name);
    static QString correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput);

    static QVariant parseJsonByType(const QString& type, const rapidjson::Value& val, QObject* parentRef);
    static QVariant parseVarByType(const QString& type, const QVariant& var, QObject* parentRef);
    static QVariant parseStringByType(const QString &defaultValue, const QString &type);
    static QVariant parseJsonByValue(const QString &type, const rapidjson::Value &val, QObject *parentRef);

    static QString gradient2colorString(const QLinearGradient& grad);
    static QVariant getParamValue(const QModelIndex& idx, const QString& name);
    static int UiHelper::tabIndexOfName(const QTabWidget* pTabWidget, const QString& name);

private:
    static std::pair<qreal, qreal> getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize);
};

#endif
