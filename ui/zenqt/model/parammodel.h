#ifndef __PARAMS_MODEL_H__
#define __PARAMS_MODEL_H__

#include <QObject>
#include <QStandardItemModel>
#include <QSortFilterProxyModel>
#include <QQuickItem>
#include "uicommon.h"

class GraphModel;
class ParamModelImpl;
class CustomUIModel;

namespace zeno
{
    class NodeImpl;
}

struct ParamItem
{
    //BEGIN: temp cache on ui model, the actual value has been stored in m_wpParam.
    QString name;
    zeno::ParamType type = Param_Null;
    zeno::reflect::Any value;
    //END
    //std::weak_ptr<zeno::CoreParam> m_wpParam;

    bool bInput = true;
    zeno::ParamControl control = zeno::NullControl;
    zeno::SocketType connectProp = zeno::NoSocket;
    zeno::SocketProperty sockProp;
    zeno::NodeDataGroup group;
    zeno::reflect::Any optCtrlprops;
    QList<QPersistentModelIndex> links;
    bool bSocketVisible = true;
    bool bVisible = true;
    bool bEnable = true;
    bool bWildcard = false;
};

class ParamFilterModel : public QSortFilterProxyModel {
    Q_OBJECT
    QML_ELEMENT

public:
    ParamFilterModel(zeno::NodeDataGroup group, QObject* parent = nullptr);
    Q_INVOKABLE int indexFromName(const QString& name) const;

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const override;

private:
    zeno::NodeDataGroup m_group;
};

class ParamsModel : public QAbstractListModel
{
    Q_OBJECT
    QML_ELEMENT

public:
    ParamsModel(zeno::NodeImpl* spNode, QObject* parent = nullptr);
    ParamsModel(const zeno::CustomUI& customui);

    Q_PROPERTY(bool showPrimSocks READ getShowPrimSocks WRITE setShowPrimSocks NOTIFY showPrimSocks_changed)
    Q_PROPERTY(int numOfOutputPrims READ getNumOfOutputPrims NOTIFY numOfOutputPrims_changed)
    Q_PROPERTY(QString maxLengthName READ getMaxLengthName NOTIFY maxLengthName_changed)

    Q_INVOKABLE int indexFromName(const QString& name, bool bInput) const;
    Q_INVOKABLE QVariant getIndexList(bool bInput) const;
    Q_INVOKABLE QStandardItemModel* customParamModel();
    Q_INVOKABLE CustomUIModel* customUIModel();
    Q_INVOKABLE ParamFilterModel* inputObjects();
    Q_INVOKABLE ParamFilterModel* inputPrims();
    Q_INVOKABLE ParamFilterModel* outputPrims();
    Q_INVOKABLE ParamFilterModel* outputObjects();

    Qt::ItemFlags flags(const QModelIndex& index) const override;

    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QHash<int, QByteArray> roleNames() const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    void getDegrees(int& inDegrees, int& outDegrees);
    bool hasVisiblePrimParam() const;

    //api:
    void setNodeIdx(const QModelIndex& nodeIdx);
    QModelIndex paramIdx(const QString& name, bool bInput) const;
    void addLink(const QModelIndex& paramIdx, const QPersistentModelIndex& linkIdx);
    int removeLink(const QModelIndex& paramIdx);
    QModelIndex removeOneLink(const QModelIndex& paramIdx, const zeno::EdgeInfo& link);
    bool removeSpecificLink(const QModelIndex& paramIdx, const QModelIndex& linkIdx);
    void addParam(const ParamItem& param);
    void removeParam(const QString& name, bool bInput);
    GraphModel* getGraph() const;

    PARAMS_INFO getInputs();
    PARAMS_INFO getOutputs();
    zeno::CustomUI customUI() const;

    int getNumOfOutputPrims() const;

    //temp:

    void batchModifyParams(const zeno::ParamsUpdateInfo& params, bool bSubnetInit = false);
    void updateUiLinksSockets(zeno::params_change_info& changes);
    void resetCustomUi(const zeno::CustomUI& customui);

    int getParamlinkCount(const QModelIndex& paramIdx);
    int numOfInputParams() const;
    bool getShowPrimSocks() const;
    void setShowPrimSocks(bool) {}

signals:
    void linkAboutToBeInserted(const zeno::EdgeInfo& link);
    void linkAboutToBeRemoved(const zeno::EdgeInfo& link);
    void portTypeChanged(const QModelIndex& idx, bool bPrime);
    void enabledVisibleChanged();
    void showPrimSocks_changed();
    void numOfOutputPrims_changed();
    void maxLengthName_changed();

private:
    void initParamItems(const zeno::CustomUI& customui);
    void initCustomUI(const zeno::CustomUI& customui);
    void initProxyModels();
    void updateCustomUiModelIncremental(const zeno::params_change_info& params, const zeno::CustomUI& customui); //增量更新m_customParamsM，防止zenoproppanl接收不到数据
    GraphModel* parentGraph() const;
    QString getMaxLengthName() const;
    void updateParamData(const QString& name, const QVariant& val, int role, bool bInput = true);
    QStandardItemModel* constructProxyModel();

    QPersistentModelIndex m_nodeIdx;
    QVector<ParamItem> m_items;

    CustomUIModel* m_customUIM;

    ParamFilterModel* m_inObjProxy;
    ParamFilterModel* m_inPrimProxy;
    ParamFilterModel* m_outObjProxy;
    ParamFilterModel* m_outPrimProxy;

    zeno::NodeImpl* m_wpNode;    //直接用裸指针，反正如果核心没了这个model肯定也没了
    std::string cbUpdateParam;
    zeno::CustomUI m_tempUI;
    mutable bool m_bReentry = false;
    bool m_bTempModel = false;  //给CustomUI面板用的临时悬空模型，不依附于任何节点
};


#endif