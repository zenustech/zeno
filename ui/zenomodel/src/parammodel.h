#ifndef __PARAM_MODEL_H__
#define __PARAM_MODEL_H__

#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>

class IGraphsModel;
class DictKeyModel;

class IParamModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;

public:
    struct _ItemInfo
    {
        QString name;
        QString type;
        QVariant pConst;    //const (default) value on socket or param.
        //CurveModel* pVar;   //variable on time frame.
        PARAM_LINKS links;
        QMap<int, QVariant> customData;
        int prop;
    };

    explicit IParamModel(
        PARAM_CLASS paramType,
        IGraphsModel* pModel,
        const QPersistentModelIndex& subgIdx,
        const QPersistentModelIndex& nodeIdx,
        QObject* parent = nullptr);
    ~IParamModel();

    bool getInputSockets(INPUT_SOCKETS& inputs);
    bool getOutputSockets(OUTPUT_SOCKETS& outputs);
    bool getParams(PARAMS_INFO& params);
    void setInputSockets(const INPUT_SOCKETS& inputs);
    void setParams(const PARAMS_INFO& params);
    void setOutputSockets(const OUTPUT_SOCKETS& outputs);

    //QAbstractItemModel
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex& child) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    bool hasChildren(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    void appendRow(
        const QString& sockName,
        const QString& type = "",
        const QVariant& deflValue = QVariant(),
        int prop = SOCKPROP_NORMAL);

    void setItem(
        const QModelIndex& idx,
        const QString& type,
        const QVariant& deflValue,
        const PARAM_LINKS& links = PARAM_LINKS());

    bool removeLink(const QString& sockName, const QModelIndex& linkIdx);

    QStringList sockNames() const;

    QModelIndex index(const QString& name) const;

    PARAM_CLASS paramClass() const;

    void clear();

private slots:
    void onKeyItemAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    QString nameFromRow(int row) const;

    void insertRow(int row,
        const QString& sockName,
        const QString& type = "",
        const QVariant& deflValue = QVariant(),
        int prop = SOCKPROP_NORMAL);

    bool _insertRow(int row,
        const QString& name,
        const QString& type = "",
        const QVariant& deflValue = QVariant(),
        int prop = SOCKPROP_NORMAL);
    bool _removeRow(const QModelIndex& index);
    void onSubIOEdited(const QVariant& value, const _ItemInfo& item);
    void exportDictkeys(DictKeyModel* pModel, DICTPANEL_INFO& panel);
    QList<EdgeInfo> exportLinks(const PARAM_LINKS& links);
    EdgeInfo exportLink(const QModelIndex& linkIdx);

    const QPersistentModelIndex m_nodeIdx;
    const QPersistentModelIndex m_subgIdx;

    const PARAM_CLASS m_class;
    QMap<QString, int> m_key2Row;
    QMap<int, QString> m_row2Key;
    QMap<QString, _ItemInfo> m_items;
    IGraphsModel* m_model;

    //only used to sync to all view param, it will not be a reference to a control on view.
    PARAM_CONTROL m_tempControl;

    bool m_bRetryLinkOp;
};

#endif