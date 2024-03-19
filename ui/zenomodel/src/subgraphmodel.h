#ifndef __ZENO_SUBGRAPH_MODEL_H__
#define __ZENO_SUBGRAPH_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QObject>
#include <memory>
#include "command.h"
#include "parammodel.h"
#include "viewparammodel.h"
#include "nodeparammodel.h"
#include "panelparammodel.h"


class GraphsModel;

class SubGraphModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;
    friend class AddNodeCommand;

    struct _LabelItem
    {
        QString name;
        QPersistentModelIndex outSock;
        QList<QPersistentModelIndex> inSocks;
    };
    typedef QHash<QString, _LabelItem> LABELS_SET;

    struct _NodeItem
    {
        QString objid;
        QString objCls;
        QString customName;
        NODE_TYPE type;
        QPointF viewpos;
        int options;
        PARAMS_INFO paramNotDesc;

        PanelParamModel* panelParams;
        NodeParamModel* nodeParams;

        bool bCollasped;
        bool bDataChanged;      //changed before next running.

        _NodeItem()
            : options(0)
            , bCollasped(false)
            , bDataChanged(false)
            , type(NORMAL_NODE)
            , panelParams(nullptr)
            , nodeParams(nullptr)
        {
        }
    };

public:
	explicit SubGraphModel(GraphsModel* pGraphsModel, QObject* parent = nullptr);
	~SubGraphModel();

	//QAbstractItemModel
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
	QModelIndex parent(const QModelIndex& child) const override;
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    bool hasChildren(const QModelIndex &parent = QModelIndex()) const override;
	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
	QModelIndexList match(const QModelIndex &start, int role,
                          const QVariant &value, int hits = 1,
                          Qt::MatchFlags flags =
                          Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    QModelIndexList getNodesByCls(const QString& nodeCls);
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;

    //SubGraphModel
    bool insertRow(int row, const NODE_DATA &nodeData, const QModelIndex &parent = QModelIndex());
    QModelIndex index(QString id, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex index(uint32_t id) const;
    void appendItem(const NODE_DATA& nodeData, bool enableTransaction = false);
    void removeNode(const QString& nodeid, bool enableTransaction = false);
    void removeNode(int row, bool enableTransaction = false);
    void removeNodeByDescName(const QString& descName);

    QVariant getParamValue(const QString& nodeid, const QString& paramName);

    NODE_DATA nodeData(const QModelIndex &index) const;

    void updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info);
    SubGraphModel* clone(GraphsModel* parent);
    GraphsModel* getGraphsModel() const { return m_pGraphsModel; }
    QModelIndex nodeParamIndex(const QModelIndex &nodeIdx, PARAM_CLASS cls, const QString &paramName) const;;
    ViewParamModel* viewParams(const QModelIndex& index);
    ViewParamModel* nodeParams(const QModelIndex& index);

    bool setParamValue(
        PARAM_CLASS cls,
        const QModelIndex& idx,
        const QString& sockName,
        const QVariant& value,
        const QString& type = "",
        PARAM_CONTROL ctrl = CONTROL_NONE,
        SOCKET_PROPERTY prop = SOCKPROP_NORMAL);

    QString name() const;
    void setName(const QString& name);

    SUBGRAPH_TYPE type() const;
    void setType(SUBGRAPH_TYPE type);
    void setMtlid(const QString& mtlid);
    QString mtlid();
    void setForkLock(bool bLocked);
    bool forkLocked();

    void replaceSubGraphNode(const QString& oldName, const QString& newName);
    void setViewRect(const QRectF& rc);
    QRectF viewRect() const { return m_rect; }

    void clear();
    void onModelInited();
    void collaspe();
    void expand();

    //net labels
    bool addNetLabel(const QModelIndex& sock, const QString& name, bool bInput);
    void updateNetLabel(const QModelIndex& trigger, const QString& oldName, const QString& newName);
    void removeNetLabel(const QModelIndex& trigger, const QString& name);
    QModelIndex getNetOutput(const QString& name) const;
    QStringList dumpLabels() const;
    QModelIndexList getNetInputSocks(const QString& name) const;
    void setCommandParam(const QModelIndex& sock, bool bMarked);

public slots:
    void onDoubleClicked(const QString &nodename);

private:
    SubGraphModel(const SubGraphModel& rhs);

    bool _insertNode(int row, const NODE_DATA& nodeData, const QModelIndex &parent = QModelIndex());
    bool itemFromIndex(const QModelIndex& index, _NodeItem& retNode) const;
    bool _removeRow(const QModelIndex &index);
    NODE_DATA item2NodeData(const _NodeItem& item) const;
    void importNodeItem(const NODE_DATA& data, const QModelIndex& nodeIdx, _NodeItem& ret);
    bool checkCustomName(const QString &name);
    void _removeNetLabels(const NodeParamModel* nodeParams);
    void _uniqueView(const QModelIndex& index, bool bInSocket, bool bOutSocket, QModelIndexList& viewLst);

    QString m_name;
    SUBGRAPH_TYPE m_type;
    QString m_mtlid;
    bool m_bForkLocked;
    QHash<QString, int> m_key2Row;
    QHash<int, QString> m_row2Key;
    QHash<QString, _NodeItem> m_nodes;

    QHash<QString, QSet<QString>> m_name2identLst;

    QHash<uint32_t, QString> m_num2strId;
    QHash<QString, uint32_t> m_str2numId;

    LABELS_SET m_labels;

    QRectF m_rect;
    GraphsModel* m_pGraphsModel;
    QUndoStack* m_stack;
};

#endif
