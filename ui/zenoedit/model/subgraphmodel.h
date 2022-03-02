#ifndef __ZENO_SUBGRAPH_MODEL_H__
#define __ZENO_SUBGRAPH_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QObject>
#include <memory>

#include <zenoui/model/modeldata.h>
#include "command.h"

struct PlainNodeItem
{
    void setData(const QVariant &value, int role) {
        m_datas[role] = value;
    }

    QVariant data(int role) const {
        auto it = m_datas.find(role);
        if (it == m_datas.end())
            return QVariant();
        return it.value();
    }

    NODE_DATA m_datas;
};

//typedef std::shared_ptr<PlainNodeItem> NODEITEM_PTR;

class GraphsModel;

class SubGraphModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;
    friend class AddNodeCommand;

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
	QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;
    bool setHeaderData(int section, Qt::Orientation orientation, const QVariant &value,
                       int role = Qt::EditRole) override;
	QModelIndexList match(const QModelIndex &start, int role,
                          const QVariant &value, int hits = 1,
                          Qt::MatchFlags flags =
                          Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    QHash<int, QByteArray> roleNames() const override;
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;

    //SubGraphModel
    bool insertRow(int row, const NODE_DATA &nodeData, const QModelIndex &parent = QModelIndex());
    QModelIndex index(QString id, const QModelIndex &parent = QModelIndex()) const;
    void appendItem(const NODE_DATA& nodeData, bool enableTransaction = false);
    void appendNodes(const QList<NODE_DATA>& nodes, bool enableTransaction = false);
    void removeNode(const QString& nodeid, bool enableTransaction = false);
    void removeNode(int row, bool enableTransaction = false);
    void removeNodeByDescName(const QString& descName);

    void updateParam(const QString& nodeid, const QString& paramName, const QVariant& var, bool enableTransaction = false);
    QVariant getParamValue(const QString& nodeid, const QString& paramName);

    void updateSocket(const QString& nodeid, const SOCKET_UPDATE_INFO& info);
    void updateSocketDefl(const QString& nodeid, const PARAM_UPDATE_INFO& info);
    //it's not good programming pratice to expose NODE_DATA as it break the encapsulation.
    NODE_DATA itemData(const QModelIndex &index) const override;

    QVariant getNodeStatus(const QString& nodeid, int role);
    void updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info);
    SubGraphModel* clone(GraphsModel* parent);
    GraphsModel* getGraphsModel() const { return m_pGraphsModel; }

    void beginMacro(const QString& name);
    void endMacro();

    QString name() const;
    void setName(const QString& name);

    void replaceSubGraphNode(const QString& oldName, const QString& newName);
    void setViewRect(const QRectF& rc);
    QRectF viewRect() const { return m_rect; }

    NODES_DATA nodes();
    void clear();
    void reload();
    void onModelInited();
    void setInputSocket(const QString& id, const QString& inSock, const QString& outId, const QString& outSock, const QVariant& defaultValue);

public slots:
    void onDoubleClicked(const QString &nodename);
    void undo();
    void redo();

private:
    SubGraphModel(const SubGraphModel& rhs);

    bool _insertRow(int row, const NODE_DATA& nodeData, const QModelIndex &parent = QModelIndex());
    bool itemFromIndex(const QModelIndex &index, NODE_DATA& retNode) const;
    bool _removeRow(const QModelIndex &index);

    QString m_name;
    QMap<QString, int> m_key2Row;
    QMap<int, QString> m_row2Key;
    QMap<QString, NODE_DATA> m_nodes;

    QRectF m_rect;
    GraphsModel* m_pGraphsModel;
    QUndoStack* m_stack;
};

#endif
