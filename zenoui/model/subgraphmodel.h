#ifndef __ZENO_SUBGRAPH_MODEL_H__
#define __ZENO_SUBGRAPH_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QObject>
#include <memory>

#include "../model/modeldata.h"
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
    NODE_DATA itemData(const QModelIndex &index) const override;
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
    void removeNode(const QString& nodeid, bool enableTransaction = false);
    void removeNode(int row, bool enableTransaction = false);
    void removeLink(const EdgeInfo& info, bool enableTransaction = false);
    void addLink(const EdgeInfo& info, bool enableTransaction = false);
    void updateParam(const QString& nodeid, const QString& paramName, const QVariant& var, bool enableTransaction = false);
    QVariant getParamValue(const QString& nodeid, const QString& paramName);
    void setPos(const QString& nodeid, const QPointF& pt);
    void updateNodeState(const QString& nodeid, int role, const QVariant& newValue, bool enableTransaction = false);

    void beginTransaction(const QString& name);
    void endTransaction();

    void setName(const QString& name);
    void setViewRect(const QRectF& rc);
    QRectF viewRect() const { return m_rect; }
    QString name() const;
    NODE_DESCS descriptors();
    NODES_DATA dumpGraph();
    void clear();
    void reload();
    QUndoStack* undoStack() const;

signals:
    void linkChanged(bool bAdd, const QString& outputId, const QString& outputPort,
                const QString& inputId, const QString& inputPort);
    void paramUpdated(const QString& nodeid, const QString& paramName, const QVariant& val);
    void clearLayout();
    void reloaded();

public slots:
    void onDoubleClicked(const QString &nodename);
    void undo();
    void redo();
    void onParamValueChanged(const QString& nodeid, const QString& paramName, const QVariant &var);

private:
    bool _insertRow(int row, const NODE_DATA& nodeData, const QModelIndex &parent = QModelIndex());
    bool itemFromIndex(const QModelIndex &index, NODE_DATA& retNode) const;
    bool _removeRow(const QModelIndex &index);
    void _addLink(const EdgeInfo &info);
    void _removeLink(const EdgeInfo& info);

    QString m_name;
    QMap<QString, int> m_key2Row;
    QMap<int, QString> m_row2Key;
    QMap<QString, NODE_DATA> m_nodes; 
    
    QRectF m_rect;
    GraphsModel* m_pGraphsModel;
    QUndoStack* m_stack;
};

#endif
