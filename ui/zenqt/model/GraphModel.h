#ifndef __NODESMODEL_H__
#define __NODESMODEL_H__

#include <QObject>
#include <QAbstractItemModel>
#include <QAbstractListModel>
#include <QString>
#include <QQuickItem>
#include "parammodel.h"
#include "LinkModel.h"
#include <zeno/core/Graph.h>
#include <optional>

class GraphModel;
class GraphsTreeModel;

struct NodeItem : public QObject
{
    Q_OBJECT

public:
    //temp cached data for spNode->...
    QString name;
    QString cls;
    QPointF pos;

    std::string m_cbSetPos;
    std::string m_cbSetStatus;

    std::weak_ptr<zeno::INode> m_wpNode;
    ParamsModel* params = nullptr;
    zeno::NodeStatus status = zeno::None;

    //for subgraph, but not include assets:
    std::optional<GraphModel*> optSubgraph;

    NodeItem(QObject* parent);
    ~NodeItem();
    void init(GraphModel* pGraphM, std::shared_ptr<zeno::INode> spNode);
    QString getName() {
        return name;
    }

private:
    void unregister();
};

class GraphModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;

    //Q_PROPERTY(CONTROL_TYPE control READ getControl WRITE setControl NOTIFY control_changed)

    QML_ELEMENT

public:
    GraphModel(std::shared_ptr<zeno::Graph> spGraph, GraphsTreeModel* pTree, QObject* parent = nullptr);
    ~GraphModel();
    Q_INVOKABLE LinkModel* getLinkModel() const { return m_linkModel; }
    Q_INVOKABLE int indexFromId(const QString& name) const;
    Q_INVOKABLE void addLink(const QString& fromNodeStr, const QString& fromParamStr,
        const QString& toNodeStr, const QString& toParamStr);
    Q_INVOKABLE QVariant removeLink(const QString& nodeName, const QString& paramName, bool bInput);
    Q_INVOKABLE QString name() const;

    //TEST API
    Q_INVOKABLE QString owner() const;

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    QHash<int, QByteArray> roleNames() const override;

    //GraphModel:
    zeno::NodeData createNode(const QString& nodeCls, const QString& cate, const QPointF& pos);
    void appendSubgraphNode(QString name, QString cls, NODE_DESCRIPTOR desc, GraphModel* subgraph, const QPointF& pos);
    bool removeNode(const QString& name);
    QString updateNodeName(const QModelIndex& idx, QString newName);
    void addLink(const zeno::EdgeInfo& link);
    QList<SEARCH_RESULT> search(const QString& content, SearchType searchType, SearchOpt searchOpts) const;
    GraphModel* getGraphByPath(const QStringList& objPath);
    QStringList currentPath() const;
    //QModelIndex index(const QString& name) const;
    QModelIndex indexFromName(const QString& name) const;
    void undo();
    void redo();
    void beginTransaction(const QString& name);
    void endTransaction();

    //test functions:
    void updateParamName(QModelIndex nodeIdx, int row, QString newName);
    void removeParam(QModelIndex nodeIdx, int row);
    void removeLink(const QModelIndex& linkIdx);
    void removeLink(const zeno::EdgeInfo& link);
    ParamsModel* params(QModelIndex nodeIdx);
    GraphModel* subgraph(QModelIndex nodeIdx);
    GraphsTreeModel* treeModel() const;

signals:
    void reloaded();
    void clearLayout();
    void nameUpdated(const QModelIndex& nodeIdx, const QString& oldName);

private:
    void registerCoreNotify();
    void unRegisterCoreNotify();
    QModelIndex nodeIdx(const QString& name) const;
    void _appendNode(std::shared_ptr<zeno::INode> spNode);
    void _addLink(QPair<QString, QString> fromParam, QPair<QString, QString> toParam);
    bool _removeLink(const zeno::EdgeInfo& edge);
    void _updateName(const QString& oldName, const QString& newName);
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    QString m_graphName;
    QHash<QString, int> m_name2Row;
    QHash<int, QString> m_row2name;
    QHash<QString, NodeItem*> m_nodes;

    std::weak_ptr<zeno::Graph> m_wpCoreGraph;

    std::string m_cbCreateNode;
    std::string m_cbRemoveNode;
    std::string m_cbRenameNode;
    std::string m_cbAddLink;
    std::string m_cbRemoveLink;

    GraphsTreeModel* m_pTree;
    LinkModel* m_linkModel;

    friend class NodeItem;
};

#endif