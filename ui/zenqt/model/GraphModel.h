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

struct NodeItem : public QObject
{
    Q_OBJECT

public:
    //QString name;
    //QString cls;
    ParamsModel* params = nullptr;
    QPointF pos;

    std::weak_ptr<zeno::INode> spNode;

    //for subgraph:
    std::optional<GraphModel*> optSubgraph;
    //GraphModel* pSubgraph = nullptr;

    NodeItem(QObject* parent) : QObject(parent) {}
    QString getName() {
        std::shared_ptr<zeno::INode> spCoreNode = spNode.lock();
        return spCoreNode ? QString::fromStdString(spCoreNode->name) : "";
    }

};

class GraphModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;

    //Q_PROPERTY(CONTROL_TYPE control READ getControl WRITE setControl NOTIFY control_changed)

    QML_ELEMENT

public:
    GraphModel(std::shared_ptr<zeno::Graph> spGraph, QObject* parent = nullptr);
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
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QHash<int, QByteArray> roleNames() const override;

    //GraphModel:
    void registerCoreNotify(std::shared_ptr<zeno::Graph> coreGraph);
    zeno::NodeData createNode(const QString& nodeCls, const QPointF& pos);
    void appendSubgraphNode(QString name, QString cls, NODE_DESCRIPTOR desc, GraphModel* subgraph, const QPointF& pos);
    void removeNode(QString name);
    void addLink(const zeno::EdgeInfo& link);
    QList<SEARCH_RESULT> search(const QString& content, SearchType searchType, SearchOpt searchOpts) const;
    GraphModel* getGraphByPath(const QString& objPath);
    //QModelIndex index(const QString& name) const;
    QModelIndex indexFromName(const QString& name) const;
    void undo();
    void redo();
    void beginTransaction(const QString& name);
    void endTransaction();

    //test functions:
    void updateParamName(QModelIndex nodeIdx, int row, QString newName);
    void removeParam(QModelIndex nodeIdx, int row);
    void removeLink(int row);
    ParamsModel* params(QModelIndex nodeIdx);
    GraphModel* subgraph(QModelIndex nodeIdx);

signals:
    void reloaded();
    void clearLayout();

private:
    QModelIndex nodeIdx(const QString& name) const;
    void _appendNode(std::shared_ptr<zeno::INode> spNode);
    void _addLink(QPair<QString, QString> fromParam, QPair<QString, QString> toParam);

    QString m_graphName;
    QHash<QString, int> m_name2Row;
    QHash<int, QString> m_row2name;
    QHash<QString, NodeItem*> m_nodes;

    std::weak_ptr<zeno::Graph> m_spCoreGraph;

    std::string cbCreateNode;

    LinkModel* m_linkModel;
};

#endif