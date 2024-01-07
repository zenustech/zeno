#ifndef __NODESMODEL_H__
#define __NODESMODEL_H__

#include <QObject>
#include <QAbstractItemModel>
#include <QAbstractListModel>
#include <QString>
#include <QQuickItem>
#include "ParamsModel.h"
#include "LinkModel.h"

class GraphModel;

struct NodeItem : public QObject
{
    Q_OBJECT

public:
    QString ident;
    QString name;
    ParamsModel* params = nullptr;
    QPointF pos;

    //for subgraph:
    GraphModel* pSubgraph = nullptr;

    NodeItem(QObject* parent) : QObject(parent) {}
};

//为什么不base StandardModel，是因为StandardItem本身还得挂载一个模型，有点冗余，干脆自己实现一个图treemodel.
class GraphModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;

    //Q_PROPERTY(CONTROL_TYPE control READ getControl WRITE setControl NOTIFY control_changed)

    QML_ELEMENT

public:
    GraphModel(const QString& graphName, QObject* parent = nullptr);
    ~GraphModel();
    Q_INVOKABLE LinkModel* getLinkModel() const { return m_linkModel; }
    Q_INVOKABLE int indexFromId(const QString& ident) const;
    Q_INVOKABLE void addLink(const QString& fromNodeStr, const QString& fromParamStr,
        const QString& toNodeStr, const QString& toParamStr);
    Q_INVOKABLE QVariant removeLink(const QString& nodeIdent, const QString& paramName, bool bInput);

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

    //NodesModel:
    void appendNode(QString ident, QString name, const QPointF& pos);
    void appendSubgraphNode(QString ident, QString name, NODE_DESCRIPTOR desc, GraphModel* subgraph, const QPointF& pos);
    void removeNode(QString ident);
    void addLink(QPair<QString, QString> fromParam, QPair<QString, QString> toParam);

    //test functions:
    void updateParamName(QModelIndex nodeIdx, int row, QString newName);
    void removeParam(QModelIndex nodeIdx, int row);
    void removeLink(int row);
    ParamsModel* params(QModelIndex nodeIdx);
    GraphModel* subgraph(QModelIndex nodeIdx);

private:
    QModelIndex nodeIdx(const QString& ident) const;

    QString m_graphName;
    QHash<QString, int> m_id2Row;
    QHash<int, QString> m_row2id;
    QHash<QString, NodeItem*> m_nodes;

    LinkModel* m_linkModel;
};

#endif