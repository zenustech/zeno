#ifndef __GRAPHSTREEMODEL_H__
#define __GRAPHSTREEMODEL_H__

#include <QObject>
#include <QAbstractItemModel>
#include <QString>
#include <QQuickItem>
#include "graphmodel.h"


//为什么不base StandardModel，是因为StandardItem有点冗余，干脆自己实现一个简易的图treemodel.
class GraphsTreeModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;
    QML_ELEMENT

public:
    GraphsTreeModel(QObject* parent = nullptr);
    ~GraphsTreeModel();
    void init(GraphModel* mainModel);

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
    QHash<int, QByteArray> roleNames() const override;

    //适配TreeView.qml
    //! Return the depth for the given index
    Q_INVOKABLE int depth(const QModelIndex& index) const;

    // 返回当前索引节点所在的图模型
    Q_INVOKABLE GraphModel* graph(const QModelIndex& index) const;

    Q_INVOKABLE QString name(const QModelIndex& index) const;

    //! Clear the model.
    Q_INVOKABLE void clear();

    /*!
    *  Return the root item to the QML Side.
    *  This method is not meant to be used in client code.
    */
    Q_INVOKABLE QModelIndex rootIndex();

    Q_INVOKABLE GraphModel* getGraphByPath(const QString& objPath);

    Q_INVOKABLE QModelIndex getIndexByPath(const QString& objPath);

    //util methods
    bool isDirty() const;
    void clearDirty();
    QList<SEARCH_RESULT> search(const QString& content, int searchType, int searchOpts) const;

signals:
    void dirtyChanged();
    void modelClear();

public slots:
    void onGraphRowsInserted(const QModelIndex& parent, int first, int last);
    void onGraphRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void onGraphRowsRemoved(const QModelIndex& parent, int first, int last);
    void onNameUpdated(const QModelIndex& nodeIdx, const QString& oldName);

private:
    QModelIndex innerIndex(const QModelIndex& treeIdx) const;

    GraphModel* m_main;
};

#endif