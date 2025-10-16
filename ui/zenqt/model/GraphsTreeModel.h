#ifndef __GRAPHSTREEMODEL_H__
#define __GRAPHSTREEMODEL_H__

#include <QObject>
#include <QStandardItemModel>
#include <QString>
#include <QQuickItem>
#include "GraphModel.h"


//为什么不base StandardModel，是因为StandardItem有点冗余，干脆自己实现一个简易的图treemodel.
class GraphsTreeModel : public QStandardItemModel
{
    Q_OBJECT
    typedef QStandardItemModel _base;
    QML_ELEMENT

public:
    GraphsTreeModel(QObject* parent = nullptr);
    ~GraphsTreeModel();
    void init(GraphModel* mainModel);

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

    Q_INVOKABLE GraphModel* getGraphByPath(const QStringList& objPath);

    Q_INVOKABLE QModelIndex getIndexByPath(const QStringList& objPath);

    Q_INVOKABLE QModelIndex getIndexByUuidPath(const zeno::ObjPath& objPath);

    //util methods
    bool isDirty() const;
    void clearDirty();
    inline void markDirty(bool dirty) {
        if (m_dirty != dirty) {
            m_dirty = dirty;
            emit dirtyChanged();
        }
    }
    QList<SEARCH_RESULT> search(const QString& content, int searchType, int searchOpts) const;
    QList<SEARCH_RESULT> searchByUuidPath(const zeno::ObjPath& uuidPath);

signals:
    void dirtyChanged();
    void modelClear();

public slots:
    void onGraphRowsInserted(const QModelIndex& parent, int first, int last);
    void onGraphRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void onGraphRowsRemoved(const QModelIndex& parent, int first, int last);
    void onNameUpdated(const QModelIndex& nodeIdx, const QString& oldName);

private:
    QStandardItem* initNodeItem(const QModelIndex& nodeidx) const;

    GraphModel* m_main;
    bool m_dirty;
};

#endif