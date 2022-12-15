#ifndef __LINK_MODEL_H__
#define __LINK_MODEL_H__

#include <QtWidgets>

class LinkModel : public QAbstractItemModel
{
    Q_OBJECT

    struct _linkItem
    {
        QString ident;
        QPersistentModelIndex fromSock;
        QPersistentModelIndex toSock;
    };

public:
    LinkModel(QObject* parent = nullptr);
    ~LinkModel();

    //QAbstractItemModel
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex& child) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    bool hasChildren(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QVariant headerData(int section, Qt::Orientation orientation,
        int role = Qt::DisplayRole) const override;
    bool setHeaderData(int section, Qt::Orientation orientation, const QVariant& value,
        int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    //LinkModel
    int addLink(const QModelIndex& fromSock, const QModelIndex& toSock);
    QModelIndex index(const QModelIndex& fromSock, const QModelIndex& toSock);
    void setInputSocket(const QModelIndex& index, const QModelIndex& sockIdx);
    void setOutputSocket(const QModelIndex& index, const QModelIndex& sockIdx);
    void clear();

private:
    QVector<_linkItem> m_items;
};


#endif