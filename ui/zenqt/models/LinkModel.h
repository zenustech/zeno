#ifndef __LINKMODEL_H__
#define __LINKMODEL_H__

#include <QtWidgets>

class LinkModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;

    struct _linkItem {
        QPersistentModelIndex fromParam;
        QPersistentModelIndex toParam;
    };

    typedef QVector<_linkItem> LINKS_ITEM;

public:
    LinkModel(QObject* parent = nullptr);
    ~LinkModel();

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    //api:
    QModelIndex addLink(const QModelIndex& fromParam, const QModelIndex& toParam);
private:
    LINKS_ITEM m_items;
};


#endif