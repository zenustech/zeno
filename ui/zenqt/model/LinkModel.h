#ifndef __LINKMODEL_H__
#define __LINKMODEL_H__

#include <QtWidgets>
#include <QUuid>
#include <zeno/core/common.h>

class LinkModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;

    struct _linkItem {
        QPersistentModelIndex fromParam;
        QString fromKey;        //need to be updated when renaming key.
        QPersistentModelIndex toParam;
        QString toKey;
        QUuid uuid;
        bool bObjLink = true;
        bool m_bCollasped = false;
    };

    typedef QVector<_linkItem> LINKS_ITEM;

public:
    LinkModel(QObject* parent = nullptr);
    ~LinkModel();

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QHash<int, QByteArray> roleNames() const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    //api:
    QModelIndex addLink(const QModelIndex& fromParam, const QString& fromKey,
        const QModelIndex& toParam, const QString& toKey, bool bObjLink);
private:
    LINKS_ITEM m_items;
};


#endif