#ifndef __ZSUBNET_LISTITEM_DELEGATE_H__
#define __ZSUBNET_LISTITEM_DELEGATE_H__

#include <QtWidgets>

class GraphsModel;

class ZSubnetListItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    ZSubnetListItemDelegate(GraphsModel* model, QObject* parent = nullptr);
    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const;

private:
    GraphsModel* m_model;
};


#endif