#ifndef __ZCOMBOBOX_ITEM_DELEGATE_H__
#define __ZCOMBOBOX_ITEM_DELEGATE_H__

#include <QtWidgets>

class ZComboBoxItemDelegate2 : public QStyledItemDelegate
{
    Q_OBJECT
public:
    ZComboBoxItemDelegate2(QObject* parent = nullptr);
    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

#endif