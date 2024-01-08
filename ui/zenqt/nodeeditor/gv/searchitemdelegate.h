#ifndef __SEARCHITEM_DELEGATE_H__
#define __SEARCHITEM_DELEGATE_H__

#include <QtWidgets>

class QAbstractItemView;

class SearchItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    SearchItemDelegate(const QString& search, QObject* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const override;

private:
    QAbstractItemView* m_view;
    const QString m_search;
};

#endif
