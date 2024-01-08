#ifndef __SUBNET_TREEITEM_DELEGATE_H__
#define __SUBNET_TREEITEM_DELEGATE_H__

#include <QtWidgets>

class SubnetItemDelegated : public QStyledItemDelegate
{
	Q_OBJECT
	typedef QStyledItemDelegate _base;
public:
	SubnetItemDelegated(QWidget* parent);
	void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
	void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

protected:
	void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const override;

private:
	void drawExpandArrow(QPainter* painter, const QStyleOptionViewItem& option) const;
};


#endif
