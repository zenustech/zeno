#ifndef __SUBNET_TREEITEM_DELEGATE_H__
#define __SUBNET_TREEITEM_DELEGATE_H__

#include <QtWidgets>

class SubnetItemDelegated : public QStyledItemDelegate
{
	Q_OBJECT
	typedef QStyledItemDelegate _base;
public:
	SubnetItemDelegated(QWidget* parent);
	void setModelData(QWidget* editor,
		QAbstractItemModel* model,
		const QModelIndex& index) const;
	void paint(QPainter* painter, const QStyleOptionViewItem& option,
		const QModelIndex& index) const;
	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const;

protected:
	void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const;

private:
	void drawExpandArrow(QPainter* painter, const QStyleOptionViewItem& option) const;
};


#endif