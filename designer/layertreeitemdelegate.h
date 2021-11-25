#ifndef __LAYER_TREEITEM_DELEGATE_H__
#define __LAYER_TREEITEM_DELEGATE_H__

class LayerTreeView;

class LayerTreeitemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    LayerTreeitemDelegate(QWidget* parent);
    void paint(QPainter* painter, const QStyleOptionViewItem& option,
        const QModelIndex& index) const;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const;

private:
    LayerTreeView *m_treeview;
};


#endif