#ifndef __LAYER_TREEITEM_DELEGATE_H__
#define __LAYER_TREEITEM_DELEGATE_H__

class LayerTreeView;

class LayerTreeitemDelegate : public QStyledItemDelegate
{
    typedef QStyledItemDelegate _base;
    Q_OBJECT
public:
    LayerTreeitemDelegate(QWidget* parent);
    void paint(QPainter* painter, const QStyleOptionViewItem& option,
        const QModelIndex& index) const;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const;
    bool editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option,
                             const QModelIndex &index) override;

private:
    QRect getLockRect(const QStyleOptionViewItem *option) const;
    QRect getViewRect(const QStyleOptionViewItem *option) const;

    LayerTreeView *m_treeview;
};


#endif