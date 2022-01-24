#ifndef __ZSUBNET_LISTITEM_DELEGATE_H__
#define __ZSUBNET_LISTITEM_DELEGATE_H__

#include <QtWidgets>

class GraphsModel;

class ZSubnetListItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    ZSubnetListItemDelegate(GraphsModel* model, QObject* parent = nullptr);
    ~ZSubnetListItemDelegate();

    QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    void setEditorData(QWidget* editor, const QModelIndex& index) const override;
    void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
    void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const  QModelIndex& index) const override;

    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const;
    bool editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option,
        const QModelIndex& index) override;

private slots:
    void onDelete(const QModelIndex& index);

private:
    GraphsModel* m_model;
};


#endif