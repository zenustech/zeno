#ifndef __ZSUBNET_LISTITEM_DELEGATE_H__
#define __ZSUBNET_LISTITEM_DELEGATE_H__

#include <QtWidgets>

class GraphsPlainModel;
class IGraphsModel;

class SubgEditValidator : public QValidator
{
    Q_OBJECT
public:
    explicit SubgEditValidator(QObject* parent = nullptr);
    ~SubgEditValidator();
    State validate(QString&, int&) const override;
    void fixup(QString&) const override;
};

class ZSubnetListItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    ZSubnetListItemDelegate(IGraphsModel* model, QObject* parent = nullptr);
    ~ZSubnetListItemDelegate();

    QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    void setEditorData(QWidget* editor, const QModelIndex& index) const override;
    void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
    void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const  QModelIndex& index) const override;

    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const override;
    bool editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option,
        const QModelIndex& index) override;

private slots:
    void onDelete(const QModelIndex& index);

private:
    IGraphsModel* m_model;
};


#endif
