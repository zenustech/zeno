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
    void setSelectedIndexs(const QModelIndexList& list);

    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const override;
    bool editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option,
        const QModelIndex& index) override;

public slots:
    void onDelete();
private slots:
    void onRename(const QModelIndex &index);
    void onSaveSubgraph(const QModelIndex& index);

private:
    void getSubgraphs(const QModelIndex& index, QModelIndexList& subgraphs);
    void setForkLock(QSortFilterProxyModel* model, bool bLocked);
    void setSubgraphType(QSortFilterProxyModel* model, bool bPreset);

private:
    IGraphsModel* m_model;
    QModelIndexList m_selectedIndexs;
};

class SubListSortProxyModel :public QSortFilterProxyModel
{
public:
    explicit SubListSortProxyModel(QObject* parent = nullptr);
protected:
    bool lessThan(const QModelIndex& source_left, const QModelIndex& source_right) const override;
    bool filterAcceptsRow(int source_row, const QModelIndex& source_parent) const override;
};

#endif
