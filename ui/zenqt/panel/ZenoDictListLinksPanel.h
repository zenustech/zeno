#ifndef __ZENO_DICTLIST_PANEL_H__
#define __ZENO_DICTLIST_PANEL_H__

#include <QtWidgets>

class IconDelegate : public QStyledItemDelegate {
    Q_OBJECT

public:
    explicit IconDelegate(bool bfirst, QObject* parent = nullptr);

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
private:
    bool m_bFirstColumn;
};

class DragDropModel : public QAbstractTableModel {
    Q_OBJECT

public:
    explicit DragDropModel(int row, int column, int allowDragColumn, QObject* parent = nullptr);
    //base
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    //drag
    Qt::DropActions supportedDropActions() const override;
    QMimeData* mimeData(const QModelIndexList& indexes) const override;
    bool dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) override;
private:
    int m_allowDragColumn;
    QList<QString> m_reorderedTexts;

    QStringList horizontalHeaderLabels;
    QVector<QVector<QString>> dataMatrix;
};

class ZenoDictListLinksTable : public QTableView {
    Q_OBJECT

public:
    ZenoDictListLinksTable(int row, int column, int allowDragColumn, QWidget* parent = nullptr);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private slots:
    void slt_clicked(const QModelIndex& index);
private:
    int m_allowDragColumn;
    DragDropModel* m_model;
};

#endif